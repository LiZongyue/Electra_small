"""
Electra Small
"""
import os
import torch

from Electra_small.runner import Runner
from Electra_small.modeling import Electra
from Electra_small.dataset import TextDataset
from Electra_small.configs import ElectraFileConfig, ElectraModelConfig, ElectraTrainConfig
from torch.utils.data import DataLoader, RandomSampler
from transformers import ElectraForPreTraining, ElectraTokenizer, ElectraConfig, ElectraForMaskedLM


class PreTrainDataset(TextDataset):
    """
    Subclass DataSet
    """

    def __init__(self, file_path: str, train_config):
        super().__init__(train_config)
        assert os.path.isfile(file_path)
        with open(file_path, encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]
        res_lines = [item for item in lines if not (item.startswith(' ='))]

        self.examples = res_lines
        self.tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator')


def load_and_cache_examples(train_config, train_data_file, validation_data_file,
                            eval_data_file, dev=False, evaluate=False):
    # Load and cache examples for different dataset
    flag_train = False
    if evaluate:
        file_path = eval_data_file
    else:
        if dev:
            file_path = validation_data_file
        else:
            file_path = train_data_file
            flag_train = True
    return PreTrainDataset(file_path, train_config), flag_train


def data_loader(file_config, train_config, dev, evaluate):
    # DataLoader
    dataset_, flag_train = load_and_cache_examples(train_config, file_config.train_data_file, file_config.validation_data_file,
                                                   file_config.eval_data_file, dev, evaluate)
    sampler_ = RandomSampler(dataset_)
    if flag_train:
        dataloader = DataLoader(
            dataset_, sampler=sampler_, batch_size=train_config.batch_size_train, collate_fn=dataset_.collate_func,
            num_workers=4
        )
    else:
        dataloader = DataLoader(
            dataset_, shuffle=False, batch_size=train_config.batch_size_val, collate_fn=dataset_.collate_func,
            num_workers=4
        )
    data_len = dataset_.__len__()

    return dataloader, data_len


class ElectraSmall(Electra):
    _generator_: ElectraForMaskedLM
    _discriminator_: ElectraForPreTraining

    def __init__(self, model_config, train_config):
        super().__init__(train_config)

        self.config_generator = ElectraConfig(embedding_size=model_config.embedding_size,
                                              hidden_size=model_config.hidden_size_mlm,
                                              num_hidden_layers=model_config.num_hidden_layers_mlm,
                                              intermediate_size=model_config.intermediate_size_mlm,
                                              num_attention_heads=model_config.attention_heads_mlm)
        self.config_discriminator = ElectraConfig(embedding_size=model_config.embedding_size,
                                                  hidden_size=model_config.hidden_size_ce,
                                                  num_hidden_layers=model_config.num_hidden_layers_ce,
                                                  intermediate_size=model_config.intermediate_size_ce,
                                                  num_attention_heads=model_config.attention_heads_ce)

        self._generator_ = ElectraForMaskedLM(self.config_generator)
        self._discriminator_ = ElectraForPreTraining(self.config_discriminator)

        self._tie_embedding()
        self.device = torch.device('cuda:{}'.format(self.train_config.gpu_id)) if torch.cuda.is_available() else 'cpu'


def main():
    file_config = {
        "train_data_file": "C:/Users/Zongyue Li/Documents/Github/BNP/Data/wiki.train.raw",
        "validation_data_file": "C:/Users/Zongyue Li/Documents/Github/BNP/Data/wiki.valid.raw",
        "eval_data_file": "C:/Users/Zongyue Li/Documents/Github/BNP/Data/wiki.test.raw",
        "save_path": None
    }

    model_config = {
        "embedding_size": 128,
        "hidden_size_mlm": 64,
        "hidden_size_ce": 256,
        "num_hidden_layers_mlm": 12,
        "num_hidden_layers_ce": 12,
        "intermediate_size_mlm": 256,
        "intermediate_size_ce": 1024,
        "attention_heads_mlm": 1,
        "attention_heads_ce": 4,
    }

    train_config = {
        "gpu_id": 0,  # gpu
        "learning_rate": 5e-4,
        "warmup_steps": 10000,
        "n_epochs": 50,
        "batch_size_train": 128,
        "batch_size_val": 4,
        "softmax_temperature": 1,
        "lambda_": 50,
        "add_special_tokens": True,
        "max_length": 128
    }

    file_config = ElectraFileConfig(**file_config)
    model_config = ElectraModelConfig(**model_config)
    train_config = ElectraTrainConfig(**train_config)

    electra_small = ElectraSmall(model_config, train_config)

    train_data_loader, train_data_len = data_loader(file_config=file_config, train_config=train_config, dev=False,
                                                    evaluate=False)
    valid_data_loader, valid_data_len = data_loader(file_config=file_config, train_config=train_config, dev=True,
                                                    evaluate=False)

    runner = Runner(electra_small, train_config, file_config)

    loss_train, loss_validation = runner.train_validation(train_dataloader=train_data_loader,
                                                          validation_dataloader=valid_data_loader,
                                                          data_len_train=train_data_len,
                                                          data_len_validation=valid_data_len)

    runner.plot_loss(loss_train=loss_train, loss_validation=loss_validation)


if __name__ == "__main__":
    main()
