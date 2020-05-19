import os
import torch

from tqdm import tqdm
from termcolor import colored
from Electra_small.runner import Runner
from Electra_small.modeling import Electra
from Electra_small.dataset import TextDataset
from torch.utils.data import DataLoader, RandomSampler
from transformers import ElectraTokenizer, ElectraForPreTraining, ElectraForMaskedLM
from Electra_small.configs import ElectraFileConfig, ElectraModelConfig, ElectraTrainConfig


class Pft_Dataset(TextDataset):
    """
    Subclass DataSet
    """

    def __init__(self, file_path: str, train_config):
        super().__init__(train_config)

        files = os.listdir(file_path)  # get all files under the dir
        txts = []
        print(colored("Pre processing the data...", "red"))
        for file in tqdm(files):  # iterate the dir
            position = file_path + '/' + file  # construct path with "/"
            with open(position, "r", encoding='utf-8') as f:  # open file
                data = f.read()  # read file
                data = data.replace('<br />', '')
                txts.append(data)

        self.examples = txts
        self.tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator')


def data_loader(file_config, train_config):
    # DataLoader
    dataset_tr = Pft_Dataset(file_config.train_data_file, train_config)
    sampler = RandomSampler(dataset_tr)
    dataset_val = Pft_Dataset(file_config.validation_data_file)
    dataloader_tr = DataLoader(
        dataset_tr, sampler=sampler, batch_size=train_config.batch_size_train, collate_fn=dataset_tr.collate_func,
        num_workers=4
    )
    dataloader_val = DataLoader(
        dataset_val, shuffle=False, batch_size=train_config.batch_size_val, collate_fn=dataset_val.collate_func,
        num_workers=4
    )
    data_len_tr = dataset_tr.__len__()
    data_len_val = dataset_val.__len__()
    return dataloader_tr, dataloader_val, data_len_val, data_len_tr


class ElectraBase(Electra):

    def __init__(self, model_config, train_config):
        super().__init__(model_config, train_config)

        self._generator_ = ElectraForMaskedLM.from_pretrained("google/electra-small-generator")
        self._discriminator_ = ElectraForPreTraining.from_pretrained("google/electra-small-discriminator")

        self._tie_embedding()
        self.device = torch.device('cuda:{}'.format(self.train_config.gpu_id)) if torch.cuda.is_available() else 'cpu'


def main():
    file_config = {
        "train_data_file": "/Users/Jackie/Documents/GitHub/Data/aclImdb/train/unsup",
        "validation_data_file": None,
        "eval_data_file": None,
        "save_path": "/Users/Jackie/Documents/GitHub/output/Discriminator{}.p",
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
        "batch_size_train": 1,
        "batch_size_val": 4,
        "softmax_temperature": 1,
        "lambda_": 50,
    }

    file_config = ElectraFileConfig(**file_config)
    model_config = ElectraModelConfig(**model_config)
    train_config = ElectraTrainConfig(**train_config)

    train_data_loader, valid_data_loader, train_data_len, valid_data_len = data_loader(file_config, train_config)
    electra_small = ElectraBase(model_config, train_config)
    runner = Runner(electra_small, train_config, file_config)

    valid_data_loader = None
    valid_data_len = None
    loss_train, loss_validation = runner.train_validation(train_dataloader=train_data_loader,
                                                          validation_dataloader=valid_data_loader,
                                                          data_len_train=train_data_len,
                                                          data_len_validation=valid_data_len)

    runner.plot_loss(loss_train=loss_train, loss_validation=loss_validation)


if __name__ == "__main__":
    main()
