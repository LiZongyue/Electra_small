from Electra_small.dataset import SupTextDataset
from Electra_small.modeling import ElectraForClassification
from Electra_small.runner import FineTuningRunner
from Electra_small.configs import ElectraFileConfig, ElectraTrainConfig
from torch.utils.data import DataLoader
from transformers import ElectraConfig, ElectraTokenizer


def data_loader(tokenizer, file_config, train_config):
    # DataLoader
    dataset_tr = SupTextDataset(tokenizer, file_config.train_data_file)
    dataset_val = SupTextDataset(tokenizer, file_config.validation_data_file)
    dataset_eval = SupTextDataset(tokenizer, file_config.eval_data_file)
    dataloader_tr = DataLoader(
        dataset_tr, shuffle=True, batch_size=train_config.batch_size_train, collate_fn=dataset_tr.collate_func,
        num_workers=4
    )
    dataloader_val = DataLoader(
        dataset_val, shuffle=False, batch_size=train_config.batch_size_val, collate_fn=dataset_val.collate_func,
        num_workers=4
    )

    dataloader_eval = DataLoader(
        dataset_eval, shuffle=False, batch_size=train_config.batch_size_val, collate_fn=dataset_val.collate_func,
        num_workers=4
    )
    data_len_tr = dataset_tr.__len__()
    data_len_val = dataset_val.__len__()
    data_len_eval = dataset_eval.__len__()
    return dataloader_tr, dataloader_val, dataloader_eval, data_len_tr, data_len_val, data_len_eval


class ElectraFineTuningRunner(FineTuningRunner):

    def __init__(self, model_config, train_config, file_config):
        super().__init__(model_config, train_config, file_config)
        self.tokenizer = ElectraTokenizer.from_pretrained('google/electra-base-discriminator')
        self.model = ElectraForClassification(model_config)
        self.model.electra.load_state_dict(self.state_to_load)
        self.param_opt = self.model.classifier.parameters()


def main():
    file_config = {
        "train_data_file": "C:/Users/Zongyue Li/Documents/GitHub/BNP/Data/aclImdb/train/data_100_train.csv",
        "validation_data_file": "C:/Users/Zongyue Li/Documents/GitHub/BNP/Data/aclImdb/train/data_100_val.csv",
        "eval_data_file": "C:/Users/Zongyue Li/Documents/GitHub/BNP/Data/aclImdb/train/data_24800_eval.csv",
        "import_path": "C:/Users/Zongyue Li/Documents/Github/BNP/output/Discriminator9.p",
    }

    model_config = {
        "embedding_size": 768,
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "intermediate_size": 3072,
        "num_labels": 1,
    }

    train_config = {
        "gpu_id": 0,  # gpu
        "learning_rate": 3e-5,
        "warmup_steps": 400,
        "n_epochs": 100,
        "batch_size_train": 32,
        "batch_size_val": 16,
        "train_head_epoch": 10,
        "lr_decay": 0.8,
    }

    file_config = ElectraFileConfig(**file_config)
    model_config = ElectraConfig(**model_config)
    train_config = ElectraTrainConfig(**train_config)

    finetune = ElectraFineTuningRunner(model_config, train_config, file_config)
    tokenizer = finetune.__tokenizer_getter__()

    dataloader_tr, dataloader_val, dataloader_eval, data_len_tr, data_len_val, data_len_eval =\
        data_loader(tokenizer=tokenizer, file_config=file_config, train_config=train_config)

    finetune.run(train_dataloader=dataloader_tr, validation_dataloader=dataloader_val, train_data_len=data_len_tr,
                 eval_dataloader=dataloader_eval)


if __name__ == '__main__':
    main()
