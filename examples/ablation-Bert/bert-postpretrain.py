from Electra_small.runner import BertRunner
from Electra_small.dataset import TextDataset
from Electra_small.modeling import BertBase
from torch.utils.data import DataLoader, RandomSampler
from transformers import BertTokenizer
from Electra_small.configs import ElectraFileConfig, ElectraTrainConfig


class Pft_Dataset(TextDataset):
    """
    Subclass TextDataSet
    """

    def __init__(self, file_path: str, train_config):
        super().__init__(file_path, train_config)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def data_loader(file_config, train_config):
    # DataLoader
    dataset_tr = Pft_Dataset(file_config.train_data_file, train_config)
    sampler = RandomSampler(dataset_tr)
    dataset_val = Pft_Dataset(file_config.validation_data_file, train_config)
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
    return dataloader_tr, dataloader_val, data_len_tr, data_len_val


def main():
    file_config = {
        "train_data_file": "C:/Users/Zongyue Li/Documents/GitHub/BNP/Data/aclImdb/train/unsup/train.txt",
        "validation_data_file": "C:/Users/Zongyue Li/Documents/GitHub/BNP/Data/aclImdb/train/unsup/val.txt",
        "eval_data_file": None,
        "save_path": "C:/Users/Jackie/Documents/GitHub/output/Bert{}.p",

    }

    train_config = {
        "gpu_id": 0,  # gpu
        "learning_rate": 2e-5,
        "warmup_steps": 20000,
        "n_epochs": 50,
        "batch_size_train": 32,
        "batch_size_val": 16,
        "max_length": 128,
    }

    file_config = ElectraFileConfig(**file_config)
    train_config = ElectraTrainConfig(**train_config)

    train_data_loader, valid_data_loader, train_data_len, valid_data_len = data_loader(file_config, train_config)
    bert_base = BertBase(train_config)
    runner = BertRunner(bert_base, train_config, file_config)

    loss_train, loss_validation = runner.train_validation(train_dataloader=train_data_loader,
                                                          validation_dataloader=valid_data_loader,
                                                          data_len_train=train_data_len,
                                                          data_len_validation=valid_data_len)

    runner.plot_loss(loss_train=loss_train, loss_validation=loss_validation)


if __name__ == "__main__":
    main()
