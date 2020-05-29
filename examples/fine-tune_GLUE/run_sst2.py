from Electra_small.modeling import ElectraForClassification
from Electra_small.dataset import SupTextDataset
from Electra_small.configs import ElectraFileConfig, ElectraTrainConfig
from Electra_small.runner import FineTuningRunner
from torch.utils.data import DataLoader
from transformers import ElectraConfig, ElectraTokenizer, AdamW


def data_loader(tokenizer, file_config, train_config):
    # DataLoader
    dataset_tr = SupTextDataset(tokenizer, file_config.train_data_file)
    dataset_val = SupTextDataset(tokenizer, file_config.validation_data_file)
    dataloader_tr = DataLoader(
        dataset_tr, shuffle=True, batch_size=train_config.batch_size_train, collate_fn=dataset_tr.collate_func,
        num_workers=4
    )
    dataloader_val = DataLoader(
        dataset_val, shuffle=False, batch_size=train_config.batch_size_val, collate_fn=dataset_val.collate_func,
        num_workers=4
    )

    data_len_tr = dataset_tr.__len__()
    data_len_val = dataset_val.__len__()
    return dataloader_tr, dataloader_val, data_len_tr, data_len_val


class SST2Runner(FineTuningRunner):

    def __init__(self, model_config, train_config, file_config):
        super().__init__(model_config, train_config, file_config)

        self.tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator')
        self.model = ElectraForClassification(model_config)

        self.model.electra.load_state_dict(self.state_to_load)
        self.param_opt = self.model.classifier.parameters()

    def run(self, train_dataloader, validation_dataloader, train_data_len, eval_dataloader):
        # train head
        for epoch_id in range(self.train_config.train_head_epoch):
            optimizer = AdamW(list(self.param_opt),
                              lr=self.train_config.learning_rate)
            mean_loss_train, mean_loss_val, acc_tr, acc_val = self.train_validation(train_dataloader,
                                                                                    validation_dataloader,
                                                                                    train_data_len, False,
                                                                                    optimizer)
            print(
                f"Epoch {epoch_id}: train_head_loss={mean_loss_train:.4f}, train_head_acc={acc_tr:.4f}, "
                f"valid_head_loss={mean_loss_val:.4f}, val_head_acc={acc_val:.4f}")

        # train all
        num_layers = len(self.model.layers)
        param_optimizer = list(self.model.named_parameters())
        optimizer_parameters = self.get_layer_wise_lr_decay(num_layers, param_optimizer,
                                                            self.train_config.learning_rate,
                                                            self.train_config.lr_decay)
        optimizer = AdamW(optimizer_parameters, lr=self.train_config.learning_rate, eps=1e-8, weight_decay=0.01)
        for epoch_id in range(self.train_config.n_epochs):
            mean_loss_train, mean_loss_val, acc_tr, acc_val = self.train_validation(train_dataloader,
                                                                                    validation_dataloader,
                                                                                    train_data_len, True, optimizer)
            print(
                f"Epoch {epoch_id}: train_loss={mean_loss_train:.4f}, train_acc={acc_tr:.4f}, "
                f"valid_loss={mean_loss_val:.4f}, val_acc={acc_val:.4f}")


def main():
    file_config = {
        "train_data_file": "C:/Users/Zongyue Li/Documents/Github/BNP/Data/glue_data/SST-2/train.tsv",
        "validation_data_file": "C:/Users/Zongyue Li/Documents/Github/BNP/Data/glue_data/SST-2/dev.tsv",
        "eval_data_file": "C:/Users/Zongyue Li/Documents/Github/BNP/Data/glue_data/SST-2/test.tsv",
        "import_path": "C:/Users/Zongyue Li/Documents/Github/BNP/output/D_1.p",
    }

    model_config = {
        "embedding_size": 128,
        "hidden_size": 256,
        "num_hidden_layers": 12,
        "intermediate_size": 1024,
        "num_labels": 1,
    }

    train_config = {
        "gpu_id": 0,  # gpu
        "learning_rate": 3e-4,
        "warmup_steps": 400,
        "n_epochs": 3,
        "batch_size_train": 32,
        "batch_size_val": 128,
        "train_head_epoch": 3,
        "lr_decay": 0.8,
    }

    file_config = ElectraFileConfig(**file_config)
    model_config = ElectraConfig(**model_config)
    train_config = ElectraTrainConfig(**train_config)

    sst2 = SST2Runner(model_config, train_config, file_config)
    tokenizer = sst2.__tokenizer_getter__()

    dataloader_tr, dataloader_val, data_len_tr, data_len_val = \
        data_loader(tokenizer=tokenizer, file_config=file_config, train_config=train_config)

    sst2.run(train_dataloader=dataloader_tr, validation_dataloader=dataloader_val, train_data_len=data_len_tr,
             eval_dataloader=None)


if __name__ == '__main__':
    main()
