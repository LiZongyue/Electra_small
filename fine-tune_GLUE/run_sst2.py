import torch
from torch import nn
import os
import math
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ELectra_small.modeling import ElectraForClassification
from ELectra_small.configs import ElectraTrainConfig
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from transformers import ElectraConfig, ElectraTokenizer, ElectraForMaskedLM, get_linear_schedule_with_warmup, AdamW


class TextDataset(Dataset):
    """
    Subclass DataSet
    """

    def __init__(self, tokenizer, file_path: str, block_size=128):
        assert os.path.isfile(file_path)

        train_data = pd.read_csv(file_path, sep='\t')
        lines = train_data['sentence'].tolist()
        # Open the file. Could be train data file, validation data file and evaluation data file

        batch_encoding = tokenizer.batch_encode_plus(lines, add_special_tokens=True, max_length=block_size)

        self.examples = []
        self.examples = batch_encoding["input_ids"]
        self.labels = np.array(train_data['label'].tolist()).reshape(-1, 1).tolist()

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return torch.tensor(self.examples[index], dtype=torch.long), torch.tensor(self.labels[index], dtype=torch.long)

    @staticmethod
    def load_and_cache_examples(tokenizer, train_data_file, validation_data_file,
                                eval_data_file, dev=False, evaluate=False):
        # Load and cache examples for different dataset
        if evaluate:
            file_path = eval_data_file
        else:
            if dev:
                file_path = validation_data_file
            else:
                file_path = train_data_file
        return TextDataset(tokenizer, file_path=file_path)

    @classmethod
    def data_loader(cls, tokenizer, train_config, train_data_file, validation_data_file, eval_data_file, dev, evaluate):
        # DataLoader
        dataset_ = cls.load_and_cache_examples(tokenizer, train_data_file, validation_data_file,
                                               eval_data_file, dev, evaluate)
        sampler_ = RandomSampler(dataset_)
        dataloader = DataLoader(
            dataset_, sampler=sampler_, batch_size=train_config.batch_size
        )
        data_len = dataset_.__len__()

        return dataloader, data_len


class SST2Runner(object):

    def __init__(self, model_config, train_config):
        self.model_config = model_config
        self.train_config = train_config

        self.tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator')

        self.electraforclassification = ElectraForClassification(model_config)
        self.electraforclassification.load_electra_weights("/Users/cpius/Downloads/electra_state_dict.p")
        # TODO: change the data file

        self.optimizer = None
        self.scheduler = None

        self.device = torch.device('cuda:{}'.format(self.train_config.gpu_id))

    def __tokenizer_getter__(self):
        return self.tokenizer

    def train_validation(self, train_dataloader, validation_dataloader, data_len_train, data_len_validation):
        self.optimizer = self.init_optimizer(self.electraforclassification, self.train_config.learning_rate)
        self.scheduler = self.scheduler_electra(self.optimizer)

        self.electraforclassification.to(self.device)

        loss_train = []
        loss_validation = []
        for epoch_id in range(self.train_config.n_epochs):
            # Train
            self.electraforclassification.train()

            for idx, data in enumerate(train_dataloader):
                loss_tr = self.train_one_step(epoch_id, idx, data, data_len_train)
                loss_train.append(loss_tr)

            with torch.no_grad():
                for idx, data in enumerate(validation_dataloader):
                    loss_val = self.validation_one_step(epoch_id, idx, data, data_len_validation)
                    loss_validation.append(loss_val)

        return loss_train, loss_validation

    def train_one_step(self, epoch_id, idx, data, data_len_train):
        self.electraforclassification.train()

        data = data.to(self.device)
        loss = self.process_model(data)
        print(f'Epoch: {epoch_id + 1} | '
              f'batch: {idx + 1} / {math.ceil(data_len_train / self.train_config.batch_size)} | '
              f'Train Loss: {loss:.4f}')
        # Autograd
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        return loss.item()

    def validation_one_step(self, epoch_id, idx, data, data_len_validation):
        self.electraforclassification.eval()
        data = data.to(self.device)
        loss = self.process_model(data)
        print(f'Epoch: {epoch_id + 1} | '
              f'batch: {idx + 1} / {math.ceil(data_len_validation / self.train_config.batch_size)} | '
              f'Validation Loss: {loss:.4f}')
        return loss.item()

    def process_model(self, data):
        example_input, example_labels = data
        # example_input = torch.randint(0, 30522, (3, 10)).long(),  input shape (bs, seq_len)
        # example_labels = # torch.randint(0, 3, (3,)).long()  # labels shape (bs, )
        scores = self.electraforclassification(example_input)  # output scores/logits shape (bs, num_labels)
        loss = self.electraforclassification.get_loss(scores, example_labels)
        return loss

    def scheduler_electra(self, optimizer):  # , data_len, batch_size):
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.train_config.warmup_steps,
            num_training_steps=self.train_config.num_training_steps
        )
        return scheduler

    @staticmethod
    def init_optimizer(model, learning_rate):
        #  Initialize Optimizer AdamW
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=1e-8)

        return optimizer


def main():
    train_data_file = "C:/Users/Zongyue Li/Documents/Github/BNP/Electra_small/Data/wiki.train.raw"
    validation_data_file = "C:/Users/Zongyue Li/Documents/Github/BNP/Electra_small/Data/wiki.valid.raw"
    eval_data_file = "C:/Users/Zongyue Li/Documents/Github/BNP/Electra_small/Data/wiki.test.raw"

    model_config = {
        "embedding_size": 128,
        "hidden_size": 128,
        "num_hidden_layers": 6,
        "intermediate_size": 512,
        "num_labels": 3,
    }

    train_config = {
        "gpu_id": 0,  # gpu
        "learning_rate": 1e-3,
        "warmup_steps": 10,
        "n_epochs": 50,
        "batch_size": 8,
        "softmax_temperature": 1,
        "lambda_": 50,
    }

    model_config = ElectraConfig(**model_config)
    train_config = ElectraTrainConfig(**train_config)

    sst2 = SST2Runner(model_config, train_config)
    tokenizer = sst2.__tokenizer_getter__()

    train_data_loader, train_data_len = TextDataset.data_loader(tokenizer=tokenizer, train_config=train_config,
                                                                train_data_file=train_data_file,
                                                                validation_data_file=validation_data_file,
                                                                eval_data_file=eval_data_file,
                                                                dev=False, evaluate=False)
    valid_data_loader, valid_data_len = TextDataset.data_loader(tokenizer=tokenizer, train_config=train_config,
                                                                train_data_file=train_data_file,
                                                                validation_data_file=validation_data_file,
                                                                eval_data_file=eval_data_file,
                                                                dev=True, evaluate=False)

    loss_train, loss_validation = sst2.train_validation(train_dataloader=train_data_loader,
                                                        validation_dataloader=valid_data_loader,
                                                        data_len_train=train_data_len,
                                                        data_len_validation=valid_data_len)


if __name__ == '__main__':
    main()