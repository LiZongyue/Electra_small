import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from torch.nn.utils.rnn import pad_sequence
from Electra_small.modeling import ElectraForClassification
from Electra_small.configs import ElectraFileConfig, ElectraTrainConfig
from torch.utils.data import DataLoader, Dataset, RandomSampler
from transformers import ElectraConfig, ElectraTokenizer, get_linear_schedule_with_warmup, AdamW


def data_loader(tokenizer, file_config, train_config):
    # DataLoader
    dataset_tr = TextDataset(tokenizer, file_config.train_data_file)
    dataset_val = TextDataset(tokenizer, file_config.validation_data_file)
    dataloader_tr = DataLoader(
        dataset_tr, shuffle=True, batch_size=train_config.batch_size_train, collate_fn=collate_func, num_workers=4
    )
    dataloader_val = DataLoader(
        dataset_val, shuffle=False, batch_size=train_config.batch_size_val, collate_fn=collate_func, num_workers=4
    )
    data_len_tr = dataset_tr.__len__()
    data_len_val = dataset_val.__len__()
    return dataloader_tr, dataloader_val, data_len_tr, data_len_val


def collate_func(batch):
    x, y = zip(*batch)
    x_padded = pad_sequence(x, batch_first=True)
    y_padded = []
    for item in y:
        y_padded.append(item.long())
    y_padded = torch.tensor(y_padded)
    return [x_padded, y_padded]


class TextDataset(Dataset):
    """
    Subclass DataSet
    """

    def __init__(self, tokenizer, file_path: str, block_size=64):
        assert os.path.isfile(file_path)

        train_data = pd.read_csv(file_path, sep='\t')
        lines = train_data['sentence'].tolist()

        batch_encoding = tokenizer.batch_encode_plus(lines, add_special_tokens=True, max_length=block_size,
                                                     pad_to_max_length=False)
        self.examples = []
        self.examples = batch_encoding["input_ids"]
        self.labels = np.array(train_data['label'].tolist())

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return torch.tensor(self.examples[index], dtype=torch.long), torch.tensor(self.labels[index], dtype=torch.long)


class SST2Runner(object):

    def __init__(self, model_config, train_config):
        self.model_config = model_config
        self.train_config = train_config

        self.tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator')

        self.electraforclassification = ElectraForClassification(model_config)

        pre_trained_weights = torch.load("C:/Users/Zongyue Li/Documents/Github/BNP/output/D_4.p",
                                         map_location=torch.device('cpu'))
        state_to_load = {}
        for name, param in pre_trained_weights.items():
            if name.startswith('electra.'):
                state_to_load[name[len('electra.'):]] = param

        self.electraforclassification.electra.load_state_dict(state_to_load)
        self.optimizer = None
        self.scheduler = None

        self.device = torch.device('cuda:{}'.format(self.train_config.gpu_id)) if torch.cuda.is_available() else 'cpu'

    def __tokenizer_getter__(self):
        return self.tokenizer

    @staticmethod
    def sigmoid(logits):
        return 1 / (1 + np.exp(-logits))

    def run(self, train_dataloader, validation_dataloader, train_data_len):
        for epoch_id in range(self.train_config.n_epochs):
            mean_loss_train, mean_loss_val, acc_tr, acc_val = self.train_validation(train_dataloader,
                                                                                    validation_dataloader,
                                                                                    train_data_len)
            print(
                f"Epoch {epoch_id}: train_loss={mean_loss_train:.4f}, train_acc={acc_tr:.4f}, "
                f"valid_loss={mean_loss_val:.4f}, val_acc={acc_val:.4f}")

    def train_validation(self, train_dataloader, validation_dataloader, train_data_len):
        self.optimizer = self.init_optimizer(self.electraforclassification, self.train_config.learning_rate)
        self.scheduler = self.scheduler_electra(self.optimizer, train_data_len)

        self.electraforclassification.to(self.device)

        logits_train = []
        logits_val = []
        labels_train = []
        labels_val = []
        loss_train = []
        loss_validation = []
        # Train
        self.electraforclassification.train()
        for idx, data in enumerate(train_dataloader):
            label_tr, loss_tr, logits_tr = self.train_one_step(data)
            loss_train.append(loss_tr)
            logits_train.append(logits_tr.detach().cpu().numpy())
            labels_train.append(label_tr.cpu().numpy())
        with torch.no_grad():
            for idx, data in enumerate(validation_dataloader):
                label_val, loss_val, logits_val = self.validation_one_step(data)
                loss_validation.append(loss_val)
                logits_val.append(logits_val.detach().cpu().numpy())
                labels_val.append(label_val.cpu().numpy())

        mean_loss_train = np.mean(np.array(loss_train))
        mean_loss_val = np.mean(np.array(loss_validation))
        acc_tr = accuracy_score(labels_train, (self.sigmoid(logits_train) > 0.5).astype(int))
        acc_val = accuracy_score(labels_val, (self.sigmoid(logits_val) > 0.5).astype(int))

        return mean_loss_train, mean_loss_val, acc_tr, acc_val

    def train_one_step(self, data):
        self.electraforclassification.train()

        labels, loss, logits = self.process_model(data)
        # Autograd
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        return labels, loss.item(), logits

    def validation_one_step(self, data):
        self.electraforclassification.eval()

        labels, loss, logits = self.process_model(data)
        return labels, loss.item(), logits

    def process_model(self, data):
        example_input, example_labels = data
        example_input = example_input.to(self.device)
        example_labels = example_labels.to(self.device)
        # example_input = torch.randint(0, 30522, (3, 10)).long(),  input shape (bs, seq_len)
        # example_labels = # torch.randint(0, 3, (3,)).long()  # labels shape (bs, )
        scores = self.electraforclassification(example_input)  # output scores/logits shape (bs, num_labels)
        scores = scores.to(self.device)
        loss = self.electraforclassification.get_loss(scores, example_labels)
        return example_labels, loss, scores

    def scheduler_electra(self, optimizer, train_data_len):  # , data_len, batch_size):
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.train_config.warmup_steps,
            num_training_steps=int(train_data_len / self.train_config.batch_size_train * self.train_config.n_epochs)
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
    file_config = {
        "train_data_file": "C:/Users/Zongyue Li/Documents/Github/BNP/Data/glue_data/SST-2/train.tsv",
        "validation_data_file": "C:/Users/Zongyue Li/Documents/Github/BNP/Data/glue_data/SST-2/dev.tsv",
        "eval_data_file": "C:/Users/Zongyue Li/Documents/Github/BNP/Data/glue_data/SST-2/test.tsv",
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
        "n_epochs": 100,
        "batch_size_train": 32,
        "batch_size_val": 128
    }

    file_config = ElectraFileConfig(**file_config)
    model_config = ElectraConfig(**model_config)
    train_config = ElectraTrainConfig(**train_config)

    sst2 = SST2Runner(model_config, train_config)
    tokenizer = sst2.__tokenizer_getter__()

    dataloader_tr, dataloader_val, data_len_tr, data_len_val = data_loader(tokenizer=tokenizer,
                                                                           file_config=file_config,
                                                                           train_config=train_config)

    sst2.run(train_dataloader=dataloader_tr, validation_dataloader=dataloader_val, train_data_len=data_len_tr)


if __name__ == '__main__':
    main()
