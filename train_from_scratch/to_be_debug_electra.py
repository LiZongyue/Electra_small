"""
Electra Small
"""

import os
import math
import numpy
import torch
import pickle
import matplotlib.pyplot as plt

from torch import nn
from tkinter import _flatten
from .config_electra import ElectraModelConfig, ElectraTrainConfig
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from transformers import ElectraTokenizer, ElectraForPreTraining, ElectraConfig, ElectraForMaskedLM, \
    get_linear_schedule_with_warmup, AdamW


class TextDataset(Dataset):
    """
    Subclass DataSet
    """

    def __init__(self, tokenizer, train_config, file_path: str, block_size=512):
        self.train_config = train_config
        
        assert os.path.isfile(file_path)

        block_size = block_size - (tokenizer.max_len - tokenizer.max_len_single_sentence)

        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            directory, "electra" + "_cached_lm_" + str(block_size) + "_" + filename
        )

        self.examples = []
        with open(file_path, encoding="utf-8") as f:
            text = f.read()
        # Open the file. Could be train data file, validation data file and evaluation data file

        text_line = text.split('\n')
        # Read the data line by line, then tokenize sentences and convert them to ids one by one

        tokenized_text_ = []
        for line in text_line:
            temp = tokenizer.tokenize(line)
            tokenized_text_.append(tokenizer.convert_tokens_to_ids(temp))

        tokenized_text = list(_flatten(tokenized_text_))
        # flatten the list to 1d array

        for i in range(0, len(tokenized_text) - block_size + 1, block_size):  # Truncate in block of block_size
            self.examples.append(tokenizer.build_inputs_with_special_tokens(tokenized_text[i: i + block_size]))

        # Note that we are loosing the last truncated example here for the sake of simplicity (no padding)
        # If your dataset is small, first you should look for a bigger one :-) and second you
        # can change this behavior by adding (model specific) padding.

        with open(cached_features_file, "wb") as handle:
            pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item], dtype=torch.long)

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

    def data_loader(self, tokenizer, dev, evaluate):
        # DataLoader
        dataset_ = self.load_and_cache_examples(tokenizer, dev, evaluate)
        sampler_ = RandomSampler(dataset_)
        dataloader = DataLoader(
            dataset_, sampler=sampler_, batch_size=self.train_config.batch_size
        )
        data_len = dataset_.__len__()

        return dataloader, data_len


class Electra(object):

    def __init__(self, model_config, train_config):
        self.model_config = model_config
        self.train_config = train_config

        self.tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator')

        self.config_generator = ElectraConfig(embedding_size=model_config.embedding_size,
                                              hidden_size=model_config.hidden_size_mlm,
                                              num_hidden_layers=model_config.num_hidden_layers_mlm,
                                              intermediate_size=model_config.intermediate_size_mlm)
        self.config_discriminator = ElectraConfig(embedding_size=model_config.embedding_size,
                                                  hidden_size=model_config.hidden_size_ce,
                                                  num_hidden_layers=model_config.num_hidden_layers_ce,
                                                  intermediate_size=model_config.intermediate_size_ce)

        self.generator = ElectraForMaskedLM(self.config_generator)
        self.discriminator = ElectraForPreTraining(self.config_discriminator)

        self.optimizer = None
        self.scheduler = None

        self.device = torch.device('cuda:{}'.format(self.train_config.gpu_id))

    def train_validation(self, train_dataloader, validation_dataloader):
        self.optimizer = self.init_optimizer(self.generator, self.discriminator, self.train_config.learning_rate)
        self.scheduler = self.scheduler_electra(self.optimizer)

        self.generator.to(self.device)
        self.discriminator.to(self.device)

        loss_train = []
        loss_validation = []
        for epoch_id in range(self.train_config.n_epoch):
            # Train
            self.generator.train()
            self.discriminator.train()
            for idx, data in enumerate(train_dataloader):
                loss_tr = self.train_one_step(epoch_id, idx, data)
                loss_train.append(loss_tr)

            with torch.no_grad():
                for idx, data in enumerate(validation_dataloader):
                    loss_val = self.validation_one_step(epoch_id, idx, data)
                    loss_validation.append(loss_val)

        return loss_train, loss_validation

    def train_one_step(self, epoch_id, idx, data, data_len_train):
        self.generator.train()
        self.discriminator.train()

        data = data.to(self.device)
        outputs_generator = self.generator(data, masked_lm_labels=data)
        loss_generator = outputs_generator[:1][0]

        labels_discriminator, input_discriminator = self.soft_max(data, outputs_generator,
                                                                  self.train_config.softmax_temperature)
        outputs_discriminator = self.discriminator(input_discriminator, labels=labels_discriminator)
        loss_discriminator = outputs_discriminator[:1][0]

        loss = loss_generator + self.train_config.lambda_ * loss_discriminator
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
        self.generator.eval()
        self.discriminator.eval()
        data = data.to(self.device)

        outputs_generator = self.generator(data, masked_lm_labels=data)
        loss_generator = outputs_generator[:1][0]

        labels_discriminator, input_discriminator = self.soft_max(data, outputs_generator,
                                                                  self.train_config.softmax_temperature)
        outputs_discriminator = self.discriminator(input_discriminator, labels=labels_discriminator)
        loss_discriminator = outputs_discriminator[:1][0]
        loss = loss_generator + self.train_config.lambda_ * loss_discriminator
        print(f'Epoch: {epoch_id + 1} | '
              f'batch: {idx + 1} / {math.ceil(data_len_validation / self.train_config.batch_size)} | '
              f'Validation Loss: {loss:.4f}')
        return loss.item()

    @staticmethod
    def soft_max(input_data, output_data, softmax_temperature):
        m = nn.Softmax(dim=1)
        output_softmax = torch.distributions.Categorical(
            m(output_data[1] / softmax_temperature)).sample()  # get output_IDs of model_mlm by applyng sampling.
        labels_ce = 1 - torch.eq(input_data, output_softmax).int()
        return labels_ce, output_softmax

    def scheduler_electra(self, optimizer):  # , data_len, batch_size):
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.train_config.warmup_steps,
            num_training_steps=self.train_config.num_training_steps
        )
        return scheduler

    @staticmethod
    def init_optimizer(model1, model2, learning_rate):
        #  Initialize Optimizer AdamW
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model1.named_parameters() if not any(nd in n for nd in no_decay)] +
                          [p for n, p in model2.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
            {"params": [p for n, p in model1.named_parameters() if any(nd in n for nd in no_decay)] +
                       [p for n, p in model2.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=1e-8)

        return optimizer

    @staticmethod
    def plot_loss(loss_train, loss_validation):
        plt.plot(loss_train)
        plt.plot(loss_validation)
        plt.show()


def main():
    train_data_file = "../wiki.train.raw",
    validation_data_file = "../wiki.valid.raw",
    eval_data_file = "../wiki.test.raw",

    model_config = {
        "embedding_size": 64,
        "hidden_size_mlm": 64,
        "hidden_size_ce": 128,
        "num_hidden_layers_mlm": 3,
        "num_hidden_layers_ce": 6,
        "intermediate_size_mlm": 256,
        "intermediate_size_ce": 512,
    }

    train_config = {
        "gpu_id": 0,  # gpu
        "learning_rate": 1e-5,
        "warmup_steps": 10,
        "n_epochs": 50,
        "batch_size": 8,
        "softmax_temperature": 1,
        "lambda_": 50,
    }

    model_config = ElectraModelConfig(**model_config)
    train_config = ElectraTrainConfig(**train_config)


    electra = Electra(model_config, train_config)
    loss_train, loss_validation = electra.train_validation()
    '''
    optimizer_mlm_, optimizer_ce_ = init_optimizer(model_mlm=model_mlm_, model_ce=model_ce_, learning_rate=lr)
    scheduler_mlm_, scheduler_ce_ = scheduler(optimizer_mlm=optimizer_mlm_, optimizer_ce=optimizer_ce_)
    # , data_len = data_len_, batch_size = batch_size_)
    loss_train_, loss_validation_ = train_validation(model_mlm=model_mlm_, model_ce=model_ce_,
                                                     train_dataloader=train_dataloader_,
                                                     validation_dataloader=validation_dataloader_,
                                                     optimizer_mlm=optimizer_mlm_,
                                                     optimizer_ce=optimizer_ce_,
                                                     data_len_train=data_len_train_,
                                                     data_len_validation=data_len_validation_, batch_size=batch_size_,
                                                     epoch=epoch_, lambda_=lambda__,
                                                     softmax_temperature=softmax_temperature_,
                                                     scheduler_mlm=scheduler_mlm_, scheduler_ce=scheduler_ce_)
    # scheduler_mlm = scheduler_mlm_, scheduler_ce = scheduler_ce_,
    Electra.plot_loss(loss_train_, loss_validation_)

    '''


if __name__ == "__main__":
    main()
