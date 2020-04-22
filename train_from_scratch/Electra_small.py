# -*- coding: utf-8 -*-
"""Electra Small

Original file is located at
    https://colab.research.google.com/drive/1JA2-lpO-H818ByT1H6RdNAnBieBUevBZ
"""

# Commented out IPython magic to ensure Python compatibility.
# %cd drive/My\ Drive/transformers-master/

# pip install transformers

# pip install .

# pip install -r ./examples/requirements.txt

import os
import math
import numpy
import torch
import pickle
import matplotlib.pyplot as plt

from torch import nn
from tkinter import _flatten
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from transformers import ElectraTokenizer, ElectraForPreTraining, ElectraConfig, ElectraForMaskedLM, \
    get_linear_schedule_with_warmup, AdamW


def init_():
    """
    A pretrained Tokenizer and a basic Electra model will be initialized once the function is called.
    Electra includes 2 parts, one is for languages generation, which is called electra_mlm,
    another is for tokenprediction, which is called electra_ce.
    :return:
    """
    tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator')

    configuration_mlm = ElectraConfig(embedding_size=64, hidden_size=64, num_hidden_layers=3, intermediate_size=256)
    configuration_ce = ElectraConfig(embedding_size=64, hidden_size=128, num_hidden_layers=6, intermediate_size=512)

    model_mlm = ElectraForMaskedLM(configuration_mlm)
    model_ce = ElectraForPreTraining(configuration_ce)

    # TODO: Parameters to setup the model should be passed instead assigned manually inside the function.
    return tokenizer, model_mlm, model_ce


class TextDataset(Dataset):
    """
    Subclass DataSet
    """

    def __init__(self, tokenizer, file_path: str, block_size=512):
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

        # flatten the list to 1d array
        tokenized_text = list(_flatten(tokenized_text_))

        print('tokenized!')
        for i in range(0, len(tokenized_text) - block_size + 1, block_size):  # Truncate in block of block_size
            self.examples.append(tokenizer.build_inputs_with_special_tokens(tokenized_text[i: i + block_size]))

        # Note that we are loosing the last truncated example here for the sake of simplicity (no padding)
        # If your dataset is small, first you should loook for a bigger one :-) and second you
        # can change this behavior by adding (model specific) padding.

        with open(cached_features_file, "wb") as handle:
            pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item], dtype=torch.long)


def load_and_cache_examples(tokenizer, dev=False, evaluate=False):
    # Load and cache examples for different dataset
    if evaluate:
        file_path = eval_data_file
    else:
        if dev:
            file_path = validation_data_file
        else:
            file_path = train_data_file
    return TextDataset(tokenizer, file_path=file_path)


def data_loader(tokenizer, batch_size, dev, evaluate):
    # DataLoader
    dataset_ = load_and_cache_examples(tokenizer, dev, evaluate)
    sampler_ = RandomSampler(dataset_)
    dataloader = DataLoader(
        dataset_, sampler=sampler_, batch_size=batch_size
    )
    data_len = dataset_.__len__()

    return dataloader, data_len


def init_optimizer(model_mlm, model_ce, learning_rate):
    #  Initialize 2 Optimizers AdamW for both of the 2 models.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model_mlm.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
        {"params": [p for n, p in model_mlm.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer_mlm = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=1e-8)

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model_ce.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
        {"params": [p for n, p in model_ce.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer_ce = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=1e-8)
    # TODO: the learning_rate of 2 models would be different. Check it in the papar.
    return optimizer_mlm, optimizer_ce


# Use Softmax Function to get the output_ids of model_mlm. Create labels for model_ce
m = nn.Softmax(dim=1)


def soft_max(input_data, output_data):
    output_softmax = torch.distributions.Categorical(m(output_data[1])).sample()  # get output_IDs of model_mlm
    labels_ce = 1 - torch.eq(input_data, output_softmax).int()
    return labels_ce, output_softmax


'''
def scheduler(optimizer_mlm, optimizer_ce):#, data_len, batch_size):

    scheduler_mlm = get_linear_schedule_with_warmup(
        optimizer_mlm, num_warmup_steps = 20, num_training_steps=100
    )

    scheduler_ce = get_linear_schedule_with_warmup(
        optimizer_ce, num_warmup_steps = 20, num_training_steps=100
    )

    return scheduler_mlm, scheduler_ce
'''


# Plot Loss to choose model_ce without over/underfitting

def plot_loss(loss_train, loss_validation):
    plt.plot(loss_train)
    plt.plot(loss_validation)
    plt.show()


# Train

def train_validation(model_mlm, model_ce, train_dataloader, validation_dataloader, optimizer_mlm, optimizer_ce,
                     data_len_train, data_len_validation, batch_size, epoch, lambda_):
    # scheduler_mlm, scheduler_ce,
    model_mlm.zero_grad()
    model_ce.zero_grad()

    model_mlm.to(device)
    model_ce.to(device)

    loss_train = []
    loss_validation = []
    for epoch in range(epoch + 1):
        model_mlm.train()  # For training
        model_ce.train()

        for idx, batch_data in enumerate(train_dataloader):
            batch_data = batch_data.to(device)
            # Model MLM
            outputs_mlm = model_mlm(batch_data, masked_lm_labels=batch_data)
            # input_ids: Indices of input sequence tokens in the vocabulary.(in this case, it corresponds to batch_data)
            loss_mlm = outputs_mlm[:1][0]
            # TODO: Check what should be the masked_lm_labels exactly.

            '''
            get labels and input for the model_ce
            outputs_mlm[1] is Prediction scores of the language modeling head 
            (scores for each vocabulary token before SoftMax).
            '''
            labels_ce, input_ce = soft_max(batch_data, outputs_mlm)
            # Model CE
            outputs_ce = model_ce(input_ce, labels=labels_ce)
            '''
            input_ids: Indices of input sequence tokens in the vocabulary.
            labels:  Labels for computing the ELECTRA loss. 
            	Input should be a sequence of tokens (see input_ids docstring) Indices should be in [0, 1]. 
            	0 indicates the token is an original token, 1 indicates the token was replaced.
            '''
            loss_ce = outputs_ce[:1][0]
            # Total Loss
            loss = loss_mlm + lambda_ * loss_ce
            # Lambda is a hyperparameter described in the paper
            # TODO: fine tune lambda.

            # Autograd
            optimizer_mlm.zero_grad()
            optimizer_ce.zero_grad()
            loss.backward()
            optimizer_mlm.step()
            optimizer_ce.step()

            # if(scheduler_mlm is not None and scheduler_ce is not None):
            # scheduler_mlm.step()
            # scheduler_ce.step()

            loss_train.append(loss.item())
            print(f'Epoch: {epoch + 1} | batch: {idx + 1}/{data_len_train/batch_size} | Train Loss: {loss.item():.4f}')

        model_mlm.eval()  # For validation
        model_ce.eval()
        with torch.no_grad():
            for idx, batch_data in enumerate(validation_dataloader):
                batch_data = batch_data.to(device)

            outputs_mlm_val = model_mlm(batch_data, masked_lm_labels=batch_data)
            loss_mlm_val = outputs_mlm_val[:1][0]
            labels_ce_val, output_val_ = soft_max(batch_data, outputs_mlm_val)

            outputs_ce_val = model_ce(output_val_, labels=labels_ce_val)
            loss_ce_val = outputs_ce_val[:1][0]

            loss_val = loss_mlm_val + lambda_ * loss_ce_val

            loss_validation.append(loss_val.item())
            print(
                f'Epoch: {epoch + 1} | batch: {idx + 1}/{data_len_validation/batch_size} | Validation Loss: {loss_val.item():.4f}')

        # Save Models after each epoch.
        torch.save(model_mlm, "/content/drive/My Drive/Electra_mlm_{}.pt".format(epoch))
        torch.save(model_ce, "/content/drive/My Drive/Electra_ce_{}.pt".format(epoch))

    return loss_train, loss_validation


# Train
device = torch.device('cuda:0')

batch_size_ = 8
lambda__ = 10
epoch_ = 100
lr = 1e-5
split_num_ = 10

# txt_generation(split_num_)

validation_data_file = "/content/drive/My Drive/wiki.valid.raw"
eval_data_file = "/content/drive/My Drive/wiki.test.raw"
train_data_file = '/content/drive/My Drive/wiki.train.raw'

tokenizer_, model_mlm_, model_ce_ = init_()
train_dataloader_, data_len_train_ = data_loader(tokenizer=tokenizer_, batch_size=batch_size_, dev=False,
                                                 evaluate=False)
validation_dataloader_, data_len_validation_ = data_loader(tokenizer=tokenizer_, batch_size=batch_size_, dev=True,
                                                           evaluate=False)
# evaluation_dataloader_, data_len_evaluation_= data_loader(tokenizer = tokenizer_, batch_size = batch_size_, dev = False, evaluate = True)

print("Data loaded successfully")

optimizer_mlm_, optimizer_ce_ = init_optimizer(model_mlm=model_mlm_, model_ce=model_ce_, learning_rate=lr)
# scheduler_mlm_, scheduler_ce_ = scheduler(optimizer_mlm = optimizer_mlm_, optimizer_ce = optimizer_ce_)#, data_len = data_len_, batch_size = batch_size_)
loss_train_, loss_validation_ = train_validation(model_mlm=model_mlm_, model_ce=model_ce_,
                                                 train_dataloader=train_dataloader_,
                                                 validation_dataloader=validation_dataloader_,
                                                 optimizer_mlm=optimizer_mlm_,
                                                 optimizer_ce=optimizer_ce_,
                                                 data_len_train=data_len_train_,
                                                 data_len_validation=data_len_validation_,
                                                 batch_size=batch_size_, epoch=epoch_, lambda_=lambda__)
# scheduler_mlm = scheduler_mlm_, scheduler_ce = scheduler_ce_,
plot_loss(loss_train_, loss_validation_)

# TODO: Encapsulate the training
