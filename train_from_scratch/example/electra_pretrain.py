"""
Electra Small
"""
import copy
import os
import math
import torch
import numpy as np
import matplotlib.pyplot as plt

from torch import nn
from torch.nn.utils.rnn import pad_sequence
from Electra_small.configs import ElectraModelConfig, ElectraTrainConfig
from torch.utils.data import DataLoader, Dataset, RandomSampler
from transformers import ElectraTokenizer, ElectraForPreTraining, ElectraConfig, ElectraForMaskedLM, \
    get_linear_schedule_with_warmup, AdamW

tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator')  # global variable


class TextDataset(Dataset):
    """
    Subclass DataSet
    """

    def __init__(self, file_path: str):
        super().__init__()
        assert os.path.isfile(file_path)
        with open(file_path, encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

        self.examples = lines

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]


def load_and_cache_examples(train_data_file, validation_data_file,
                            eval_data_file, dev=False, evaluate=False):
    # Load and cache examples for different dataset
    if evaluate:
        file_path = eval_data_file
    else:
        if dev:
            file_path = validation_data_file
        else:
            file_path = train_data_file
    return TextDataset(file_path=file_path)


def data_loader(train_config, train_data_file, validation_data_file, eval_data_file, dev, evaluate):
    # DataLoader
    dataset_ = load_and_cache_examples(train_data_file, validation_data_file, eval_data_file, dev, evaluate)
    sampler_ = RandomSampler(dataset_)
    dataloader = DataLoader(
        dataset_, sampler=sampler_, batch_size=train_config.batch_size, collate_fn=collate_func, num_workers=4
    )
    data_len = dataset_.__len__()

    return dataloader, data_len


def collate_func(batch):
    batch_encoding = tokenizer.batch_encode_plus(batch, add_special_tokens=True, max_length=240)
    batch_flag = batch_encoding['input_ids']

    mask_labels = []
    x = []
    for item in batch_flag:
        x.append(torch.tensor(item, dtype=torch.long))
        mask_labels.append((torch.rand(len(item)) > 0.85).long())

    x_padded = pad_sequence(x, batch_first=True)
    mask_labels_padded = pad_sequence(mask_labels, batch_first=True)
    return [x_padded, mask_labels_padded]


class Electra(nn.Module):
    _generator_: ElectraForMaskedLM
    _discriminator_: ElectraForPreTraining

    def __init__(self, model_config, train_config):
        super().__init__()
        self.model_config = model_config
        self.train_config = train_config

        self.config_generator = ElectraConfig(embedding_size=model_config.embedding_size,
                                              hidden_size=model_config.hidden_size_mlm,
                                              num_hidden_layers=model_config.num_hidden_layers_mlm,
                                              intermediate_size=model_config.intermediate_size_mlm)
        self.config_discriminator = ElectraConfig(embedding_size=model_config.embedding_size,
                                                  hidden_size=model_config.hidden_size_ce,
                                                  num_hidden_layers=model_config.num_hidden_layers_ce,
                                                  intermediate_size=model_config.intermediate_size_ce)

        self._generator_ = ElectraForMaskedLM(self.config_generator)
        self._discriminator_ = ElectraForPreTraining(self.config_discriminator)

        self._tie_embedding()

        self.device = torch.device('cuda:{}'.format(self.train_config.gpu_id))

    def _tie_embedding(self):
        self._discriminator_.electra.embeddings.word_embeddings.weight = \
            self._generator_.electra.embeddings.word_embeddings.weight
        self._discriminator_.electra.embeddings.position_embeddings.weight = \
            self._generator_.electra.embeddings.position_embeddings.weight
        self._discriminator_.electra.embeddings.token_type_embeddings.weight = \
            self._generator_.electra.embeddings.token_type_embeddings.weight

    def discriminator_getter(self):
        return self._discriminator_

    def generator_getter(self):
        return self._generator_

    def forward(self, data: torch.tensor):
        # A mask, which indicates whether the token in the data tensor is masked or not. Contains only boolean values.
        x, mask_labels = data
        mask_labels = mask_labels.bool()
        # x = x.to(self.device)
        # mask_labels = mask_labels.to(self.device)
        data_generator = copy.deepcopy(x)
        data_generator[mask_labels] = 103
        label_generator = copy.deepcopy(x)
        label_generator[~mask_labels] = -100
        attention_mask = data_generator != 0

        # data_generator = data_generator.to(self.device)
        # label_generator = label_generator.to(self.device)
        # attention_mask = attention_mask.to(self.device)

        score_generator = self._generator_(data_generator, attention_mask, masked_lm_labels=label_generator)
        # TODO: ablations here. attention_mask is assigned.
        loss_generator = score_generator[:1][0]
        output_generator = self.soft_max(score_generator)
        # Get the Softmax result for next step.
        input_discriminator = torch.zeros_like(x)

        input_discriminator[mask_labels] = output_generator[mask_labels]  # Part A
        input_discriminator[~mask_labels] = x[~mask_labels]  # Part B
        # Input of the Discriminator will be replaced tokens (PartA) and non-replaced tokens (PartB)
        labels_discriminator = torch.zeros_like(x)
        labels_discriminator[~mask_labels] = 0
        labels_discriminator[mask_labels] = (1 - torch.eq(x[mask_labels], output_generator[mask_labels]).int()).long()
        # If a token is replaced, 1 will be assigned to the discriminator's label, if not, 0 will be assigned.
        outputs_discriminator = self._discriminator_(input_discriminator, attention_mask, labels=labels_discriminator)
        # TODO: ablations here. attention_mask is assigned.
        loss_discriminator = outputs_discriminator[:1][0]

        loss = loss_generator + self.train_config.lambda_ * loss_discriminator

        return loss

    def soft_max(self, output_data):
        m = nn.Softmax(dim=1)
        output_softmax = torch.distributions.Categorical(
            m(output_data[1] / self.train_config.softmax_temperature)).sample()
        # get output_IDs of model_mlm by applyng sampling.
        return output_softmax


class Runner(object):

    def __init__(self, electra, train_config):
        self.electra_small = electra
        self.train_config = train_config
        # self.device = torch.device('cuda:{}'.format(self.train_config.gpu_id))
        self.optimizer = None
        self.scheduler = None

    def train_validation(self, train_dataloader, validation_dataloader, data_len_train, data_len_validation):

        # self.electra_small.to(self.device)
        self.optimizer = self.init_optimizer(self.electra_small.generator_getter(),
                                             self.electra_small.discriminator_getter(),
                                             learning_rate=self.train_config.learning_rate)
        self.scheduler = self.scheduler_electra(self.optimizer)
        loss_train = []
        loss_validation = []
        for epoch_id in range(self.train_config.n_epochs):
            # Train
            self.electra_small.train()
            for idx, data in enumerate(train_dataloader):
                loss_tr = self.train_one_step(epoch_id, idx, data, data_len_train)
                loss_train.append(loss_tr)

            with torch.no_grad():
                for idx, data in enumerate(validation_dataloader):
                    loss_val = self.validation_one_step(epoch_id, idx, data, data_len_validation)
                    loss_validation.append(loss_val)

            torch.save(self.electra_small.discriminator_getter().cpu().state_dict(),
                       "C:/Users/Zongyue Li/Documents/Github/BNP/Electra_small"
                       "/output/Discriminator{}.p".format(epoch_id + 1))
            # TODO: Change the directory more generally and change the save method.

        return loss_train, loss_validation

    def train_one_step(self, epoch_id, idx, data, data_len_train):
        self.electra_small.train()

        loss = self.electra_small(data)
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
        self.electra_small.eval()

        loss = self.electra_small(data)
        print(f'Epoch: {epoch_id + 1} | '
              f'batch: {idx + 1} / {math.ceil(data_len_validation / self.train_config.batch_size)} | '
              f'Validation Loss: {loss:.4f}')
        return loss.item()

    # TODO: Use Negative Sampling to optimize the training speed.

    def scheduler_electra(self, optimizer):  # , data_len, batch_size):
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.train_config.warmup_steps,
            num_training_steps=self.train_config.num_training_steps
        )
        return scheduler

    @staticmethod
    def init_optimizer(model1, model2, learning_rate):
        # Initialize Optimizer AdamW
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


'''
    def process_model(self, data):
        data_generator, label_generator, mask_label = data
        # Mask tokens who won't be replaced during training. -100 will be assigned to the masked tokens, whose loss will
        # not be calculated. The rest 15% tokens will not be changed.
        score_generator = self.generator(data_generator, masked_lm_labels=label_generator)
        # TODO: The score_generator will be different if the masked_lm_labels parameter is default as None. Check out.
        loss_generator = score_generator[:1][0]
        output_generator = self.soft_max(score_generator)
        # Get the Softmax result for next step.
        input_discriminator = torch.zeros_like(data)

        input_discriminator[mask_label] = output_generator[mask_label]  # Part A
        input_discriminator[~mask_label] = data[~mask_label]  # Part B
        # Input of the Discriminator will be replaced tokens (PartA) and non-replaced tokens (PartB)
        labels_discriminator = torch.zeros_like(data)
        labels_discriminator[~mask_label] = 0
        labels_discriminator[mask_label] = (1 - torch.eq(data[mask_label], output_generator[mask_label]).int()).long()
        # If a token is replaced, 1 will be assigned to the discriminator's label, if not, 0 will be assigned.
        outputs_discriminator = self.discriminator(input_discriminator, labels=labels_discriminator)
        loss_discriminator = outputs_discriminator[:1][0]

        loss = loss_generator + self.train_config.lambda_ * loss_discriminator

        return loss
'''


def main():
    train_data_file = "C:/Users/Zongyue Li/Documents/Github/BNP/Data/wiki.train.raw"
    validation_data_file = "C:/Users/Zongyue Li/Documents/Github/BNP/Data/wiki.valid.raw"
    eval_data_file = "C:/Users/Zongyue Li/Documents/Github/BNP/Data/wiki.test.raw"

    # TODO: Change the strings to a dictionary and pack them into a config object.

    model_config = {
        "embedding_size": 128,
        "hidden_size_mlm": 32,
        "hidden_size_ce": 128,
        "num_hidden_layers_mlm": 6,
        "num_hidden_layers_ce": 6,
        "intermediate_size_mlm": 256,
        "intermediate_size_ce": 512,
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

    model_config = ElectraModelConfig(**model_config)
    train_config = ElectraTrainConfig(**train_config)

    electra_small = Electra(model_config, train_config)

    train_data_loader, train_data_len = data_loader(train_config=train_config,
                                                    train_data_file=train_data_file,
                                                    validation_data_file=validation_data_file,
                                                    eval_data_file=eval_data_file,
                                                    dev=False, evaluate=False)
    valid_data_loader, valid_data_len = data_loader(train_config=train_config,
                                                    train_data_file=train_data_file,
                                                    validation_data_file=validation_data_file,
                                                    eval_data_file=eval_data_file,
                                                    dev=True, evaluate=False)

    runner = Runner(electra_small, train_config)

    loss_train, loss_validation = runner.train_validation(train_dataloader=train_data_loader,
                                                          validation_dataloader=valid_data_loader,
                                                          data_len_train=train_data_len,
                                                          data_len_validation=valid_data_len)

    runner.plot_loss(loss_train=loss_train, loss_validation=loss_validation)


if __name__ == "__main__":
    main()

# Note: on 2020/5/13, without attention mask, loss decreased extremely quickly to around 1~2 (original version). After
# 1 epoch, loss will be ~20.
# TODO: Ablations