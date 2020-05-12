"""
Electra Small
"""
import copy
import os
import math
import torch
import matplotlib.pyplot as plt

from torch import nn
from torch.nn.utils.rnn import pad_sequence
from Electra_small.configs import ElectraModelConfig, ElectraTrainConfig
from torch.utils.data import DataLoader, Dataset, RandomSampler
from transformers import ElectraTokenizer, ElectraForPreTraining, ElectraConfig, ElectraForMaskedLM, \
    get_linear_schedule_with_warmup, AdamW


class TextDataset(Dataset):
    """
    Subclass DataSet
    """

    def __init__(self, tokenizer, file_path: str, block_size=512):
        assert os.path.isfile(file_path)

        with open(file_path, encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

        batch_encoding = tokenizer.batch_encode_plus(lines, add_special_tokens=True, max_length=block_size,
                                                     pad_to_max_length=True)
        self.examples = batch_encoding["input_ids"]

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

    @classmethod
    def data_loader(cls, tokenizer, train_config, train_data_file, validation_data_file, eval_data_file, dev, evaluate):
        # DataLoader
        dataset_ = cls.load_and_cache_examples(tokenizer, train_data_file, validation_data_file,
                                               eval_data_file, dev, evaluate)
        sampler_ = RandomSampler(dataset_)
        dataloader = DataLoader(
            dataset_, sampler=sampler_, batch_size=train_config.batch_size#, collate_fn=cls.collate_fn
        )
        data_len = dataset_.__len__()

        return dataloader, data_len

    @staticmethod
    def collate_fn(batch):
        x_padded = pad_sequence(batch, batch_first=True)

        return x_padded


class ElectraRunner(object):

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

        # self.device = torch.device('cuda:{}'.format(self.train_config.gpu_id))

    def __tokenizer_getter__(self):
        return self.tokenizer

    def train_validation(self, train_dataloader, validation_dataloader, data_len_train, data_len_validation):
        self.optimizer = self.init_optimizer(self.generator, self.discriminator, self.train_config.learning_rate)
        self.scheduler = self.scheduler_electra(self.optimizer)

        #self.generator.to(self.device)
        #self.discriminator.to(self.device)

        loss_train = []
        loss_validation = []
        for epoch_id in range(self.train_config.n_epochs):
            # Train
            self.generator.train()
            self.discriminator.train()
            for idx, data in enumerate(train_dataloader):
                loss_tr = self.train_one_step(epoch_id, idx, data, data_len_train)
                loss_train.append(loss_tr)

            with torch.no_grad():
                for idx, data in enumerate(validation_dataloader):
                    loss_val = self.validation_one_step(epoch_id, idx, data, data_len_validation)
                    loss_validation.append(loss_val)

            torch.save(self.discriminator.cpu().state_dict(), "C:/Users/Zongyue Li/Documents/Github/BNP/Electra_small"
                                                              "/output/Discriminator{}.p".format(epoch_id))
            # TODO: Change the directory more generally and change the save method.

        return loss_train, loss_validation

    def train_one_step(self, epoch_id, idx, data, data_len_train):
        self.generator.train()
        self.discriminator.train()

        #data = data.to(self.device)
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
        self.generator.eval()
        self.discriminator.eval()
        #data = data.to(self.device)
        loss = self.process_model(data)
        print(f'Epoch: {epoch_id + 1} | '
              f'batch: {idx + 1} / {math.ceil(data_len_validation / self.train_config.batch_size)} | '
              f'Validation Loss: {loss:.4f}')
        return loss.item()

    def soft_max(self, output_data):
        m = nn.Softmax(dim=1)
        output_softmax = torch.distributions.Categorical(
            m(output_data[1] / self.train_config.softmax_temperature)).sample()
        # get output_IDs of model_mlm by applyng sampling.
        return output_softmax

    # TODO: Use Negative Sampling to optimize the training speed.

    def scheduler_electra(self, optimizer):  # , data_len, batch_size):
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.train_config.warmup_steps,
            num_training_steps=self.train_config.num_training_steps
        )
        return scheduler

    def process_model(self, data):
        mask_label = torch.rand(data.size()[0], len(data[0])) > 0.85
        # A mask, which indicates whether the token in the data tensor is masked or not. Contains only boolean values.
        label_generator = copy.deepcopy(data)
        label_generator[~mask_label] = -100
        # Mask tokens who won't be replaced during training. -100 will be assigned to the masked tokens, whose loss will
        # not be calculated. The rest 15% tokens will not be changed.
        score_generator = self.generator(data, masked_lm_labels=label_generator)
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

    electra = ElectraRunner(model_config, train_config)

    tokenizer = electra.__tokenizer_getter__()

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

    loss_train, loss_validation = electra.train_validation(train_dataloader=train_data_loader,
                                                           validation_dataloader=valid_data_loader,
                                                           data_len_train=train_data_len,
                                                           data_len_validation=valid_data_len)

    electra.plot_loss(loss_train=loss_train, loss_validation=loss_validation)


if __name__ == "__main__":
    main()
