import math
import torch
import matplotlib.pyplot as plt

# from apex import amp  TODO: use apex to do half precision calculation
from transformers import get_linear_schedule_with_warmup, AdamW


class Runner(object):

    def __init__(self, electra, train_config, file_config):
        self.electra_small = electra
        self.train_config = train_config
        self.file_config = file_config
        self.device = torch.device('cuda:{}'.format(self.train_config.gpu_id)) if torch.cuda.is_available() else 'cpu'
        self.optimizer = None
        self.scheduler = None

    def train_validation(self, train_dataloader, validation_dataloader, data_len_train, data_len_validation):

        self.electra_small.to(self.device)
        self.optimizer = self.init_optimizer(self.electra_small.generator_getter(),
                                             self.electra_small.discriminator_getter(),
                                             learning_rate=self.train_config.learning_rate)

        self.scheduler = self.scheduler_electra(self.optimizer, data_len=data_len_train)
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

            torch.save(self.electra_small.discriminator_getter().state_dict(),
                       self.file_config.save_path.format(epoch_id + 1))

        return loss_train, loss_validation

    def train_one_step(self, epoch_id, idx, data, data_len_train):
        self.electra_small.train()

        loss = self.electra_small(data)
        print(f'Epoch: {epoch_id + 1} | '
              f'batch: {idx + 1} / {math.ceil(data_len_train / self.train_config.batch_size_train)} | '
              f'Train Loss: {loss:.4f}')
        # Autograd
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        return loss.item()

    def validation_one_step(self, epoch_id, idx, data, data_len_validation):
        self.electra_small.eval()

        loss = self.electra_small(data)
        print(f'Epoch: {epoch_id + 1} | '
              f'batch: {idx + 1} / {math.ceil(data_len_validation / self.train_config.batch_size_val)} | '
              f'Validation Loss: {loss:.4f}')
        return loss.item()

    def scheduler_electra(self, optimizer, data_len):
        num_training_steps = int(data_len / self.train_config.batch_size_train * self.train_config.n_epochs)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.train_config.warmup_steps,
            num_training_steps=num_training_steps
        )
        return scheduler

    @staticmethod
    def init_optimizer(model1, model2, learning_rate):
        # Initialize Optimizer AdamW
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model1.named_parameters() if not any(nd in n for nd in no_decay)] +
                          [p for n, p in model2.named_parameters() if not any(nd in n for nd in no_decay)] +
                          [p for n, p in model1.named_parameters() if any(nd in n for nd in no_decay)] +
                          [p for n, p in model2.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=1e-8)

        return optimizer

    @staticmethod
    def plot_loss(loss_train, loss_validation):
        plt.plot(loss_train)
        plt.plot(loss_validation)
        plt.show()
