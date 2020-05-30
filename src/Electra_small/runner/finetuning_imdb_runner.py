import torch
import numpy as np

from sklearn.metrics import accuracy_score
from transformers import get_linear_schedule_with_warmup, AdamW
from Electra_small.modeling import ElectraForClassification, BertForClassification


class FineTuningRunner:

    def __init__(self, model_config, train_config, file_config):
        self.model_config = model_config
        self.train_config = train_config
        self.file_config = file_config
        self.tokenizer = None
        self.model = None

        pre_trained_weights = torch.load(self.file_config.import_path, map_location=torch.device('cpu'))
        self.state_to_load = {}
        for name, param in pre_trained_weights.items():
            if name.startswith('electra.'):
                self.state_to_load[name[len('electra.'):]] = param
            elif name.startswith('bert.'):
                self.state_to_load[name[len('bert.'):]] = param

        self.optimizer = None
        self.scheduler = None

        self.device = torch.device('cuda:{}'.format(self.train_config.gpu_id)) if torch.cuda.is_available() else 'cpu'

    def __tokenizer_getter__(self):
        return self.tokenizer

    @staticmethod
    def sigmoid(logits):
        return 1 / (1 + np.exp(-logits))

    def run(self, train_dataloader, validation_dataloader, train_data_len, eval_dataloader):
        for epoch_id in range(self.train_config.train_head_epoch):
            optimizer = AdamW(list(self.param_opt),
                              lr=self.train_config.learning_rate)
            mean_loss_train, mean_loss_val, acc_tr, acc_val = self.train_validation(train_dataloader,
                                                                                    validation_dataloader,
                                                                                    train_data_len, False, optimizer)
            print(
                f"Epoch {epoch_id}: train_head_loss={mean_loss_train:.4f}, train_head_acc={acc_tr:.4f}, "
                f"valid_head_loss={mean_loss_val:.4f}, val_head_acc={acc_val:.4f}")

        # train all
        num_layers = len(self.model.layers)
        param_optimizer = list(self.model.named_parameters())
        optimizer_parameters = self.get_layer_wise_lr_decay(num_layers, param_optimizer,
                                                            self.train_config.learning_rate, self.train_config.lr_decay)
        optimizer = AdamW(optimizer_parameters, lr=self.train_config.learning_rate, eps=1e-8, weight_decay=0.01)
        for epoch_id in range(self.train_config.n_epochs):
            mean_loss_train, mean_loss_val, acc_tr, acc_val = self.train_validation(train_dataloader,
                                                                                    validation_dataloader,
                                                                                    train_data_len, True, optimizer)
            print(
                f"Epoch {epoch_id}: train_loss={mean_loss_train:.4f}, train_acc={acc_tr:.4f}, "
                f"valid_loss={mean_loss_val:.4f}, val_acc={acc_val:.4f}")
        loss_evaluation = []
        logits_eval = []
        labels_eval = []
        for idx, data in enumerate(eval_dataloader):
            label_eval, loss_eval, logits_evaluation = self.validation_one_step(data)
            loss_evaluation.append(loss_eval)
            logits_eval.append(logits_evaluation.detach().cpu().numpy())
            labels_eval.append(label_eval.cpu().numpy())

        logits_eval = np.concatenate(logits_eval)
        labels_eval = np.concatenate(labels_eval)

        mean_loss_eval = np.mean(np.array(loss_evaluation))
        acc_eval = accuracy_score(labels_eval, (self.sigmoid(logits_eval) > 0.5).astype(int))

        print(f"eval loss={mean_loss_eval:.4f}, eval acc={acc_eval:.4f}")

    def train_validation(self, train_dataloader, validation_dataloader, train_data_len, head: bool, optimizer):
        self.optimizer = optimizer
        self.scheduler = self.scheduler_electra(self.optimizer, train_data_len)

        if head:
            if isinstance(self.model, ElectraForClassification):
                self.model.electra.required_grad = False
            elif isinstance(self.model, BertForClassification):
                self.model.bert.required_grad = False
            else:
                raise Exception("Model is not recognized. Make sure your model is Bert base or Electra Base.")
        else:
            if isinstance(self.model, ElectraForClassification):
                self.model.electra.required_grad = True
            elif isinstance(self.model, BertForClassification):
                self.model.bert.required_grad = True
            else:
                raise Exception("Model is not recognized. Make sure your model is Bert base or Electra Base.")

        self.model.to(self.device)

        logits_train = []
        logits_val = []
        labels_train = []
        labels_val = []
        loss_train = []
        loss_validation = []
        # Train
        self.model.train()
        for idx, data in enumerate(train_dataloader):
            label_tr, loss_tr, logits_tr = self.train_one_step(data)
            loss_train.append(loss_tr)
            logits_train.append(logits_tr.detach().cpu().numpy())
            labels_train.append(label_tr.cpu().numpy())
        with torch.no_grad():
            for idx, data in enumerate(validation_dataloader):
                label_val, loss_val, logits_validation = self.validation_one_step(data)
                loss_validation.append(loss_val)
                logits_val.append(logits_validation.detach().cpu().numpy())
                labels_val.append(label_val.cpu().numpy())

        logits_train = np.concatenate(logits_train)
        labels_train = np.concatenate(labels_train)

        logits_val = np.concatenate(logits_val)
        labels_val = np.concatenate(labels_val)

        mean_loss_train = np.mean(np.array(loss_train))
        mean_loss_val = np.mean(np.array(loss_validation))
        acc_tr = accuracy_score(labels_train, (self.sigmoid(logits_train) > 0.5).astype(int))
        acc_val = accuracy_score(labels_val, (self.sigmoid(logits_val) > 0.5).astype(int))

        return mean_loss_train, mean_loss_val, acc_tr, acc_val

    def train_one_step(self, data):
        self.model.train()

        labels, loss, logits = self.process_model(data)
        # Autograd
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        return labels, loss.item(), logits

    def validation_one_step(self, data):
        self.model.eval()

        labels, loss, logits = self.process_model(data)
        return labels, loss.item(), logits

    def process_model(self, data):
        example_input, example_labels = data
        example_input = example_input.to(self.device)
        example_labels = example_labels.to(self.device).float()
        # example_input = torch.randint(0, 30522, (3, 10)).long(),  input shape (bs, seq_len)
        # example_labels = # torch.randint(0, 3, (3,)).long()  # labels shape (bs, )
        scores = self.model(example_input)  # output scores/logits shape (bs, num_labels)
        scores = scores.to(self.device)
        loss = self.model.get_loss(scores, example_labels)
        return example_labels, loss, scores

    def scheduler_electra(self, optimizer, train_data_len):  # , data_len, batch_size):
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.train_config.warmup_steps,
            num_training_steps=int(train_data_len / self.train_config.batch_size_train * self.train_config.n_epochs)
        )
        return scheduler

    def get_layer_wise_lr_decay(self, num_layers, param_optimizer, lr, lr_decay):
        if isinstance(self.model, ElectraForClassification):
            layers_str = 'electra.encoder.layer.{}.'
            embedding_str = 'electra.embeddings.'
        elif isinstance(self.model, BertForClassification):
            layers_str = 'bert.encoder.layer.{}.'
            embedding_str = 'bert.embeddings.'
        else:
            raise Exception("Model is mistaking at get_layer_wise_lr_decay func.")

        optimizer_paramters = []
        included_params_names = []

        for i in range(num_layers):
            layer_group = []
            for n, p in param_optimizer:
                if n.startswith(layers_str.format(i)):
                    included_params_names.append(n)
                    layer_group.append(p)
            optimizer_paramters.append({'params': layer_group, 'lr': lr * lr_decay ** (num_layers - i)})

        embedding_group = []
        for n, p in param_optimizer:
            if n.startswith(embedding_str):
                included_params_names.append(n)
                embedding_group.append(p)
        optimizer_paramters.append({'params': embedding_group, 'lr': lr * lr_decay ** (num_layers + 1)})

        rest_group = []
        for n, p in param_optimizer:
            if n not in included_params_names:
                rest_group.append(p)
        optimizer_paramters.append({'params': rest_group, "lr": lr})

        return optimizer_paramters