import copy
import torch
import gc

from torch import nn
from transformers import BertForMaskedLM


class BertBase(nn.Module):

    def __init__(self, train_config):
        super().__init__()
        self.train_config = train_config
        self.model = BertForMaskedLM.from_pretrained("bert-base-uncased")
        self.device = torch.device('cuda:{}'.format(self.train_config.gpu_id)) if torch.cuda.is_available() else 'cpu'
        self.mask_value = 100
        self.ignored_token_value = -103

    def forward(self, data: torch.tensor):
        data, mask_labels = data
        mask_labels = mask_labels.bool()
        data = data.to(self.device)
        mask_labels = mask_labels.to(self.device)
        data_bert = copy.deepcopy(data)
        data_bert[mask_labels] = self.mask_value
        label_bert = copy.deepcopy(data)
        label_bert[~mask_labels] = self.ignored_token_value
        attention_mask = data_bert != 0

        data_bert = data_bert.to(self.device)
        label_bert = label_bert.to(self.device)
        attention_mask = attention_mask.to(self.device)

        # garbage collection
        gc.collect()

        outputs = self.model(data_bert, masked_lm_labels=label_bert, attention_mask=attention_mask)

        loss, = outputs[:1]
        return loss

    def bert_getter(self):
        return self.model
