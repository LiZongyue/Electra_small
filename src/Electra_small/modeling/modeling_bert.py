import copy
import torch

from torch import nn
from transformers import BertForMaskedLM


class BertBase(nn.Module):

    def __init__(self, train_config):
        super().__init__()
        self.train_config = train_config
        self.model = BertForMaskedLM.from_pretrained("bert-base-uncased")
        self.device = torch.device('cuda:{}'.format(self.train_config.gpu_id)) if torch.cuda.is_available() else 'cpu'

    def forward(self, data: torch.tensor):
        x, mask_labels = data
        mask_labels = mask_labels.bool()
        x = x.to(self.device)
        mask_labels = mask_labels.to(self.device)
        data_bert = copy.deepcopy(x)
        data_bert[mask_labels] = 103
        label_bert = copy.deepcopy(x)
        label_bert[~mask_labels] = -100
        attention_mask = data_bert != 0

        data_bert = data_bert.to(self.device)
        label_bert = label_bert.to(self.device)
        attention_mask = attention_mask.to(self.device)

        outputs = self.model(data_bert, masked_lm_labels=label_bert, attention_mask=attention_mask)

        loss, = outputs[:1]
        return loss

    def bert_getter(self):
        return self.model
