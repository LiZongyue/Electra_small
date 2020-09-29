import torch.nn as nn

from transformers.modeling_bert import BertModel, BertPreTrainedModel


class BertForClassification(BertPreTrainedModel):
    """
    Bert for downstream task -- sentence-level classification
    """

    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None):
        """

        :param input_ids:
        :param attention_mask:
        :param token_type_ids:
        :param position_ids:
        :param head_mask:
        :param inputs_embeds:
        :return: predicted logits scores with shape (batch_size, 1)
        """
        if attention_mask is None:
            attention_mask = input_ids != 0

        bert_hidden_states = self.bert(input_ids, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds)
        bert_cls_output = bert_hidden_states[0][:, 0]  # load the hidden states of [CLS] token
        bert_cls_output = self.dropout(bert_cls_output)

        logits = self.classifier(bert_cls_output)

        return logits

    def get_loss(self, scores, labels):
        """
        :param scores:
        :param labels: torch.LongTensor of shape :obj: (batch_size, 1), Indices should be in ``[0, ..., config.num_labels - 1]``
        :return:
        """
        loss = self.loss_fn(scores, labels)
        return loss

    def load_bert_weights(self, state_dict):
        self.bert.load_state_dict(state_dict)
