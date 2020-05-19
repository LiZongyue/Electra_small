import torch
import torch.nn as nn

from transformers.modeling_electra import ElectraModel, ElectraPreTrainedModel


class ElectraForClassification(ElectraPreTrainedModel):
    """
    Electra for downstream task -- sentence-level classification
    """

    def __init__(self, config):
        super().__init__(config)

        self.electra = ElectraModel(config)
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
        discriminator_hidden_states = self.electra(input_ids, attention_mask, token_type_ids, position_ids, head_mask,
                                                   inputs_embeds)
        discriminator_cls_output = discriminator_hidden_states[0][:, 0]  # load the hidden states of [CLS] token
        discriminator_cls_output = self.dropout(discriminator_cls_output)

        logits = self.classifier(discriminator_cls_output)

        return logits

    def get_loss(self, scores, labels):
        """
        :param labels: torch.LongTensor of shape :obj: (batch_size, 1), Indices should be in ``[0, ..., config.num_labels - 1]``
        :return:
        """
        loss = self.loss_fn(scores, labels)
        return loss

    def load_electra_weights(self, state_dict):
        self.electra.load_state_dict(state_dict)
