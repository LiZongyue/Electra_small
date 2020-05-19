import torch.nn
import copy
import torch

from torch import nn
from transformers import ElectraForPreTraining, ElectraForMaskedLM


class Electra(nn.Module):
    _generator_: ElectraForMaskedLM
    _discriminator_: ElectraForPreTraining

    def __init__(self, model_config, train_config):
        super().__init__()
        self.model_config = model_config
        self.train_config = train_config

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
        x = x.to(self.device)
        mask_labels = mask_labels.to(self.device)
        data_generator = copy.deepcopy(x)
        data_generator[mask_labels] = 103
        label_generator = copy.deepcopy(x)
        label_generator[~mask_labels] = -100
        attention_mask = data_generator != 0

        data_generator = data_generator.to(self.device)
        label_generator = label_generator.to(self.device)
        attention_mask = attention_mask.to(self.device)

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
