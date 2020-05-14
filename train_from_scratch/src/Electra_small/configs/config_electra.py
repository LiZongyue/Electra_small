"""
File for all classed of configs for electra
"""
from .config_utils import BaseConfig


class ElectraModelConfig(BaseConfig):
    def __init__(self, **kwargs):
        super().__init__()
        self.embedding_size = kwargs.pop("embedding_size", 128)
        self.hidden_size_mlm = kwargs.pop("hidden_size_mlm", 64)
        self.hidden_size_ce = kwargs.pop("hidden_size_ce", 256)
        self.num_hidden_layers_mlm = kwargs.pop("num_hidden_layers_mlm", 12)
        self.num_hidden_layers_ce = kwargs.pop("num_hidden_layers_ce", 12)
        self.intermediate_size_mlm = kwargs.pop("intermediate_size_mlm", 256)
        self.intermediate_size_ce = kwargs.pop("intermediate_size_ce", 1024)
        self.attention_heads_mlm = kwargs.pop("attention_heads_mlm", 1)
        self.attention_heads_ce = kwargs.pop("attention_heads_ce", 4)


class ElectraTrainConfig(BaseConfig):
    def __init__(self, **kwargs):
        super().__init__()
        self.gpu_id = kwargs.pop("gpu_id", 0)
        self.learning_rate = kwargs.pop("learning_rate", 1e-4)
        self.warmup_steps = kwargs.pop("warmup_steps", 10000)
        self.n_epochs = kwargs.pop("n_epochs", 110)
        self.batch_size = kwargs.pop("batch_size", 128)
        self.softmax_temperature = kwargs.pop("softmax_temperature", 1)
        self.lambda_ = kwargs.pop("lambda_", 50)
        self.num_training_steps = kwargs.pop("num_training_steps", 9900000)
