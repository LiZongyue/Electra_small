"""
File for all classed of configs for electra
"""
from .config_utils import BaseConfig


class ElectraModelConfig(BaseConfig):
    def __init__(self, **kwargs):
        super().__init__()
        self.embedding_size = kwargs.pop("embedding_size", 64)
        self.hidden_size_mlm = kwargs.pop("hidden_size_mlm", 64)
        self.hidden_size_ce = kwargs.pop("hidden_size_ce", 128)
        self.num_hidden_layers_mlm = kwargs.pop("num_hidden_layers_mlm", 3)
        self.num_hidden_layers_ce = kwargs.pop("num_hidden_layers_ce", 6)
        self.intermediate_size_mlm = kwargs.pop("intermediate_size_mlm", 256)
        self.intermediate_size_ce = kwargs.pop("intermediate_size_ce", 512)


class ElectraTrainConfig(BaseConfig):
    def __init__(self, **kwargs):
        super().__init__()
        self.gpu_id = kwargs.pop("gpu_id", 0)
        self.learning_rate = kwargs.pop("learning_rate", 1e-5)
        self.warmup_steps = kwargs.pop("warmup_steps", 10)
        self.n_epochs = kwargs.pop("n_epochs", 50)
        self.batch_size = kwargs.pop("batch_size", 8)
        self.softmax_temperature = kwargs.pop("softmax_temperature", 1)
        self.lambda_ = kwargs.pop("lambda_", 50)
        self.num_training_steps = kwargs.pop("num_training_steps", 10000)
