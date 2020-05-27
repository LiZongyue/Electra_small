"""
File for all classed of configs for electra
"""
from .config_utils import BaseConfig


class ElectraFileConfig(BaseConfig):
    def __init__(self, **kwargs):
        super().__init__()
        self.train_data_file = kwargs.pop("train_data_file", "C:/Users/Zongyue Li/Documents/Github/BNP/Data/wiki"
                                                             ".train.raw")
        self.validation_data_file = kwargs.pop("validation_data_file", "C:/Users/Zongyue Li/Documents/Github/BNP/Data"
                                                                       "/wiki.valid.raw")
        self.eval_data_file = kwargs.pop("eval_data_file", "C:/Users/Zongyue Li/Documents/Github/BNP/Data/wiki.test.raw")
        self.save_path = kwargs.pop("save_path", "C:/Users/Jackie/Documents/GitHub/output/Discriminator{}.p")
        self.import_path = kwargs.pop("import_path", "C:/Users/Zongyue Li/Documents/Github/BNP/output/Discriminator9.p")


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
        self.train_head_epoch = kwargs.pop("train_head_epoch", 3)
        self.batch_size_train = kwargs.pop("batch_size_train", 128)
        self.batch_size_val = kwargs.pop("batch_size_val", 4)
        self.softmax_temperature = kwargs.pop("softmax_temperature", 1)
        self.lambda_ = kwargs.pop("lambda_", 50)
        self.add_special_tokens = kwargs.pop("add_special_tokens", True)
        self.max_length = kwargs.pop("max_length", 128)
