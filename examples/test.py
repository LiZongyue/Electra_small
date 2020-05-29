from Electra_small.modeling.modeling_electra_SST import ElectraForClassification
from transformers import ElectraConfig

model_config = {
    "embedding_size": 768,
    "hidden_size": 768,
    "num_hidden_layers": 12,
    "intermediate_size": 3072,
    "num_labels": 1,
}
model_config = ElectraConfig(**model_config)
model = ElectraForClassification(model_config)

print(isinstance(model, ElectraForClassification))