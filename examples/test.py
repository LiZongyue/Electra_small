import torch
from transformers import AutoModelWithLMHead

model = AutoModelWithLMHead.from_pretrained("bert-base-uncased")

pre_trained_weights = torch.load(model, map_location=torch.device('cpu'))

state_to_load = {}
for name, param in pre_trained_weights.items():
    if name.startswith('electra.'):
        state_to_load[name[len('electra.'):]] = param
    elif name.startswith('bert.'):
        state_to_load[name[len('bert.'):]] = param