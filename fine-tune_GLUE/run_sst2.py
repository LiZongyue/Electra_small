import torch

from transformers import ElectraConfig
from Electra_small.modeling import ElectraForClassification

if __name__ == "__main__":
    model_config = {
        "embedding_size"    : 64,
        "hidden_size"       : 128,
        "num_hidden_layers" : 6,
        "intermediate_size" : 512,
        "num_labels"        : 3,
    }

    model_config = ElectraConfig(**model_config)
    ElectraForClassification = ElectraForClassification(model_config)
    ElectraForClassification.load_electra_weights("/Users/cpius/Downloads/electra_state_dict.p")

    example_input  = torch.randint(0, 30522, (3, 10)).long()  # input shape (bs, seq_len)
    example_labels = torch.randint(0, 3, (3, )).long()        # labels shape (bs, )
    scores = ElectraForClassification(example_input)          # output scores/logits shape (bs, num_labels)
    loss = ElectraForClassification.get_loss(scores, example_labels)
    print("finish")
