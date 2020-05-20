import os
import torch
from torch.utils.data import Dataset
from transformers import ElectraTokenizer
from torch.nn.utils.rnn import pad_sequence


class TextDataset(Dataset):
    examples: []
    tokenizer: ElectraTokenizer

    def __init__(self, file_path, train_config):
        super().__init__()
        self.train_config = train_config
        self.file_path = file_path
        assert os.path.isfile(file_path)
        with open(file_path, encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

        self.examples = lines

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]

    def collate_func(self, batch):
        batch_encoding = self.tokenizer.batch_encode_plus(batch, add_special_tokens=self.train_config.add_special_tokens,
                                                          max_length=self.train_config.max_length)
        batch_flag = batch_encoding['input_ids']

        mask_labels = []
        x = []
        for item in batch_flag:
            x.append(torch.tensor(item, dtype=torch.long))
            mask_labels.append((torch.rand(len(item)) > 0.85).long())

        x_padded = pad_sequence(x, batch_first=True)
        mask_labels_padded = pad_sequence(mask_labels, batch_first=True)
        return [x_padded, mask_labels_padded]
