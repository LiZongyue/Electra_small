import os
import torch
import pandas as pd
import numpy as np

from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class SupTextDataset(Dataset):
    """
    Subclass DataSet
    """

    def __init__(self, tokenizer, file_path: str, block_size=128):
        assert os.path.isfile(file_path)
        if '.tsv' in file_path:
            data = pd.read_csv(file_path, sep='\t')
        else:
            data = pd.read_csv(file_path)

        lines = data['sentences'].tolist()

        batch_encoding = tokenizer.batch_encode_plus(lines, add_special_tokens=True, max_length=block_size,
                                                     pad_to_max_length=False)
        self.examples = []
        self.examples = batch_encoding["input_ids"]
        self.labels = np.array(data['labels'].tolist())

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return torch.tensor(self.examples[index], dtype=torch.long), torch.tensor(self.labels[index], dtype=torch.long)

    @staticmethod
    def collate_func(batch):
        x, y = zip(*batch)
        x_padded = pad_sequence(x, batch_first=True)
        y_padded = []
        for item in y:
            y_padded.append([item.long()])
        y_padded = torch.tensor(y_padded)
        return [x_padded, y_padded]