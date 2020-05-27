import numpy as np
import pandas as pd


def sigmoid(logits):
    return 1 / (1 + np.exp(-logits))


logits = [[100, 200, 300], [0, 2, -1]]

res = sigmoid(np.concatenate(logits))
print(res)
