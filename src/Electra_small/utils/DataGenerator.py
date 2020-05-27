import os
from termcolor import colored
from tqdm import tqdm
import pandas as pd


class DataGenerator:

    @staticmethod
    def data_utils(file_path: str, positive: bool):
        files = os.listdir(file_path)  # get all files under the dir
        txts = []
        print(colored("Pre processing the data...", "red"))
        for file in tqdm(files):  # iterate the dir
            position = file_path + '/' + file  # construct path with "/"
            with open(position, "r", encoding='utf-8') as f:  # open file
                data = f.read()  # read file
                data = data.replace('<br />', '')
                txts.append(data)

        if positive:
            labels = [1] * len(txts)
        else:
            labels = [0] * len(txts)
        transformer = {"sentences": txts, "labels": labels}
        data = pd.DataFrame(transformer)

        return data

    def data_saver(self, file_path_pos: str, file_path_neg: str):
        data_pos = self.data_utils(file_path_pos, True)
        data_neg = self.data_utils(file_path_neg, False)

        data = pd.concat([data_neg, data_pos], ignore_index=True)
        if "train" in file_path_pos and "train" in file_path_neg:
            data.to_csv('C:/Users/Zongyue Li/Documents/GitHub/BNP/Data/aclImdb/train/train_data.csv', encoding='utf-8')
        elif "test" in file_path_pos and "test" in file_path_neg:
            data.to_csv('C:/Users/Zongyue Li/Documents/GitHub/BNP/Data/aclImdb/test/test_data.csv', encoding='utf-8')
        else:
            raise Exception("Data cannot be generated.")
