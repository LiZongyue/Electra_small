
import pandas as pd


def main():

    '''
    obj = DataGenerator()
    obj.data_saver("C:/Users/Zongyue Li/Documents/GitHub/BNP/Data/aclImdb/train/pos",
                   "C:/Users/Zongyue Li/Documents/GitHub/BNP/Data/aclImdb/train/neg")

    obj.data_saver("C:/Users/Zongyue Li/Documents/GitHub/BNP/Data/aclImdb/test/pos",
                   "C:/Users/Zongyue Li/Documents/GitHub/BNP/Data/aclImdb/test/neg")
    '''
    data = pd.read_csv("C:/Users/Zongyue Li/Documents/GitHub/BNP/Data/aclImdb/train/train_data.csv")
    print(data.shape)


if __name__ == "__main__":
    main()
