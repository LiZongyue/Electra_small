import pandas as pd

data = pd.read_csv("C:/Users/Zongyue Li/Documents/Github/BNP/Data/glue_data/SST-2/train.tsv", sep='\t')

print(data.head(5))
