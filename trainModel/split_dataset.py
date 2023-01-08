#! /usr/bin/python3
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

DATASET = "datasets/train.csv"
DATASET_LEN = 180_000
SPLIT = 0.2

if __name__ == '__main__':
    df = pd.read_csv(DATASET, encoding = "ISO-8859-1", names=['polarity', 'id', 'query', 'user', 'text'], index_col=2)
    df = df.sample(frac=1)[:DATASET_LEN] # shuffle and truncate
    df['polarity'] = df['polarity'].apply(lambda x: 1 if x == 4 else 0)


    # df = pd.read_csv("datasets/test2.csv", encoding = "ISO-8859-1")
    # df = df.sample(frac=1)[:] # shuffle and truncate
    # df.drop(df.loc[df['sentiment']=="Neutral"].index, inplace=True)
    # df['polarity'] = df['sentiment'].apply(lambda x: 1 if x == "Positive" else 0)

    tdf, vdf = train_test_split(df, test_size=SPLIT)
    
    tdf.to_csv("datasets/split/train.csv")
    tdf.to_csv("datasets/split/validation.csv")
