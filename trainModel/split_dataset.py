#! /usr/bin/python3
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

DATASET = "datasets/gop-dataset.csv"
DATASET_LEN = 14_000
SPLIT = 0.1


# map_polarity = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
map_polarity = {'Negative': 0, 'Positive': 1}

if __name__ == '__main__':
    df = pd.read_csv("datasets/gop-dataset.csv", encoding = "ISO-8859-1")
    df = df.sample(frac=1)[:] # shuffle and truncate

    df.drop(df.loc[df['sentiment']=="Neutral"].index, inplace=True)

    df['polarity'] = df['sentiment'].map(lambda x: map_polarity[x])

    tdf, vdf = train_test_split(df, test_size=SPLIT)
    
    tdf.to_csv("datasets/split/train.csv")
    vdf.to_csv("datasets/split/validation.csv")
