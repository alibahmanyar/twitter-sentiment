#!/usr/bin/env python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

from tensorflow import keras
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import sklearn


map_polarity = {0:0, 4:2, 2:1}

def map_model_out(x):
    # return x
    if x < 0.5:
        return 0
    elif x > 0.5:
        return 1
    return 2


MODEL_NO = 0 # replaced by sed

checkpoint_filepath = f'/media/ali/Drive/Static/NLP/twitter-sentiment/Models/3/tmp/checkpoint/68-0.88'

model = keras.models.load_model(f"{checkpoint_filepath}")


print(model.summary())





print("\n\nSentiment 140:")
df = pd.read_csv("datasets/test.csv", encoding = "ISO-8859-1", names=['polarity', 'id', 'query', 'user', 'text'], index_col=2)
df = df.sample(frac=1)[:] # shuffle and truncate
df.drop(df.loc[df['polarity']==2].index, inplace=True)
print(set(df['polarity'].values))

# real_labels = df['polarity'].map(lambda x: map_polarity[x])
df['polarity'] = df['polarity'].apply(lambda x: 1 if x == 4 else 0)

test_data = df['text'].to_numpy()
test_label = df['polarity'].to_numpy()

predictions = (model.predict(test_data, verbose=0))
predictions = [map_model_out(x) for x in predictions]

print(sklearn.metrics.confusion_matrix(test_label, predictions))
print(list(predictions - test_label).count(0) / len(predictions))

print(model.evaluate(test_data, test_label, verbose=0))



print("\n\nElections tweets:")
df = pd.read_csv("datasets/test2.csv", encoding = "ISO-8859-1")
df = df.sample(frac=1)[:] # shuffle and truncate
df.drop(df.loc[df['sentiment']=="Neutral"].index, inplace=True)
print(set(df['sentiment'].values))

df['sentiment'] = df['sentiment'].apply(lambda x: 1 if x == "Positive" else 0)

test_data = df['text'].to_numpy()
test_label = df['sentiment'].to_numpy()

predictions = (model.predict(test_data, verbose=0))
predictions = [map_model_out(x) for x in predictions]

print(sklearn.metrics.confusion_matrix(test_label, predictions))
print(list(predictions - test_label).count(0) / len(predictions))

print(model.evaluate(test_data, test_label, verbose=0))

