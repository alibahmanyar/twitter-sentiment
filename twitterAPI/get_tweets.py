#! /bin/python3
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import datetime
import json
import pickle
import requests
from requests_oauthlib import OAuth1Session

with open("keys.json", 'r') as f:
    token = json.loads(f.read())['bearer']

model_path = '../trainModel/tmp/checkpoint/final'
model = keras.models.load_model(model_path)

search_url = "https://api.twitter.com/2/tweets/search/recent"

query_params = {'query': '#fifaworldcup lang:en -is:retweet -is:reply -has:media', 'max_results': 100}

r = requests.get(search_url, params=query_params, headers={"Authorization": f"Bearer {token}"}, timeout=10)
# r.json()
tweets = [x['text'] for x in r.json()['data']]

for t in tweets:
    print(f'{t}\t{model.predict([t])}')
