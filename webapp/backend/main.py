#!/usr/bin/python3
import os
from flask import Flask, request, redirect, Response, jsonify, send_from_directory
from markupsafe import escape
from flask_cors import CORS, cross_origin
from tensorflow import keras

import json
import requests

app = Flask(__name__)
CORS(app)

model_filepath = 'model/final'
model = keras.models.load_model(model_filepath)

with open("keys.json", 'r') as f:
        token = json.loads(f.read())['bearer']

@app.route('/predict', methods=['GET'])
@cross_origin()
def predict():
    text = request.values.get('text')
    if text is None or len(text) == 0:
        return "", 400
    
    return jsonify({
        "status": 0,
        "msg": "success",
        "result": float(model.predict([text])[0][0])
    })


@app.route('/get_tweets', methods=['GET'])
@cross_origin()
def get_tweets():
    search_term = request.values.get('search_term')
    if search_term is None or len(search_term) == 0:
        return "", 400

    search_url = "https://api.twitter.com/2/tweets/search/recent"

    query_params = {'query': f'{search_term} lang:en -is:retweet -is:reply -has:media', 'max_results': 10}

    r = requests.get(search_url, params=query_params, headers={"Authorization": f"Bearer {token}"}, timeout=10)
    
    tweets_text = [x['text'] for x in r.json()['data']]
    tweets = []

    for t in tweets_text:
        tweets.append({'text': t, 'score': float(model.predict([t])[0][0])})

    return jsonify({
        "status": 0,
        "msg": "success",
        "result": tweets
    })

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")