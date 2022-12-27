#!/usr/bin/python3
import os
from flask import Flask, request, redirect, Response, jsonify, send_from_directory
from markupsafe import escape
from flask_cors import CORS, cross_origin
from tensorflow import keras

app = Flask(__name__)
CORS(app)

model_filepath = 'model/final'
model = keras.models.load_model(model_filepath)

@app.route('/predict', methods=['GET'])
@cross_origin()
def upload_media():
    text = request.values.get('text')
    if text is None or len(text) == 0:
        return "", 400
    
    return jsonify({
        "status": 0,
        "msg": "success",
        "result": float(model.predict([text])[0][0])
    })

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")