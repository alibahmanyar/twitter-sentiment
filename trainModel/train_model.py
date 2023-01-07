#! /bin/python3
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import datetime
import pickle


EPOCHS = 80
BATCH_SIZE = 128

path_to_glove_file = "glove/glove.txt"
path_to_dataset = "datasets/train.csv"
path_to_objects = "objects"

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
checkpoint_filepath = 'tmp/checkpoint'


def load_dataset():
    df = pd.read_csv(path_to_dataset, encoding = "ISO-8859-1", names=['polarity', 'id', 'query', 'user', 'text'], index_col=2)
    df = df.sample(frac=1)[:] # shuffle and truncate
    df['polarity'] = df['polarity'].apply(lambda x: 1 if x == 4 else 0)

    tdf, vdf = train_test_split(df, test_size=0.02)
    train_data = tdf['text'].to_numpy()
    train_label = tdf['polarity'].to_numpy()

    val_data = vdf['text'].to_numpy()
    val_label = vdf['polarity'].to_numpy()

    return train_data, train_label, val_data, val_label


def load_encoder():
    with open(f'{path_to_objects}/embedding_layer.pickle', 'rb') as f:
        embedding_layer = pickle.load(f)
    
    with open(f'{path_to_objects}/encoder.pickle', 'rb') as f:
        from_disk = pickle.load(f)
        encoder = tf.keras.layers.TextVectorization.from_config(from_disk['config'])
        # You have to call `adapt` with some dummy data (BUG in Keras)
        encoder.adapt(tf.data.Dataset.from_tensor_slices(["xyz"]))
        encoder.set_weights(from_disk['weights'])
    
    return encoder, embedding_layer


def build_model(encoder, embedding_layer):
    model = tf.keras.Sequential([
        encoder,
        embedding_layer,
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    return model


def train_model(model, train_data, train_label, val_data, val_label):
    # model = keras.models.load_model(checkpoint_filepath+"/05-0.79") # continue from previodly 

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0)
    backup_callback = tf.keras.callbacks.BackupAndRestore(backup_dir="tmp/backup")
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath+"/{epoch:02d}-{val_accuracy:.2f}",
        monitor='val_accuracy',
        mode='max',
        save_freq='epoch',
        period=1,
        save_best_only=True)

    history = model.fit(x=train_data, y=train_label, validation_data=(val_data, val_label), epochs=EPOCHS, batch_size=BATCH_SIZE,
                    callbacks=[tensorboard_callback, model_checkpoint_callback, backup_callback])

    return history

def main():
    train_data, train_label, val_data, val_label = load_dataset()
    encoder, embedding_layer = load_encoder()

    model = build_model(encoder, embedding_layer)
    train_model(model, train_data, train_label, val_data, val_label)
    model.save(f"{checkpoint_filepath}/final")
    print(model.predict(np.array(["Shit!", "Oh fuck!", "You'd better shut up!", "I wanna kill this bastard", "That was amazing", "OMG!"
                        "That was cute", "Nice one", "I loved it", "I really liked how he behaved", "He was a nice dude"])))

    

if __name__ == '__main__':
    main()
