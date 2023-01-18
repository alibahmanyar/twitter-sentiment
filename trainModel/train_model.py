#! /bin/python3
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import datetime

VOCAB_SIZE = 80_000  # full vocab for 1.6M dataset contains 850061 tokens

EPOCHS = 40
BATCH_SIZE = 128

train_dataset = "datasets/split/train.csv"
validation_dataset = "datasets/split/validation.csv"

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
checkpoint_filepath = 'tmp/checkpoint'


def load_dataset():
    tdf = pd.read_csv(train_dataset)
    vdf = pd.read_csv(validation_dataset)
    
    print("train,val lengths:", len(tdf), len(vdf))

    train_data = tdf['text'].to_numpy()
    train_label = tdf['polarity'].to_numpy()

    val_data = vdf['text'].to_numpy()
    val_label = vdf['polarity'].to_numpy()

    return train_data, train_label, val_data, val_label


def build_encoder(train_data):
    encoder = tf.keras.layers.TextVectorization(max_tokens=VOCAB_SIZE)
    encoder.adapt(train_data)
    return encoder

def build_model(encoder):
    model = tf.keras.Sequential([
        encoder,
        tf.keras.layers.Embedding(
            input_dim=len(encoder.get_vocabulary()),
            output_dim=64,
            mask_zero=True),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, kernel_regularizer=tf.keras.regularizers.L2(0.01))),
        # tf.keras.layers.Dropout(0.33),
        tf.keras.layers.Dense(64, kernel_regularizer=tf.keras.regularizers.L2(0.01), activation='relu'),
        tf.keras.layers.Dropout(0.33),
        tf.keras.layers.Dense(1)
    ])
    return model


def train_model(model, train_data, train_label, val_data, val_label):
    # model = keras.models.load_model(checkpoint_filepath+"/05-0.79") # continue from previously trained model 

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0)
    backup_callback = tf.keras.callbacks.BackupAndRestore(backup_dir="tmp/backup")
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath+"/{epoch:02d}-{val_accuracy:.2f}",
        monitor='val_accuracy',
        mode='max',
        save_freq='epoch',
        period=10,
        save_best_only=True)

    history = model.fit(x=train_data, y=train_label, validation_data=(val_data, val_label), epochs=EPOCHS, batch_size=BATCH_SIZE,
                    callbacks=[tensorboard_callback, model_checkpoint_callback, backup_callback])

    return history

def main():
    train_data, train_label, val_data, val_label = load_dataset()

    encoder = build_encoder(train_data)
    model = build_model(encoder)
    train_model(model, train_data, train_label, val_data, val_label)

    model.save(f"{checkpoint_filepath}/final")

    

if __name__ == '__main__':
    main()
