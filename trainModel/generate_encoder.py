#! /bin/python3
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import pickle

VOCAB_SIZE = 300_000  # Full vocab for 1.6M dataset contains 850061 tokens

path_to_glove_file = "glove/glove.txt"
path_to_dataset = "datasets/train.csv"
path_to_objects = "objects"

df = pd.read_csv(path_to_dataset, encoding = "ISO-8859-1", names=['polarity', 'id', 'query', 'user', 'text'], index_col=2)
df = df.sample(frac=1)[:300] # Shuffle and truncate

# We train the encoder with all the data (test data is in a separate file so we won't have leakage, but it's still debatable...)
train_data = df['text'].to_numpy()


# Train the encoder
encoder = tf.keras.layers.TextVectorization(max_tokens=VOCAB_SIZE)
encoder.adapt(train_data)

vocab = np.array(encoder.get_vocabulary())
word_index = dict(zip(vocab, range(len(vocab))))

print("Vaocab length:", len(vocab))

# Read all GloVe embeddings
embeddings_index = {}
with open(path_to_glove_file) as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, "f", sep=" ")
        embeddings_index[word] = coefs


num_tokens = len(vocab)
embedding_dim = 200
hits = 0
misses = 0

# Prepare embedding matrix
embedding_matrix = np.zeros((num_tokens, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # Words not found in embedding index will be all-zeros.
        # This includes the representation for "padding" and "OOV"
        embedding_matrix[i] = embedding_vector
        hits += 1
    else:
        misses += 1

print("Converted %d words (%d misses)" % (hits, misses))

# Create the embedding layer from embedding matrix
embedding_layer = tf.keras.layers.Embedding(
    num_tokens,
    embedding_dim,
    embeddings_initializer=keras.initializers.Constant(embedding_matrix),
    trainable=False,
    mask_zero=True
)

# Serializing generated objects with pickle
with open(f'{path_to_objects}/embedding_matrix.pickle', 'wb') as f:
    pickle.dump(embedding_matrix, f)

with open(f'{path_to_objects}/embedding_layer.pickle', 'wb') as f:
    pickle.dump(embedding_layer, f)

with open(f'{path_to_objects}/encoder.pickle', 'wb') as f:
    # Can't save the whole model because of some weird bug: https://stackoverflow.com/questions/65103526/how-to-save-textvectorization-to-disk-in-tensorflow
    pickle.dump({'config': encoder.get_config(),
                'weights': encoder.get_weights()}
                , f)