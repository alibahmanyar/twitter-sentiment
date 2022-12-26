#!/bin/python3
import tensorflow as tf
from tensorflow import keras

model_path = 'tmp/checkpoint/final'
# converter = tf.lite.TFLiteConverter.from_saved_model(model_path) # path to the SavedModel directory
# tflite_model = converter.convert()

# # Save the model.
# with open('model.tflite', 'wb') as f:
#   f.write(tflite_model)


# https://github.com/tensorflow/tensorflow/issues/53333
from tensorflow import lite

model = keras.models.load_model(model_path)
converter = lite.TFLiteConverter.from_keras_model(model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.experimental_new_converter=True
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
tf.lite.OpsSet.SELECT_TF_OPS]

tfmodel = converter.convert()
open('image_model.tflite', 'wb').write(tfmodel)