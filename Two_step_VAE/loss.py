import tensorflow_addons as tfa
import tensorflow as tf
import keras
from tensorflow.keras import layers


class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def mse(y_true, y_pred):
    return tf.reduce_mean(tf.reduce_sum(tf.math.squared_difference(y_true, y_pred), axis=(1, 2)))*1000


def mse_keras(y_true, y_pred):
    score = tf.py_function(func=mse, inp=[y_true, y_pred], Tout=tf.float32, name='mse')
    return score
