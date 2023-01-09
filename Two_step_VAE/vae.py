import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import tensorflow as tf
import keras
import pandas as pd
import numpy as np
from tensorflow.keras.layers import Conv2D, Dense, Conv2DTranspose
from tensorflow.python.keras.models import Model
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from keras.callbacks import ModelCheckpoint, TensorBoard
import time
from biosppy.signals import tools as tools
import neural_structured_learning as nsl
from scipy import stats
import sklearn.preprocessing as skp
from Two_step_VAE.loss import *
import tensorflow.keras.backend as K


def load_vae():

    '''
    2d-vae

    we use 1d vae for generate 6 Limb Lead -> 6 Precordial Lead
    input and output data contain one row zero padding on the top and bottom
    1d-vae take (8,512,1) and final output is (8,512,1)
    we use 25 latent vector, kernel size (1,16), filter of number (64, 64, 128, 128, 256, 256).

    final objective is defined as KL-Divergence + mse*1000

    encoder_inputs=keras.Input(shape=(16,512,1),name='inference_generator_input')
    number_of_filter_encoder = [64,64,128,128,256,256]
    number_of_filter_decoder = [256,256,128,128,64,64]
    kernel=(2,16)
    stride=[2,2,2,1,1,1]

    '''
    number_of_filter_encoder = [64, 64, 128, 128, 256, 256]
    number_of_filter_decoder = [256, 256, 128, 128, 64, 64]
    kernel = (2, 16)
    stride = [2, 2, 2, 1, 1, 1]
    initializer = tf.random_normal_initializer(0., 0.02)

    #  encoder
    encoder_inputs = keras.Input(shape=(8, 512, 1), name='data')

    b_conv = tf.keras.layers.Conv2D(64, kernel, strides=stride[0], padding='same')(encoder_inputs)
    b_batch = tf.keras.layers.BatchNormalization()(b_conv)
    b_output = tf.keras.layers.Activation('relu')(b_batch)

    for i in range(1, 6, 1):
        b_conv = tf.keras.layers.Conv2D(number_of_filter_encoder[i], kernel, strides=stride[i], padding='same')(b_output)
        b_batch = tf.keras.layers.BatchNormalization()(b_conv)
        b_output = tf.keras.layers.Activation('relu')(b_batch)

    b_output = layers.Flatten()(b_output)

    z_mean = layers.Dense(25, name="z_mean")(b_output)
    z_log_var = layers.Dense(25, name='z_log_var')(b_output)
    z = Sampling()([z_mean, z_log_var])

    variational_encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')
    variational_encoder.summary()

    #  decoder
    latent_inputs = keras.Input(shape=(25,))
    b_output = tf.keras.layers.Dense(1*64*256)(latent_inputs)
    b_output = tf.keras.layers.Reshape((1, 64, 256))(b_output)

    for i in range(5):
        b_conv = tf.keras.layers.Conv2DTranspose(number_of_filter_decoder[i], kernel, strides=stride[5-i], padding='same')(b_output)
        b_batch = tf.keras.layers.BatchNormalization()(b_conv)
        b_output = tf.keras.layers.Activation('relu')(b_batch)

    b_output = tf.keras.layers.Conv2DTranspose(number_of_filter_decoder[5], kernel, strides=2, padding='same')(b_output)
    decoder_outputs = tf.keras.layers.Conv2DTranspose(1, kernel, padding='same', name='model')(b_output)
    variational_decoder = keras.Model(latent_inputs, decoder_outputs)
    variational_decoder.summary()

    z_mean, z_log_var, codings = variational_encoder(encoder_inputs)
    reconstructions = variational_decoder(codings)

    latent_loss = -0.5 * K.sum(
        1 + z_log_var - K.exp(z_log_var) - K.square(z_mean),
        axis=1)

    vae = keras.models.Model(inputs=[encoder_inputs], outputs=[reconstructions])
    vae.add_loss(K.mean(latent_loss))
    vae.add_metric(K.mean(latent_loss), name="kl", aggregation="mean")

    return vae
