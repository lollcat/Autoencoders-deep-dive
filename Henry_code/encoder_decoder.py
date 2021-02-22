import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import initializers,activations, regularizers, constraints, layers
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import (Input, Dense, Lambda, Flatten, Reshape, BatchNormalization, Layer,
                          Activation, Dropout, Conv2D, Conv2DTranspose, Convolution2D, AveragePooling2D,
                          Concatenate, Add, Multiply, UpSampling2D)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
from tensorflow.keras import metrics
from tensorflow.keras import backend as K
from tensorflow.keras import backend as k
from tensorflow.keras.datasets import mnist
import scipy.stats as stats
from tensorflow.keras.layers import InputSpec



latent_dim = 2
h_dim = 7


# given mean and log variance provided by encoder -> sample z
class Sampling(Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


# decoder
def decoder(input_shape=(latent_dim,)):
    model_input = Input(shape=input_shape)
    activation = 'relu'

    # DECODER
    x = Dense(7 * 7 * 96, activation='relu')(model_input)
    x = Reshape((7, 7, 96))(x)
    x = Convolution2D(96, 3, padding='same', activation=activation, name='CONV-1')(x)
    x = UpSampling2D((2, 2))(x)
    x = Convolution2D(64, 3, padding='same', activation=activation, name='CONV-2')(x)
    x = UpSampling2D((2, 2))(x)
    x = Convolution2D(28, 3, padding='same', activation=activation, name='CONV-3')(x)
    decoded = Convolution2D(1, 3, padding='same', activation='sigmoid', name='CONV-4')(x)

    decoder = Model(inputs=model_input, outputs=[decoded], name="decoder")
    # decoder.summary()
    return decoder


# encoder
def encoder(input_shape=(28, 28, 1)):
    model_input = Input(shape=input_shape)

    # ENCODER
    activation = 'relu'
    x = Convolution2D(28, 3, padding='same', activation=activation, name='CONV-1')(model_input)
    x = AveragePooling2D(pool_size=2, name='AVG-POOL-1')(x)
    # x = Dropout(0.1, name='DROPOUT-1')(x)
    x = Convolution2D(64, 3, padding='same', activation=activation, name='CONV-2')(x)
    x = AveragePooling2D(pool_size=2, name='AVG-POOL-2')(x)
    # x = Dropout(0.1, name='DROPOUT-2')(x)
    x = Convolution2D(96, 3, padding='same', activation=activation, name='CONV-3')(x)
    encoder_features = Flatten(name='FLATTEN')(x)

    # SAMPLE means & std
    z_mean = Dense(latent_dim, activation='linear', name='latent-mean')(encoder_features)
    z_log_var = Dense(latent_dim, activation='linear', name='latent-std')(encoder_features)
    h = Dense(h_dim, activation='linear', name='h_embeding')(encoder_features)
    z_sample = Sampling()([z_mean, z_log_var])

    encoder = Model(inputs=model_input, outputs=[z_mean, z_log_var, z_sample, h], name="encoder")
    # encoder.summary()
    return encoder



