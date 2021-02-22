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
from Henry_code.encoder_decoder import encoder, decoder

from Henry_code.MADE import MADE


num_auto_regressor = 3

class IAF_VAE_Class(Model):
    """
    Notes:
        Sigmoid instead of exp in autoregressive steps
        Need to watch out for shapes of things and make sure they are as you desired for the forward pass (this is the
        most standard type of bug I get in tensorflow)
    """
    def __init__(self, num_auto_regressor=3, latent_dim=2, h_dim=7):
        super(IAF_VAE_Class, self).__init__()
        self.encoder = encoder()
        self.decoder = decoder(input_shape=(latent_dim,))
        self.autoregressors = []
        for t in range(num_auto_regressor):
            self.autoregressors.append(
                MADE(z_in=latent_dim, hidden_sizes=[30, 30], random_input_order=False, h_in=h_dim).build_model())

        self.optimizer = tf.keras.optimizers.Adam(1e-3)

    def call(self, inputs):
        z_mean, z_log_var, z_sample, h = self.encoder(inputs)
        qz_x = tf.reduce_sum(-0.5 * (tf.square(z_sample) + K.log(np.pi * 2)) - z_log_var, axis=1)
        for autoregressor in self.autoregressors:
            autoregressive_input = Concatenate()([z_sample, h])
            mu_new, log_sigma_new = autoregressor(autoregressive_input)
            #s = tf.nn.sigmoid(log_sigma_new)
            z_sample = tf.keras.layers.Multiply()(
                [tf.keras.backend.exp(log_sigma_new), z_sample]) + tf.keras.layers.Multiply()(
                [(1 - tf.keras.backend.exp(log_sigma_new)), mu_new])
            # note: tf.reduce sum reduces along all axis if axis not specified
            # should be tf.reduce_sum(axis=-1), as you don't want to reduce over batch size
            qz_x -= tf.reduce_sum(log_sigma_new, -1)

        p_z = tf.reduce_sum(-0.5 * (tf.square(z_sample) + K.log(np.pi * 2)), axis=-1)
        decoder_output = self.decoder(z_sample)

        return decoder_output, qz_x, p_z # note p_z_x and p_z are actually log probs

    def train_step(self, inputs, training=True):
        with tf.GradientTape() as tape:
            decoder_output, qz_x, p_z = self(inputs)
            p_x_given_z = - tf.reduce_sum(
                    keras.losses.binary_crossentropy(inputs, decoder_output), axis=(1, 2))

            qz_x_per_batch = tf.reduce_mean(p_x_given_z)
            p_z_per_batch = tf.reduce_mean(p_z)
            p_x_given_z_per_batch = tf.reduce_mean(p_x_given_z)

            ELBO = p_x_given_z_per_batch + p_z_per_batch - qz_x_per_batch
            loss = -ELBO
        variables = self.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        return ELBO


def IAF_VAE(encoder, decoder, latent_dim=2, h_dim=7):
    # start with the encoder
    encoder_input = Input(shape=(28, 28, 1))
    z_mean, z_log_var, z_sample, h = encoder(encoder_input)

    qz_x = tf.reduce_mean(tf.reduce_sum(-0.5 * (tf.square(z_sample) + K.log(np.pi * 2)) - z_log_var, axis=1))


    # creating auto-regressor
    autoregressor = []
    for t in range(num_auto_regressor):
        autoregressor.append(
            MADE(z_in=latent_dim, hidden_sizes=[30, 30], random_input_order=False, h_in=h_dim).build_model())
        autoregressive_input = Concatenate()([z_sample, h])
        mu_new, log_sigma_new = autoregressor[t](autoregressive_input)

        # note: a think this is meant to be tf.nn.sigmoid - see algorithm 1 in IAF paper
        z_sample = tf.keras.layers.Multiply()(
            [tf.keras.backend.exp(log_sigma_new), z_sample]) + tf.keras.layers.Multiply()(
            [(1 - tf.keras.backend.exp(log_sigma_new)), mu_new])
        qz_x -= tf.reduce_sum(log_sigma_new)

    p_z = tf.reduce_mean(tf.reduce_sum(-0.5 * (tf.square(z_sample) + K.log(np.pi * 2))))
    decoder_output = decoder(z_sample)

    iaf = Model(inputs=[encoder_input], outputs=[decoder_output], name="iaf")

    # ADD LOSS FUNCTION
    reconstruction_loss = tf.reduce_mean(
        tf.reduce_sum(
            keras.losses.binary_crossentropy(encoder_input, decoder_output), axis=(1, 2)
        )
    )
    total_loss = reconstruction_loss - p_z + qz_x
    iaf.add_loss(total_loss)
    iaf.add_metric(p_z, name="p_z")
    iaf.add_metric(qz_x, name="qz_x")
    iaf.add_metric(reconstruction_loss, name="reconstruction loss")
    # vae.summary()
    return iaf


if __name__ == "__main__":
    from Utils.load_binarized_mnist import x_train, x_test, train_ds, test_ds, image_dim
    iaf_vae = IAF_VAE_Class()
    clump_of_inputs = x_train[0:10, :, :, :]
    test_forward_pass_output = iaf_vae(clump_of_inputs)
    ELBO = iaf_vae.train_step(clump_of_inputs)


