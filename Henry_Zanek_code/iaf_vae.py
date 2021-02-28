import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import (Input, Dense, Lambda, Flatten, Reshape, BatchNormalization, Layer,
                          Activation, Dropout, Conv2D, Conv2DTranspose, Convolution2D, AveragePooling2D,
                          Concatenate, Add, Multiply, UpSampling2D)
from tensorflow.keras import backend as K
from Henry_Zanek_code.encoder_decoder import encoder, decoder

from Henry_Zanek_code.MADE import MADE


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
        z_mean, z_log_var, z_sample, epsilon, h = self.encoder(inputs)
        qz_x = -tf.reduce_sum(0.5*tf.square(epsilon) + 0.5*K.log(np.pi*2) + z_log_var, axis=-1)
        # for the below, given we sample from q(z|x), it should give higher probability to the samples on average
        # q(z_x)
        # -tf.reduce_sum(0.5 * (tf.square(z_sample) + K.log(np.pi * 2)), axis=-1)
        for autoregressor in self.autoregressors:
            autoregressive_input = Concatenate()([z_sample, h])
            mu_new, s = autoregressor(autoregressive_input)
            s = s / 10 + 1.5
            sig = tf.math.sigmoid(s)

            z_sample = tf.keras.layers.Multiply()([sig, z_sample]) + tf.keras.layers.Multiply()([(1 - sig), mu_new])
            qz_x -= tf.reduce_sum(K.log(sig), axis=-1)
            z_sample = K.reverse(z_sample, axes=-1)

        p_z = -tf.reduce_sum(0.5 * (tf.square(z_sample) + K.log(np.pi * 2)), axis=-1)
        decoder_output = self.decoder(z_sample)

        return decoder_output, qz_x, p_z

    def train_step(self, inputs, training=True):
        with tf.GradientTape() as tape:
            decoder_output, qz_x, p_z = self(inputs)
            #p_x_given_z = - tf.reduce_sum(
            #        keras.losses.binary_crossentropy(inputs, decoder_output), axis=(1, 2))
            p_x_given_z = -tf.reduce_sum(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=inputs, logits=decoder_output), axis=[1, 2])

            qz_x_per_batch = tf.reduce_mean(qz_x)
            p_z_per_batch = tf.reduce_mean(p_z)
            p_x_given_z_per_batch = tf.reduce_mean(p_x_given_z)

            ELBO = p_x_given_z_per_batch + p_z_per_batch - qz_x_per_batch
            loss = -ELBO
        variables = self.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        return ELBO, p_x_given_z_per_batch, qz_x_per_batch,  p_z_per_batch

    def test_step(self, inputs, training=False):
        with tf.GradientTape() as tape:
            decoder_output, qz_x, p_z = self(inputs)
            #p_x_given_z = - tf.reduce_sum(
            #        keras.losses.binary_crossentropy(inputs, decoder_output), axis=(1, 2))
            p_x_given_z = -tf.reduce_sum(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=inputs, logits=decoder_output), axis=[1, 2])

            qz_x_per_batch = tf.reduce_mean(qz_x)
            p_z_per_batch = tf.reduce_mean(p_z)
            p_x_given_z_per_batch = tf.reduce_mean(p_x_given_z)

            ELBO = p_x_given_z_per_batch + p_z_per_batch - qz_x_per_batch
            loss = -ELBO
        return ELBO, p_x_given_z_per_batch, qz_x_per_batch,  p_z_per_batch


if __name__ == "__main__":
    from Utils.load_binarized_mnist import x_train, x_test, train_ds, test_ds, image_dim
    iaf_vae = IAF_VAE_Class()
    clump_of_inputs = x_train[0:10, :, :, :]
    test_forward_pass_output = iaf_vae(clump_of_inputs)
    ELBO = iaf_vae.train_step(clump_of_inputs)
    print(ELBO)


