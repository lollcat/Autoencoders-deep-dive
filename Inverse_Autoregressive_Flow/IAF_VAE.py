from tensorflow.keras import Model
import tensorflow as tf
from Inverse_Autoregressive_Flow.Encoder import IAF_Encoder
from Inverse_Autoregressive_Flow.Decoder import Decoder

import numpy as np

class IAF_VAE(Model):
    """
    TODO maybe rewrite the hyper parameter config?
    """
    def __init__(self, latent_representation_dim, x_dim, layer_nodes=64, n_autoregressive_units=3):
        super(IAF_VAE, self).__init__()

        # currently setting all node numbers to be equal to layer_nodes
        self.encoder = IAF_Encoder(latent_representation_dim=latent_representation_dim, layer_nodes=64,
                                   n_autoregressive_units=n_autoregressive_units,
                                   autoregressive_unit_layer_width=layer_nodes,
                                   First_Encoder_to_IAF_step_dim=layer_nodes)

        self.decoder = Decoder(x_dim, layer_nodes)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    def call(self, x, training=False):
        z, log_probs_z_given_x, log_prob_z_prior = self.encoder(x)
        decoded_logits = self.decoder(z)
        return decoded_logits, log_probs_z_given_x, log_prob_z_prior


    def get_encoding(self, x):
        z, log_probs_z_given_x, log_prob_z_prior = self.encoder(x)
        return z

    @tf.function
    def train_step(self, x_data, training=True):
        with tf.GradientTape() as tape:
            decoded_logits, log_probs_z_given_x, log_prob_z_prior = self(x_data, training=training)
            # TODO - maybe consider making x_data flat so we don't have to double reduce sum
            #log_prob_x_given_z_decode = tf.reduce_sum(x_data * tf.math.log(reconstruction) + \
            #                                          (1 - x_data) * tf.math.log(1 - reconstruction), axis=[1,2])
            log_prob_x_given_z_decode = -tf.reduce_sum(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=x_data, logits=decoded_logits), axis=[1, 2, 3])
            # compute mean over batch
            log_probs_z_given_x_batch = tf.reduce_mean(log_probs_z_given_x)
            log_prob_z_prior_batch = tf.reduce_mean(log_prob_z_prior)
            log_prob_x_given_z_decode_batch = tf.reduce_mean(log_prob_x_given_z_decode)
            # compute ELBO
            ELBO = log_prob_x_given_z_decode_batch + log_prob_z_prior_batch - log_probs_z_given_x_batch
            loss = -ELBO
        variables = self.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        return ELBO, log_prob_x_given_z_decode_batch, log_probs_z_given_x_batch, log_prob_z_prior_batch


    @tf.function
    def test_step(self, x_data):
        decoded_logits, log_probs_z_given_x, log_prob_z_prior = self(x_data)
        log_prob_x_given_z_decode = -tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=x_data, logits=decoded_logits), axis=[1, 2])
        # compute mean over batch
        log_probs_z_given_x_batch = tf.reduce_mean(log_probs_z_given_x)
        log_prob_z_prior_batch = tf.reduce_mean(log_prob_z_prior)
        log_prob_x_given_z_decode_batch = tf.reduce_mean(log_prob_x_given_z_decode)
        # compute ELBO
        ELBO = log_prob_x_given_z_decode_batch + log_prob_z_prior_batch - log_probs_z_given_x_batch
        return ELBO, log_prob_x_given_z_decode_batch, log_probs_z_given_x_batch, log_prob_z_prior_batch


if __name__ == "__main__":
    # just do this file thing so we don't have to mess around with pycharm configs here
    from pathlib import Path; import os
    cwd_path = Path.cwd(); set_path = str(cwd_path.parent); os.chdir(set_path)

    # Let's go!
    tf.config.run_functions_eagerly(True)
    from Utils.load_plain_mnist import x_test, image_dim
    minitest = x_test[0:50, :, :]

    latent_representation_dim = 32
    vae = IAF_VAE(latent_representation_dim, image_dim)
    decoded, log_probs_z_given_x, log_prob_z_prior = vae(minitest)
    print(decoded.shape, log_probs_z_given_x.shape)
    ELBO, log_prob_x_given_z_decode_batch, log_probs_z_given_x_batch, log_prob_z_prior_batch = \
        vae.train_step(minitest)
    print(ELBO)