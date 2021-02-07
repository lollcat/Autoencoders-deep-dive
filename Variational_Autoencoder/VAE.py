from tensorflow.keras import Model
import tensorflow as tf
from Variational_Autoencoder.Decoder import Decoder
from Variational_Autoencoder.Encoder_diag_cov import Encoder
import numpy as np

class VAE(Model):
    def __init__(self, latent_representation_dim, x_dim, layer_nodes=64):
        super(VAE, self).__init__()
        self.encoder = Encoder(latent_representation_dim, layer_nodes)
        self.decoder = Decoder(x_dim, layer_nodes)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    def call(self, x):
        z, log_probs_z_given_x, log_prob_z_prior = self.encoder(x)
        decoded_logits = self.decoder(z)
        return decoded_logits, log_probs_z_given_x, log_prob_z_prior


    def get_encoding(self, x):
        z, log_probs_z_given_x, log_prob_z_prior = self.encoder(x)
        return z

    @tf.function
    def train_step(self, x_data):
        with tf.GradientTape() as tape:
            decoded_logits, log_probs_z_given_x, log_prob_z_prior = self(x_data)
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
    tf.config.run_functions_eagerly(True)
    from Utils.load_data import x_test, image_dim
    minitest = x_test[0:50, :, :]

    latent_representation_dim = 32
    vae = VAE(latent_representation_dim, image_dim)
    decoded, log_probs_z_given_x, log_prob_z_prior = vae(minitest)
    print(decoded.shape, log_probs_z_given_x.shape)
    ELBO, log_prob_x_given_z_decode_batch, log_probs_z_given_x_batch, log_prob_z_prior_batch = \
        vae.train_step(minitest)
    print(ELBO)