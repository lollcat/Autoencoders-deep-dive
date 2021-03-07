from tensorflow.keras import Model
import tensorflow as tf
from Inverse_Autoregressive_Flow.Encoder import IAF_Encoder
from Inverse_Autoregressive_Flow.Decoder import Decoder

import numpy as np

class IAF_VAE(Model):
    """
    TODO maybe rewrite the hyper parameter config?
    """
    def __init__(self, latent_representation_dim, x_dim,
                 n_autoregressive_units=3, autoregressive_unit_layer_width=8000, First_Encoder_to_IAF_step_dim=64,
                 encoder_FC_layer_nodes=450,
                 decoder_layer_width = 450):
        super(IAF_VAE, self).__init__()

        # currently setting all node numbers to be equal to layer_nodes
        self.encoder = IAF_Encoder(latent_representation_dim=latent_representation_dim,
                                   layer_nodes=encoder_FC_layer_nodes,
                                   n_autoregressive_units=n_autoregressive_units,
                                   autoregressive_unit_layer_width=autoregressive_unit_layer_width,
                                   First_Encoder_to_IAF_step_dim=First_Encoder_to_IAF_step_dim)

        self.decoder = Decoder(x_dim, latent_representation_dim)
        #self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        self.optimizer = tf.keras.optimizers.Adamax()

    def call(self, x, training=False):
        z, log_probs_z_given_x, log_prob_z_prior = self.encoder(x)
        decoded_logits = self.decoder(z)
        return decoded_logits, log_probs_z_given_x, log_prob_z_prior


    def get_encoding(self, x):
        z, log_probs_z_given_x, log_prob_z_prior = self.encoder(x)
        return z

    #@tf.function
    def train_step(self, x_data, training=True):
        with tf.GradientTape() as tape:
            decoded_logits, log_probs_z_given_x, log_prob_z_prior = self(x_data, training=training)
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


    #@tf.function
    def test_step(self, x_data):
        decoded_logits, log_probs_z_given_x, log_prob_z_prior = self(x_data)
        log_prob_x_given_z_decode = -tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=x_data, logits=decoded_logits), axis=[1, 2, 3])
        # compute mean over batch
        log_probs_z_given_x_batch = tf.reduce_mean(log_probs_z_given_x)
        log_prob_z_prior_batch = tf.reduce_mean(log_prob_z_prior)
        log_prob_x_given_z_decode_batch = tf.reduce_mean(log_prob_x_given_z_decode)
        # compute ELBO
        ELBO = log_prob_x_given_z_decode_batch + log_prob_z_prior_batch - log_probs_z_given_x_batch
        return ELBO, log_prob_x_given_z_decode_batch, log_probs_z_given_x_batch, log_prob_z_prior_batch

    def get_marginal_likelihood(self, x_data, n_x_data_samples = 128):

        """
        Estimating marginal likelihood using importance sampling as proposed by Rezende et al., 2014
        # calculate MC estimate of p(x) for each point in the test set
        """
        """ Unsure about whether we take the product of datapoints in the test set or the mean? """
        running_mean = 0
        for n in range(n_x_data_samples ):
           decoded_logits, log_probs_z_given_x, log_prob_z_prior = self(x_data)
           log_prob_x_given_z_decode = -tf.reduce_sum(
               tf.nn.sigmoid_cross_entropy_with_logits(labels=x_data, logits=decoded_logits), axis=[1, 2, 3])
           #p_x_z =  tf.math.exp(log_prob_x_given_z_decode + log_prob_z_prior)
           #q_z_given_x = tf.math.exp(log_probs_z_given_x)
           # the above variables are vectors, of length equal to the number of points in x_data
           #monte_carlo_sample = tf.math.divide(p_x_z, q_z_given_x).numpy()
           #log_monte_carlo_sample = log_prob_x_given_z_decode_batch + log_prob_z_prior_batch - log_probs_z_given_x_batch
           log_monte_carlo_sample = log_prob_x_given_z_decode + log_prob_z_prior - log_probs_z_given_x
           monte_carlo_sample = tf.math.exp(tf.cast(log_monte_carlo_sample, "float64"))
           running_mean = running_mean + (monte_carlo_sample - running_mean )/ (n + 1)
        #return np.log(np.mean(running_mean.numpy()))    # averaging over points here also
        #return np.mean(np.log(running_mean.numpy()))    # averaging over points log p(x) 's
        return np.mean(np.log(running_mean.numpy()))      # p(x) for whole test set



if __name__ == "__main__":
    # just do this file thing so we don't have to mess around with pycharm configs here
    from pathlib import Path; import os
    cwd_path = Path.cwd(); set_path = str(cwd_path.parent); os.chdir(set_path)

    # Let's go!
    tf.config.run_functions_eagerly(True)
    from Utils.load_plain_mnist import x_test, image_dim
    minitest = x_test[0:50, :, :]

    latent_representation_dim = 32
    vae = IAF_VAE(latent_representation_dim, x_dim=image_dim,
                n_autoregressive_units=2, autoregressive_unit_layer_width=64,
                First_Encoder_to_IAF_step_dim=32,
                encoder_FC_layer_nodes=64,
                decoder_layer_width = 64)


    decoded, log_probs_z_given_x, log_prob_z_prior = vae(minitest)
    """
    #print(f"decoded, {decoded}")
    print(f"log_probs_z_given_x, {log_probs_z_given_x}")
    #print(f"log_prob_z_prior, {log_prob_z_prior}")
    
    print(decoded.shape, log_probs_z_given_x.shape)
    """
    ELBO, log_prob_x_given_z_decode_batch, log_probs_z_given_x_batch, log_prob_z_prior_batch = \
        vae.train_step(minitest)
    #print(f"log_prob_z_prior_batch is {log_prob_z_prior_batch}")
    print(f"log_probs_z_given_x_batch is {log_probs_z_given_x_batch}")
    print(f"ELBO is {ELBO}")
    #print(f"log_prob_x_given_z_decode_batch is {log_prob_x_given_z_decode_batch}")
    print(f"marginal likelihood {vae.get_marginal_likelihood(minitest)}")

