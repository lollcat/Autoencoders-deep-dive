from tensorflow.keras.layers import Dense, Flatten, Conv2D, Layer
import tensorflow as tf
import numpy as np
# from scipy.stats import norm # to cross check normal working as desired

class Encoder(Layer):
    def __init__(self, latent_representation_dim, layer_nodes=64, name="encoder"):
        super(Encoder, self).__init__()
        #self.latent_representation_dim = latent_representation_dim
        self.conv1 = Conv2D(32, 3, activation='relu')
        self.flatten = Flatten()
        self.layer1 = Dense(layer_nodes, activation="relu")
        self.layer2 = Dense(layer_nodes, activation="relu")
        self.means = Dense(latent_representation_dim, activation="linear")
        self.log_stds = Dense(latent_representation_dim, activation="linear")


    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.flatten(x)
        x = self.layer1(x)
        x = self.layer2(x)
        means = self.means(x)
        log_stds = self.log_stds(x)

        # Now do sampling with reparameterisation trick
        epsilon = np.random.standard_normal(means.shape)
        z = means + epsilon * tf.math.exp(log_stds)
        # add log_std as this is the jacobian for the change of variables of epsilon to z
        log_probs_z_given_x = self.independent_normal_log_prob(epsilon) - tf.math.reduce_sum(log_stds, axis=1)

        # prior probability (N(0,1))
        log_prob_z_prior = self.independent_normal_log_prob(z)

        return z, log_probs_z_given_x, log_prob_z_prior

    def independent_normal_log_prob(self, x):
        return -0.5*tf.reduce_sum(x**2 + tf.math.log(2*np.pi), axis=1)



if __name__ == "__main__":
    tf.config.run_functions_eagerly(True)
    from Utils.load_data import x_test
    latent_representation_dim = 32
    encoder = Encoder(latent_representation_dim)
    minitest = x_test[0:50, :, :]
    encoder_test = encoder(minitest)
    print(encoder_test[0].shape, encoder_test[1].shape, encoder_test[2].shape)

