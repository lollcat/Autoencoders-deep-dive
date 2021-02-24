from tensorflow.keras.layers import Dense, Flatten, Conv2D, Layer, Reshape
#from tensorflow_addons.layers import WeightNormalization
from tensorflow_probability import layers as tfp_layers
WeightNormalization = tfp_layers.weight_norm.WeightNorm
from tensorflow.keras import Model
import tensorflow as tf
import numpy as np
#from Inverse_Autoregressive_Flow.AutoregressiveNN.AutoregressiveNN import AutoRegressiveNN_Unit
from Inverse_Autoregressive_Flow.AutoregressiveNN.attempt_at_norm_autoregressive_layer.AutoregressiveNN import \
    AutoRegressiveNN_Unit
from Inverse_Autoregressive_Flow.resnet import resnet_block

class IAF_Encoder(Model):#Layer):
    def __init__(self, latent_representation_dim, layer_nodes=450, n_autoregressive_units=3,
                 autoregressive_unit_layer_width=8000, First_Encoder_to_IAF_step_dim=64, name="encoder"):
        super(IAF_Encoder, self).__init__()
        self.n_autoregressive_units = n_autoregressive_units
        self.resnet_blocks = []
        self.resnet_blocks.append(resnet_block(filters=16, kernel_size = (3,3), strides=(2,2)))
        self.resnet_blocks.append(resnet_block(filters=16, kernel_size=(3, 3), strides=(1, 1)))
        self.resnet_blocks.append(resnet_block(filters=32, kernel_size = (3,3), strides=(2,2)))
        self.resnet_blocks.append(resnet_block(filters=32, kernel_size=(3, 3), strides=(1, 1)))
        self.resnet_blocks.append(resnet_block(filters=32, kernel_size = (3,3), strides=(2,2)))

        self.flatten = Flatten(name="flatten")
        self.fc_layer = WeightNormalization(Dense(layer_nodes, activation="elu", name="layer1"),
                                            data_init=True)
        self.means = WeightNormalization(Dense(latent_representation_dim, activation="linear", name="means"),
                                         data_init=True)
        self.log_stds = WeightNormalization(Dense(latent_representation_dim, activation="linear", name="log_stds"),
                                                  data_init=True)

        if self.n_autoregressive_units > 0:
            self.First_Encoder_to_IAF = WeightNormalization(Dense(First_Encoder_to_IAF_step_dim, activation="elu",
                                                                  name="Encoder_to_IAF"), data_init=True)
            self.autoregressive_NNs = []
            for i in range(n_autoregressive_units):
                self.autoregressive_NNs.append(
                        AutoRegressiveNN_Unit(latent_representation_dim=latent_representation_dim,
                                          h_dim=First_Encoder_to_IAF_step_dim,
                                          layer_nodes=autoregressive_unit_layer_width))

    def call(self, x, training=False):
        # First Encoder NN
        for resblock in self.resnet_blocks:
            x = resblock(x)
        x = self.flatten(x)
        x = self.fc_layer(x)
        means = self.means(x)
        log_stds = self.log_stds(x)/10  # parameterisation to prevent ensure relatively low initial std
        stds = tf.math.exp(log_stds)  # parameterise to be positive

        # Now do sampling with reparameterisation trick
        epsilon = np.random.standard_normal(means.shape).astype("float32")
        z = means + stds * epsilon
        # add log_std as this is the jacobian for the change of variables of epsilon to z
        log_probs_z_given_x = self.independent_normal_log_prob(epsilon) - tf.math.reduce_sum(log_stds, axis=1)

        # Now IAF steps
        if self.n_autoregressive_units > 0:
            h = self.First_Encoder_to_IAF(x)
            for i in range(self.n_autoregressive_units):
                m, s = self.autoregressive_NNs[i]([z, h])
                sigma = tf.nn.sigmoid(s)
                z = sigma * z + (1 - sigma) * m
                log_probs_z_given_x -= tf.reduce_sum(tf.math.log(sigma), axis=1)
                z = tf.keras.backend.reverse(z, axes=1)  # IAF paper reccomends reversing the order the autoregressive step


        # prior probability (N(0,1))
        log_prob_z_prior = self.independent_normal_log_prob(z)

        return z, log_probs_z_given_x, log_prob_z_prior



    def independent_normal_log_prob(self, x):
        n = x.shape[1]
        return -0.5 * (tf.reduce_sum(x ** 2, axis=1) + n*tf.math.log(2.0 * np.pi))


    def debug_func(self, x):
    # check gradients are working
        with tf.GradientTape(persistent=True) as tape:
            # First Encoder NN
            for resblock in self.resnet_blocks:
                x = resblock(x)
            x = self.flatten(x)
            x = self.flatten(x)
            x = self.fc_layer(x)
            means = self.means(x)
            log_stds = self.log_stds(x)  #/100 # divide by 100 to start initialisation low
            stds = tf.math.exp(log_stds)
            h = self.First_Encoder_to_IAF(x)

            # Now do sampling with reparameterisation trick
            epsilon = np.random.standard_normal(means.shape).astype("float32")
            z = means + stds * epsilon
            # add log_std as this is the jacobian for the change of variables of epsilon to z
            log_probs_z_given_x = self.independent_normal_log_prob(epsilon) - tf.math.reduce_sum(log_stds, axis=1)

            # Now IAF steps
            for i in range(self.n_autoregressive_units):
                #m, s = self.autoregressive_NNs[i]([z, h])
                m, s = self.autoregressive_NNs[i]([z, h*0])
                sigma = tf.nn.sigmoid(s)
                z = sigma * z + (1 - sigma) * m
                log_probs_z_given_x -= tf.reduce_sum(tf.math.log(sigma), axis=1)

            # prior probability (N(0,1))
            log_prob_z_prior = self.independent_normal_log_prob(z)

        variables = self.variables
        tape.gradient(log_probs_z_given_x, variables)



if __name__ == "__main__":
    # just do this file thing so we don't have to mess around with pycharm configs here
    from pathlib import Path; import os
    cwd_path = Path.cwd(); set_path = str(cwd_path.parent); os.chdir(set_path)

    # let's go
    tf.config.run_functions_eagerly(True)
    from Utils.load_plain_mnist import x_test
    latent_representation_dim = 32
    encoder = IAF_Encoder(latent_representation_dim=latent_representation_dim,
                          layer_nodes=100,
                          n_autoregressive_units=3,
                          autoregressive_unit_layer_width=100,
                          First_Encoder_to_IAF_step_dim=100)
    minitest = x_test[0:50, :, :]
    z, log_probs_z_given_x, log_prob_z_prior = encoder(minitest)
    #print(f"z, {z}")
    print(f"log_probs_z_given_x, {log_probs_z_given_x}")
    #print(f"log_prob_z_prior {log_prob_z_prior}")
    #encoder.debug_func(minitest)
