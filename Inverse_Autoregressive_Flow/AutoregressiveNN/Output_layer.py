from tensorflow.keras.layers import Dense, Flatten, Layer, Reshape, Concatenate
import tensorflow as tf
import numpy as np


class Autoregressive_output_layer(Layer):
    """
    Note this class is designed for use with Input layers, where all nodes have the same dimension
    sigma and mu, as per Inverse Autoregressive Flow paper
    """
    def __init__(self, autoregressive_input_dim, previous_layer_units=64):
        super(Autoregressive_output_layer, self).__init__()
        weight_initialiser = tf.random_normal_initializer()
        self.mu_autoregressive_weights = \
            tf.Variable(initial_value=weight_initialiser(shape=(previous_layer_units, autoregressive_input_dim),
                        dtype="float32"), trainable=True)
        self.mu_biases = tf.Variable(
            initial_value=tf.zeros_initializer()(shape=(autoregressive_input_dim,), dtype="float32"),
            trainable=True)

        self.log_sigma_weights = \
            tf.Variable(initial_value=weight_initialiser(shape=(previous_layer_units, autoregressive_input_dim),
                        dtype="float32"), trainable=True)
        self.log_sigma_biases = tf.Variable(
            initial_value=tf.zeros_initializer()(shape=(autoregressive_input_dim,), dtype="float32"),
            trainable=True)

        self.autoregressive_weights_mask = np.zeros((previous_layer_units, autoregressive_input_dim))

        nodes_per_latent_representation_dim = previous_layer_units / autoregressive_input_dim
        for input_index in range(previous_layer_units):
            input_corresponding_max_autoregressive_input_index = input_index//nodes_per_latent_representation_dim
            for node_index in range(autoregressive_input_dim):
                node_corresponding_max_autoregressive_input_index = node_index
                if node_corresponding_max_autoregressive_input_index > input_corresponding_max_autoregressive_input_index:
                    self.autoregressive_weights_mask[input_index, node_index] = 1
        self.autoregressive_weights_mask = tf.convert_to_tensor(self.autoregressive_weights_mask, dtype="float32")


    def call(self, inputs):
        mu_s = tf.matmul(inputs, self.mu_autoregressive_weights * self.autoregressive_weights_mask)  \
            + self.mu_biases
        sigma_s = tf.math.exp(tf.matmul(inputs, self.log_sigma_weights * self.autoregressive_weights_mask)  \
            + self.log_sigma_biases)
        return mu_s, sigma_s

if __name__ == "__main__":
    from Input_layer import Autoregressive_input_layer
    from Middle_layer import Autoregressive_middle_layer
    # simple example with latent dim of 2

    latent_z = np.array([1, 100, 10000], dtype="float32")[np.newaxis, :]
    h = np.ones((3,), dtype="float32")[np.newaxis, :]

    input_layer = Autoregressive_input_layer(autoregressive_input_dim=3, non_autoregressive_input_dim=3, units=10)
    middle_layer = Autoregressive_middle_layer(autoregressive_input_dim=3, units=10)
    output_layer = Autoregressive_output_layer(autoregressive_input_dim=3, previous_layer_units=10)

    input_layer_result = input_layer([latent_z, h])
    middle_result = middle_layer(input_layer_result)
    mu_s, sigma_s = output_layer(middle_result)
    print(f"mu s are : {mu_s}")
    print(f"sigmas are: {sigma_s}")
