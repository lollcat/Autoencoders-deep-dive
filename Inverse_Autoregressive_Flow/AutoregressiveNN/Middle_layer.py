from tensorflow.keras.layers import Dense, Flatten, Layer, Reshape, Concatenate
import tensorflow as tf
import numpy as np


class Autoregressive_middle_layer(Layer):
    """
    Note this class is designed for use with Input layers, where all nodes have the same dimension
    """
    def __init__(self, autoregressive_input_dim, units=64):
        super(Autoregressive_middle_layer, self).__init__()
        assert units >= autoregressive_input_dim
        weight_initialiser = tf.random_normal_initializer()
        self.autoregressive_weights = \
            tf.Variable(initial_value=weight_initialiser(shape=(units, units),
                        dtype="float32"), trainable=True)
        self.autoregressive_weights_mask = np.zeros((units, units))

        nodes_per_latent_representation_dim = units/autoregressive_input_dim
        for input_index in range(units):
            input_corresponding_max_autoregressive_input_index = input_index//nodes_per_latent_representation_dim
            for node_index in range(units):
                node_corresponding_max_autoregressive_input_index = node_index//nodes_per_latent_representation_dim
                # see diagram in notes from why greater than or equal to (rather than just equal to) here
                if node_corresponding_max_autoregressive_input_index >= input_corresponding_max_autoregressive_input_index:
                    self.autoregressive_weights_mask[input_index, node_index] = 1
        self.autoregressive_weights_mask = tf.convert_to_tensor(self.autoregressive_weights_mask, dtype="float32")
        self.biases = tf.Variable(initial_value=tf.zeros_initializer()(shape=(units,), dtype="float32"),
                                  trainable=True)

    def call(self, inputs):
        x = tf.matmul(inputs, self.autoregressive_weights*self.autoregressive_weights_mask)  \
            + self.biases
        return tf.nn.leaky_relu(x)

if __name__ == "__main__":
    from Input_layer import Autoregressive_input_layer
    # simple example with latent dim of 2
    input_layer = Autoregressive_input_layer(2, 3, units=5)
    latent_z = np.array([1, 1000], dtype="float32")[np.newaxis, :]
    h = np.ones((3,), dtype="float32")[np.newaxis, :]
    input_layer_result = input_layer([latent_z, h])
    middle_layer = Autoregressive_middle_layer(2, units=5)
    middle_result = middle_layer(input_layer_result)
    print(middle_result)
