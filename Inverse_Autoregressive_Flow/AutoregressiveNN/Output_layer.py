from tensorflow.keras.layers import Dense, Flatten, Layer, Reshape, Concatenate
import tensorflow as tf
import numpy as np


class Autoregressive_output_layer(Layer):
    """
    Note this class is designed for use with Input layers, where all nodes have the same dimension
    s and m, as per Inverse Autoregressive Flow paper
    """
    def __init__(self, autoregressive_input_dim, previous_layer_units=64):
        super(Autoregressive_output_layer, self).__init__()
        weight_initialiser = tf.random_normal_initializer()
        self.m_autoregressive_weights = \
            tf.Variable(initial_value=weight_initialiser(shape=(previous_layer_units, autoregressive_input_dim),
                        dtype="float32"), trainable=True)
        self.m_biases = tf.Variable(
            initial_value=tf.zeros_initializer()(shape=(autoregressive_input_dim,), dtype="float32"),
            trainable=True)

        self.s_weights = \
            tf.Variable(initial_value=weight_initialiser(shape=(previous_layer_units, autoregressive_input_dim),
                        dtype="float32"), trainable=True)
        self.s_biases = tf.Variable(
            initial_value=tf.zeros_initializer()(shape=(autoregressive_input_dim,), dtype="float32"),
            trainable=True)

        self.autoregressive_weights_mask = np.zeros((previous_layer_units, autoregressive_input_dim))

        nodes_per_latent_representation_dim = previous_layer_units / autoregressive_input_dim
        for input_index in range(previous_layer_units):
            input_corresponding_max_autoregressive_input_index = input_index//nodes_per_latent_representation_dim
            for node_index in range(autoregressive_input_dim):
                node_corresponding_max_autoregressive_input_index = node_index
                # see masking notes for why greater than or equal to here
                if node_corresponding_max_autoregressive_input_index >= input_corresponding_max_autoregressive_input_index:
                    self.autoregressive_weights_mask[input_index, node_index] = 1
        self.autoregressive_weights_mask = tf.convert_to_tensor(self.autoregressive_weights_mask, dtype="float32")


    def call(self, inputs):
        m = tf.matmul(inputs, self.m_autoregressive_weights * self.autoregressive_weights_mask)  \
            + self.m_biases
        s = tf.matmul(inputs, self.s_weights * self.autoregressive_weights_mask)  \
            + self.s_biases  # paramaterise to make initialisation close to +2
        #return tf.keras.backend.reverse(m, axes=1), tf.keras.backend.reverse(s, axes=1)
        return m, s


if __name__ == "__main__":
    from Input_layer import Autoregressive_input_layer
    from Middle_layer import Autoregressive_middle_layer
    # simple example with latent dim of 2

    latent_z = tf.Variable(tf.convert_to_tensor(np.array([1, 100, 1000, 10000], dtype="float32")[np.newaxis, :])
                           , trainable=True)
    h = np.ones((3,), dtype="float32")[np.newaxis, :]

    input_layer = Autoregressive_input_layer(autoregressive_input_dim=latent_z.shape[-1], non_autoregressive_input_dim=3, units=10)
    middle_layer = Autoregressive_middle_layer(autoregressive_input_dim=latent_z.shape[-1], units=10)
    output_layer = Autoregressive_output_layer(autoregressive_input_dim=latent_z.shape[-1], previous_layer_units=10)

    # debug check to test autoregressiveness
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(latent_z)
        input_layer_result = input_layer([latent_z, h])
        middle_result = middle_layer(input_layer_result)
        m, s = output_layer(middle_result)
        slice_position = 0  # 0 for start, -1 for end

        m_slice = m[:, slice_position]
        input_layer_result_slice = input_layer_result[:, slice_position]
        middle_layer_result_slice = middle_result[:, slice_position]
    # m[0], s[0] should have no gradients w.r.t the inputs
    tape.gradient(input_layer_result_slice, latent_z) # this works as desire
    tape.gradient(middle_layer_result_slice, latent_z)
    tape.gradient(m_slice, latent_z)

    print(f"mu s are : {m}")
    print(f"sigmas are: {s}")