from tensorflow.keras.layers import Dense, Flatten, Layer, Reshape, Concatenate
import tensorflow as tf
import numpy as np

class Autoregressive_input_layer(Layer):
    """
    Autoregressive inputs refer to inputs were we need to maintain autoregressive structure as in MADE paper
    Specifically here the dimension of the autoregressive_input_dim is the latent representation dimension
    non_autoregressive inputs are fully connected (this represents h in IAF paper)

    """
    def __init__(self, autoregressive_input_dim, non_autoregressive_input_dim, units=64):
        super(Autoregressive_input_layer, self).__init__()
        assert units >= autoregressive_input_dim
        weight_initialiser = tf.random_normal_initializer()
        self.autoregressive_weights = \
            tf.Variable(initial_value=weight_initialiser(shape=(autoregressive_input_dim, units),
                        dtype="float32"), trainable=True)
        self.autoregressive_weights_mask = np.zeros((autoregressive_input_dim, units))
        # TODO can do this without a lengthy loop

        nodes_per_latent_representation_dim = units / autoregressive_input_dim
        for i in range(autoregressive_input_dim):
            for j in range(units):
                units_corresponding_max_autoregressive_input_index = j//nodes_per_latent_representation_dim
                if units_corresponding_max_autoregressive_input_index > i:
                    self.autoregressive_weights_mask[i, j] = 1
        self.autoregressive_weights_mask = tf.convert_to_tensor(self.autoregressive_weights_mask, dtype="float32")
        #self.L_mask = np.zeros((autoregressive_input_dim, units)).astype("float32")
        #self.L_mask = np.tril(self.L_mask, k=-1) # we can do this if units=autoregressive_input_dim

        self.non_autoregressive_weights = \
            tf.Variable(initial_value=weight_initialiser(shape=(non_autoregressive_input_dim, units),
                                                         dtype="float32"), trainable=True)
        self.biases = tf.Variable(initial_value=tf.zeros_initializer()(shape=(units, ), dtype="float32"),
                                  trainable=True)

    def call(self, inputs):
        z_autoregressive, h_non_autoregressive = inputs
        x = tf.matmul(z_autoregressive, self.autoregressive_weights*self.autoregressive_weights_mask) + \
            tf.matmul(h_non_autoregressive, self.non_autoregressive_weights) \
            + self.biases
        return tf.nn.relu(x)

if __name__ == "__main__":
    # simple example with latent dim of 2
    input_layer = Autoregressive_input_layer(2, 3, units=5)
    print(input_layer.autoregressive_weights_mask)
    latent_z = np.array([1, 1000], dtype="float32")[np.newaxis, :]
    h = np.ones((3,), dtype="float32")[np.newaxis, :]
    print(input_layer([latent_z, h]))