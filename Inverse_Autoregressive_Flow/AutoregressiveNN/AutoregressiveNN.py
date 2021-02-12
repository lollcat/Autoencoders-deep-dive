from tensorflow.keras.layers import Dense, Flatten, Layer, Reshape, Concatenate
import tensorflow as tf
import numpy as np

from Inverse_Autoregressive_Flow.AutoregressiveNN.Input_layer import Autoregressive_input_layer
from Inverse_Autoregressive_Flow.AutoregressiveNN.Middle_layer import Autoregressive_middle_layer
from Inverse_Autoregressive_Flow.AutoregressiveNN.Output_layer import Autoregressive_output_layer

class AutoRegressiveNN_Unit(Layer):
    """
    We compose this of an input layer, middle layer(s), and output layer to manage all of the shapes cleanly
    Note it is assumed that input layer and middle layer have the same number of nodes
    We currently include dead nodes (corresponding to the first regressive input which has no dependency on
    other input elements in the middle layer - as it makes the code clearer
    See debugging powerpoint for checks on derivative values that confirm autoregressivness
    """
    def __init__(self, latent_representation_dim, h_dim, layer_nodes_per_latent=64, name="AutoRegressiveNN"):
        super(AutoRegressiveNN_Unit, self).__init__()
        self.input_layer = Autoregressive_input_layer(autoregressive_input_dim=latent_representation_dim,
                                                      non_autoregressive_input_dim=h_dim,
                                                      units=layer_nodes_per_latent)
        self.middle_layer = Autoregressive_middle_layer(autoregressive_input_dim=latent_representation_dim,
                                                        units=layer_nodes_per_latent)
        self.output_layer = Autoregressive_output_layer(autoregressive_input_dim=latent_representation_dim,
                                                        previous_layer_units=layer_nodes_per_latent)
        self.m_skip_weights = \
            tf.Variable(initial_value=tf.random_normal_initializer()(
                shape=(latent_representation_dim, latent_representation_dim),dtype="float32"),
                trainable=True)
        self.s_skip_weights = \
            tf.Variable(initial_value=tf.random_normal_initializer()(
                shape=(latent_representation_dim, latent_representation_dim), dtype="float32"),
                trainable=True)
        self.skip_mask = np.ones((latent_representation_dim, latent_representation_dim))
        self.skip_mask = np.tril(self.skip_mask, k=-1)

    def call(self, inputs):
        z, h = inputs
        x = self.input_layer([z, h])
        x = self.middle_layer(x)
        m, s = self.output_layer(x)
        # TODO not sure if this is correct
        m = m + tf.matmul(z, self.m_skip_weights*self.skip_mask)
        s = s + tf.matmul(z, self.s_skip_weights*self.skip_mask)
        return m, s

if __name__ == "__main__":
    # just do this file thing so we don't have to mess around with pycharm configs here
    from pathlib import Path; import os
    cwd_path = Path.cwd(); set_path = str(cwd_path.parent.parent); os.chdir(set_path)

    # let's go
    latent_z = np.array([1, 10, 50, 100000], dtype="float32")[np.newaxis, :]
    h = np.ones((3,), dtype="float32")[np.newaxis, :]

    Autoregressive_unit = AutoRegressiveNN_Unit(latent_representation_dim=latent_z.size, h_dim=h.size, layer_nodes_per_latent=64)
    m, s = Autoregressive_unit([latent_z, h])
    print(f"mu s are : {m}")
    print(f"sigmas are: {s}")
