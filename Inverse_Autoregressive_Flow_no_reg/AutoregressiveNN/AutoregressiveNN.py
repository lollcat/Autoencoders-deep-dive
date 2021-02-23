from tensorflow.keras.layers import Dense, Flatten, Layer, Reshape, Concatenate
from tensorflow_addons.layers import WeightNormalization
import tensorflow as tf
import numpy as np

from Inverse_Autoregressive_Flow.AutoregressiveNN.Input_layer import Autoregressive_input_layer
from Inverse_Autoregressive_Flow.AutoregressiveNN.Middle_layer import Autoregressive_middle_layer
from Inverse_Autoregressive_Flow.AutoregressiveNN.Output_layer import Autoregressive_output_layer
from tensorflow_addons.layers import WeightNormalization

class AutoRegressiveNN_Unit(Layer):
    """
    # TODO should update these to remove redundant nodes
    We compose this of an input layer, middle layer(s), and output layer to manage all of the shapes cleanly
    Note it is assumed that input layer and middle layer have the same number of nodes
    We currently include dead nodes (corresponding to the first regressive input which has no dependency on
    other input elements in the middle layer - as it makes the code clearer
    See debugging powerpoint for checks on derivative values that confirm autoregressivness
    """
    def __init__(self, latent_representation_dim, h_dim, layer_nodes=8000, name="AutoRegressiveNN"):
        super(AutoRegressiveNN_Unit, self).__init__()
        self.input_layer = Autoregressive_input_layer(autoregressive_input_dim=latent_representation_dim,
                                                      non_autoregressive_input_dim=h_dim,
                                                      units=layer_nodes)
        self.middle_layer = Autoregressive_middle_layer(autoregressive_input_dim=latent_representation_dim,
                                                        units=layer_nodes)
        self.output_layer = Autoregressive_output_layer(autoregressive_input_dim=latent_representation_dim,
                                                        previous_layer_units=layer_nodes)

        self.m_skip_weights = \
            tf.Variable(initial_value=tf.random_normal_initializer()(
                shape=(latent_representation_dim, latent_representation_dim),dtype="float32"),
                trainable=True)
        self.s_skip_weights = \
            tf.Variable(initial_value=tf.random_normal_initializer()(
                shape=(latent_representation_dim, latent_representation_dim), dtype="float32"),
                trainable=True)
        self.skip_mask = np.ones((latent_representation_dim, latent_representation_dim))
        self.skip_mask = np.triu(self.skip_mask, k=1)


    def call(self, inputs, training=False):
        z, h = inputs
        x = self.input_layer([z, h])
        x = self.middle_layer(x)
        m, s = self.output_layer(x)
        m_skip = tf.matmul(z, self.m_skip_weights*self.skip_mask)
        s_skip = tf.matmul(z, self.s_skip_weights*self.skip_mask)
        m = m + m_skip
        s = s + s_skip
        s = s/10 + 1.5  # parameterise to be initially round about +1 to +2
        return m, s

    def debug(self, inputs, training=True):
        # debug check to test autoregressiveness
        z, h = inputs
        z = tf.Variable(tf.convert_to_tensor(z, dtype="float32"), trainable=True)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(z)
            x = self.input_layer([z, h])
            #x = self.batch_norm1(x)
            x = self.middle_layer(x)
            #x = self.batch_norm2(x)
            m_preskip, s_preskip = self.output_layer(x)
            m_skip = tf.matmul(z, self.m_skip_weights*self.skip_mask)
            s_skip = tf.matmul(z, self.s_skip_weights*self.skip_mask)
            m = m_preskip + m_skip
            s = s_preskip + s_skip
            s = s/10 + 1.5  # parameterise to be initially round about +1 to +2

            slice_position = 2  # 0 for start, -1 for end
            m_preskip_slice = m_preskip[:, slice_position]
            m_slice = m[:, slice_position]

        gradients_m = tape.gradient(m_slice, z)
        gradients_preskip = tape.gradient(m_preskip_slice, z)
        assert tf.reduce_sum(gradients_m[slice_position:]) == 0

        return


if __name__ == "__main__":
    # just do this file thing so we don't have to mess around with pycharm configs here
    from pathlib import Path; import os
    cwd_path = Path.cwd(); set_path = str(cwd_path.parent.parent); os.chdir(set_path)

    # let's go
    latent_z = np.array([1, 10, 50, 100000, 50, 70, 80, 53, 35, 632], dtype="float32")[np.newaxis, :]
    h = np.ones((3,), dtype="float32")[np.newaxis, :]

    Autoregressive_unit = AutoRegressiveNN_Unit(latent_representation_dim=latent_z.size, h_dim=h.size, layer_nodes=64)
    m, s = Autoregressive_unit([latent_z, h])
    print(f"mu s are : {m}")
    print(f"sigmas are: {s}")
    Autoregressive_unit.debug([latent_z, h])
