from tensorflow.keras.layers import Layer
import tensorflow as tf
import numpy as np
from tensorflow_probability import layers as tfp_layers
WeightNormalization = tfp_layers.weight_norm.WeightNorm

from Inverse_Autoregressive_Flow.AutoregressiveNN.attempt_at_norm_autoregressive_layer.Input_layer_new import \
    Autoregressive_input_layer
from Inverse_Autoregressive_Flow.AutoregressiveNN.attempt_at_norm_autoregressive_layer.middle_layer_new import \
    Middle_Layer_Mask
from Inverse_Autoregressive_Flow.AutoregressiveNN.attempt_at_norm_autoregressive_layer.output_layer_new import \
    Autoregressive_output_layer


class AutoRegressiveNN_Unit(Layer):
    def __init__(self, latent_representation_dim, h_dim, layer_nodes=8000, name="AutoRegressiveNN"):
        super(AutoRegressiveNN_Unit, self).__init__()
        self.input_layer = Autoregressive_input_layer(latent_representation_dim, units=layer_nodes)
        self.middle_layer = WeightNormalization(Middle_Layer_Mask(n_latent_dim=latent_representation_dim,
                                                                  units=layer_nodes), data_init=True)
        self.output_layer = Autoregressive_output_layer(latent_representation_dim)
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
            x = self.middle_layer(x)
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
    cwd_path = Path.cwd(); set_path = str(cwd_path.parent.parent.parent); os.chdir(set_path)

    # let's go
    latent_z = np.array([1, 10, 50, 100000, 50, 70, 80, 53, 35, 632], dtype="float32")[np.newaxis, :]
    h = np.ones((3,), dtype="float32")[np.newaxis, :]

    Autoregressive_unit = AutoRegressiveNN_Unit(latent_representation_dim=latent_z.size, h_dim=h.size, layer_nodes=64)
    m, s = Autoregressive_unit([latent_z, h])
    print(f"mu s are : {m}")
    print(f"sigmas are: {s}")
    Autoregressive_unit.debug([latent_z, h])
