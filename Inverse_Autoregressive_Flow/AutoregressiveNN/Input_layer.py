from tensorflow.keras.layers import Dense, Layer
import tensorflow as tf
import numpy as np
from Inverse_Autoregressive_Flow.AutoregressiveNN.First_Layer_Mask import First_Layer_Mask
from tensorflow_probability import layers as tfp_layers
WeightNormalization = tfp_layers.weight_norm.WeightNorm

class Autoregressive_input_layer(Layer):
    """
    Autoregressive inputs refer to inputs were we need to maintain autoregressive structure as in MADE paper
    Specifically here the dimension of the autoregressive_input_dim is the latent representation dimension
    non_autoregressive inputs are fully connected (this represents h in IAF paper)

    """
    def __init__(self, autoregressive_input_dim, units=64):
        super(Autoregressive_input_layer, self).__init__()
        assert units >= autoregressive_input_dim
        self.input_to_dense_masked = WeightNormalization(First_Layer_Mask(units))
        self.non_autoregressive_weights = WeightNormalization(Dense(units=units, activation=None, use_bias=False))
        self.biases = tf.Variable(initial_value=tf.zeros_initializer()(shape=(units, ), dtype="float32"),
                                  trainable=True)

    def call(self, inputs):
        z_autoregressive, h_non_autoregressive = inputs
        x = self.input_to_dense_masked(z_autoregressive) + \
            self.non_autoregressive_weights(h_non_autoregressive) \
            + self.biases
        return tf.nn.elu(x)

    def debug(self, inputs):
        z, h = inputs
        z = tf.Variable(tf.convert_to_tensor(z, dtype="float32"), trainable=True)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(z)
            output = self([z, h])
            slice_position = 2  # 0 for start, -1 for end
            m_preskip_slice = output[:, slice_position]



if __name__ == "__main__":
    # simple example with latent dim of 2
    input_layer = Autoregressive_input_layer(3, units=7)
    latent_z = np.array([1, 1000, 1005], dtype="float32")[np.newaxis, :]
    h = np.ones((4,), dtype="float32")[np.newaxis, :]
    print(input_layer([latent_z, h]))