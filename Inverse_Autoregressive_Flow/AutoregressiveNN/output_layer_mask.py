import tensorflow as tf
import numpy as np
from tensorflow_probability import layers as tfp_layers
WeightNormalization = tfp_layers.weight_norm.WeightNorm
from tensorflow.keras.layers import InputSpec

class Output_Layer_Mask(tf.keras.layers.Dense):
    def __init__(self, n_latent_dim, units, *args, **kwargs):
        super(Output_Layer_Mask, self).__init__(units, *args, **kwargs)
        self.n_latent_dim = n_latent_dim

    def call(self, inputs):
        return tf.matmul(inputs, self.kernel*self.autoregressive_weights_mask) + self.bias

    def build(self, input_shape):
        dtype = tf.as_dtype(self.dtype or tf.keras.backend.floatx())
        if not (dtype.is_floating or dtype.is_complex):
            raise TypeError('Unable to build `Dense` layer with non-floating point '
                            'dtype %s' % (dtype,))

        input_shape = tf.TensorShape(input_shape)
        last_dim = tf.compat.dimension_value(input_shape[-1])
        if last_dim is None:
            raise ValueError('The last dimension of the inputs to `Dense` '
                             'should be defined. Found `None`.')
        self.input_spec = InputSpec(min_ndim=2, axes={-1: last_dim})
        self.kernel = self.add_weight(
            'kernel',
            shape=[last_dim, self.units],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=True)
        self.bias = self.add_weight(
            'bias',
            shape=[self.units, ],
            initializer=self.bias_initializer,
            regularizer=self.bias_regularizer,
            constraint=self.bias_constraint,
            dtype=self.dtype,
            trainable=True)
        units = self.units
        assert units == self.n_latent_dim
        autoregressive_weights_mask = np.zeros((last_dim, self.n_latent_dim))

        nodes_per_latent_representation_dim = last_dim/ self.n_latent_dim
        for input_index in range(last_dim):
            input_corresponding_max_autoregressive_input_index = input_index//nodes_per_latent_representation_dim
            for node_index in range(self.n_latent_dim):
                node_corresponding_max_autoregressive_input_index = node_index
                # see masking notes for why greater than or equal to here
                if node_corresponding_max_autoregressive_input_index >= input_corresponding_max_autoregressive_input_index:
                    autoregressive_weights_mask[input_index, node_index] = 1
        self.autoregressive_weights_mask = self.add_weight(
            'autoregressive_weights_mask',
            shape=autoregressive_weights_mask.shape,
            dtype=self.dtype,
            initializer=lambda x, dtype: tf.convert_to_tensor(autoregressive_weights_mask, dtype=dtype),
            trainable=False)
        self.built = True


if __name__ == "__main__":
    n_nodes = 32
    latent_z = tf.convert_to_tensor(np.array([[1, 1000, 1005, 25]], dtype="float32"))
    from Inverse_Autoregressive_Flow.AutoregressiveNN.First_Layer_Mask import First_Layer_Mask
    from Inverse_Autoregressive_Flow.AutoregressiveNN.middle_layer import Middle_Layer_Mask
    lay1 = WeightNormalization(First_Layer_Mask(n_nodes), data_init=True)
    lay2 = WeightNormalization(Middle_Layer_Mask(n_latent_dim=4, units=n_nodes), data_init=True)
    lay3 = WeightNormalization(Output_Layer_Mask(n_latent_dim=4, units=4))
    lay3(lay2(lay1(latent_z)))
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(latent_z)
        lay1_output = lay1(latent_z)
        output = lay2(lay1_output)
        output = lay3(output)
        slice_last = tf.reduce_mean(output[:, -4:])
        slice_first = output[:, 0]
    #print(tape.gradient(lay1_output, latent_z))
    print(tape.gradient(slice_last, latent_z))
    #print(tape.gradient(slice_first, latent_z))
