import tensorflow as tf
import numpy as np
from tensorflow_probability import layers as tfp_layers
WeightNormalization = tfp_layers.weight_norm.WeightNorm
from tensorflow.keras.layers import InputSpec

class First_Layer_Mask(tf.keras.layers.Dense):
    def __init__(self, *args, **kwargs):
        super(First_Layer_Mask, self).__init__(*args, **kwargs)
        pass

    def call(self, inputs):
        return tf.matmul(inputs, self.kernel*self.autoregressive_weights_mask)

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
        self.bias = None
        autoregressive_input_dim = last_dim
        units = self.units
        autoregressive_weights_mask = np.zeros((autoregressive_input_dim, units))
        nodes_per_latent_representation_dim = units / autoregressive_input_dim
        for i in range(autoregressive_input_dim):
            for j in range(units):
                units_corresponding_max_autoregressive_input_index = j // nodes_per_latent_representation_dim
                if units_corresponding_max_autoregressive_input_index > i:
                    autoregressive_weights_mask[i, j] = 1
        self.autoregressive_weights_mask = self.add_weight(
            'autoregressive_weights_mask',
            shape=autoregressive_weights_mask.shape,
            dtype=self.dtype,
            initializer=lambda x, dtype: tf.convert_to_tensor(autoregressive_weights_mask, dtype=dtype),
            trainable=False)
        self.built = True


if __name__ == "__main__":
    latent_z = np.array([[1, 1000, 1005]], dtype="float32")[np.newaxis, :]
    lay = WeightNormalization(First_Layer_Mask(4))
    print(lay(latent_z))