from Inverse_Autoregressive_Flow.AutoregressiveNN.output_layer_mask import Output_Layer_Mask
from tensorflow.keras.layers import Layer
from tensorflow_probability import layers as tfp_layers
WeightNormalization = tfp_layers.weight_norm.WeightNorm

class Autoregressive_output_layer(Layer):
    def __init__(self, latent_dim):
        super(Autoregressive_output_layer, self).__init__()
        self.s_layer = WeightNormalization(Output_Layer_Mask(n_latent_dim=latent_dim, units=latent_dim))
        self.m_layer = WeightNormalization(Output_Layer_Mask(n_latent_dim=latent_dim, units=latent_dim))

    def call(self, inputs):
        s = self.s_layer(inputs)
        m = self.m_layer(inputs)
        return m, s

if __name__ == "__main__":
    import numpy as np
    from Inverse_Autoregressive_Flow.AutoregressiveNN.Weight_norm_AutoregressiveNN.Input_layer_new import Autoregressive_input_layer
    from Inverse_Autoregressive_Flow.AutoregressiveNN.Weight_norm_AutoregressiveNN.middle_layer_new import \
        Middle_Layer_Mask
    import tensorflow as tf
    # simple example with latent dim of 2
    latent_dim = 3
    n_units = 12
    input_layer = Autoregressive_input_layer(latent_dim, units=n_units)
    lay2 = WeightNormalization(Middle_Layer_Mask(n_latent_dim=4, units=n_units), data_init=True)
    output_lay = Autoregressive_output_layer(latent_dim)

    latent_z = tf.convert_to_tensor(np.array([1, 1000, 1005], dtype="float32")[np.newaxis, :])
    h = np.ones((4,), dtype="float32")[np.newaxis, :]
    print(output_lay(lay2(input_layer([latent_z, h]))))

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(latent_z)
        lay1_output = input_layer([latent_z, h])
        mid= lay2(lay1_output)
        output = output_lay(mid)
        slice_last_m = tf.reduce_mean(output[0][:, -2:])
        slice_first_m = output[0][:, 0]

        slice_last_s = tf.reduce_mean(output[1][:, -2:])
        slice_first_s = output[1][:, 0]

    #print(tape.gradient(lay1_output, latent_z))
    print(tape.gradient(slice_last_m, latent_z))
    print(tape.gradient(slice_last_s, latent_z))
