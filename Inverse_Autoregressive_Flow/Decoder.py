from tensorflow.keras.layers import Dense, Layer, Reshape, Conv2DTranspose
import tensorflow as tf
import numpy as np

class Decoder(Layer):
    def __init__(self, x_dim, latent_representation_dim):
        super(Decoder, self).__init__()
        self.reshape = Reshape([1, 1, latent_representation_dim])
        # padding=valid means no padding # same keeps dimensions the same
        # currently mixing around to get shapes nice but not very principled
        # strides 2 doubles the first 2 dimensions
        assert x_dim[0]%4 == 0
        first_conv_height = first_conv_width = int(x_dim[0]/4)
        self.conv1 = Conv2DTranspose(filters=32, kernel_size=(first_conv_height, first_conv_width), strides=(1,1),
                                     padding='valid', activation=tf.nn.elu)
        self.conv2 = Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=(2,2), padding='same', activation=tf.nn.elu)
        self.conv3 = Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=(2,2), padding='same', activation=tf.nn.elu)
        self.conv4 = Conv2DTranspose(filters=1, kernel_size=(3, 3), strides=(1, 1), padding='same')


    def call(self, x):
        x = self.reshape(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x

if __name__ == "__main__":
    from pathlib import Path; import os
    cwd_path = Path.cwd(); set_path = str(cwd_path.parent); os.chdir(set_path)

    import tensorflow as tf
    from Utils.load_binarized_mnist import x_test, image_dim
    from Inverse_Autoregressive_Flow.Encoder import IAF_Encoder as Encoder

    latent_representation_dim = 3
    encoder = Encoder(latent_representation_dim, layer_nodes=64, n_autoregressive_units=3,
                 autoregressive_unit_layer_width=64, First_Encoder_to_IAF_step_dim=64)
    minitest = x_test[0:50, :, :]
    encoder_test = encoder(minitest)[0] # just grab z
    decoder = Decoder(image_dim, latent_representation_dim=latent_representation_dim)
    decoder_test = decoder(encoder_test)
    print(decoder_test.shape)