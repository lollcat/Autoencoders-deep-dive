from tensorflow.keras.layers import Dense, Layer, Reshape, Conv2DTranspose, BatchNormalization, Dropout
from Inverse_Autoregressive_Flow.resnet import resnet_block
import tensorflow as tf
import numpy as np

class Decoder(Layer):
    """
    Has symmetric structure to encoder
    """
    def __init__(self, x_dim, latent_representation_dim, layer_nodes=450):
        super(Decoder, self).__init__()
        self.fc_layer1 = Dense(layer_nodes, activation=tf.nn.elu)
        self.fc_layer_dropout = Dropout(0.5)
        self.fc_layer2 = Dense(512, activation=tf.nn.elu)   # equivalent to flatten part of encoder
        self.reshape = Reshape([4, 4, 32])  # reshape for input into resnets
        self.resnet_blocks = []
        self.resnet_blocks.append(resnet_block(filters=32, kernel_size=(3, 3), strides=(2, 2), type="transposed"))
        self.resnet_blocks.append(resnet_block(filters=32, kernel_size=(3, 3), strides=(1, 1), type = "transposed"))
        self.resnet_blocks.append(resnet_block(filters=16, kernel_size=(3, 3), strides=(2, 2), type = "transposed"))
        self.resnet_blocks.append(resnet_block(filters=16, kernel_size=(3, 3), strides=(1, 1), type = "transposed"))
        self.resnet_blocks.append(resnet_block(filters=1, kernel_size=(3, 3), strides=(2, 2), type = "transposed"))


    def call(self, x, training=False):
        x = self.fc_layer1(x)
        x = self.fc_layer_dropout(x)
        x = self.fc_layer2(x)
        x = self.reshape(x)
        for i, resblock in enumerate(self.resnet_blocks):
            x = resblock(x)
            if i == 0:
                x = x[:, 0:7, 0:7, :] # need shapes like this to end at (batch_size, 28, 28, 1) for mnist dimensions
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