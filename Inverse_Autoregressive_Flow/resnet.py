from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, Conv2DTranspose
from tensorflow.keras.regularizers import l2
import tensorflow as tf
import numpy as np


class resnet_block(Layer):
    def __init__(self, filters=16, kernel_size = (3,3), strides=(2,2), type="strided"):
        super(resnet_block, self).__init__()
        self.strides = strides
        if type == "strided":
            conv = Conv2D
        elif type == "transposed": # deconvolution
            conv = Conv2DTranspose
        # padding=same gives input and output same heigh/width
        self.conv = conv(filters=filters, kernel_size=kernel_size, strides=strides,
                           padding='same',
                           kernel_initializer='he_normal',
                         kernel_regularizer=l2(1e-4))
        if strides != (1,1):
            self.identity_mapping = conv(filters=filters, kernel_size=(1, 1), strides=strides, padding="same")

        self.batch_norm = BatchNormalization()

    def call(self, input, training=False):
        x = self.conv(input)
        x = self.batch_norm(x)
        x = tf.nn.elu(x)
        if self.strides != (1,1):
            x += self.identity_mapping(input)
        else:
            x += input
        return x

if __name__ == "__main__":
    from Utils.load_binarized_mnist import x_test
    resnet = resnet_block(strides=(1,1))
    print(resnet(x_test[0:10, :, :]).shape)
