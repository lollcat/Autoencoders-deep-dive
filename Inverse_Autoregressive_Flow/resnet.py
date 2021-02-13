from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization
import tensorflow as tf
import numpy as np


class resnet_block(Layer):
    def __init__(self, filters=16, kernel_size = (3,3), strides=(2,2)):
        super(resnet_block, self).__init__()
        # padding=same gives input and output same heigh/width
        self.conv = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
                           padding='same',
                           kernel_initializer='he_normal')
        self.batch_norm = BatchNormalization()
        self.identity_mapping = Conv2D(filters=filters, kernel_size=(1,1), strides=strides, padding="same")

    def call(self, input, training=False):
        x = self.conv(input)
        x = self.batch_norm(x)
        x = tf.nn.relu(x)
        x += self.identity_mapping(input)
        return x

if __name__ == "__main__":
    from Utils.load_binarized_mnist import x_test
    resnet = resnet_block()
    print(resnet(x_test[0:10, :, :]).shape)
