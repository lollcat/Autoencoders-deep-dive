from Utils.load_data import x_train, x_test, train_ds, test_ds, image_dim

from tensorflow.keras.layers import Dense, Flatten, Conv2D, Layer
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions


class Encoder(Layer):
    def __init__(self, latent_representation_dim=2):
        super(Encoder, self).__init__()
        self.conv1 = Conv2D(32, 3, activation='relu')
        self.flatten = Flatten()
        self.distribution_layer = Dense(tfp.layers.IndependentNormal.params_size(latent_representation_dim),
                                        tfp.layers.IndependentNormal(latent_representation_dim))

    def normal_output(self, inputs):
        x = self.conv1(inputs)
        x = self.flatten(x)
        dist = self.distribution_layer(x)
        return dist

if __name__ == "__main__":
    batch_size = 32
    minitest = x_test[0:batch_size, :, :]
    encoder = Encoder()
    encoder.normal_output(minitest).shape
