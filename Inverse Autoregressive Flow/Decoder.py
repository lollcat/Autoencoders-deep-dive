from tensorflow.keras.layers import Dense, Layer
import tensorflow as tf
import numpy as np

class Decoder(Layer):
    def __init__(self, x_dim, layer_nodes=64):
        super(Decoder, self).__init__()
        self.d1 = Dense(layer_nodes, activation="relu")
        self.d2 = Dense(layer_nodes, activation="relu")
        self.output_layer = Dense(np.prod(x_dim), activation="linear")  # activation="sigmoid")
        self.output_shaper = tf.keras.layers.Reshape(x_dim)


    def call(self, x):
        x = self.d1(x)
        x = self.d2(x)
        x = self.output_layer(x)
        return self.output_shaper(x)

if __name__ == "__main__":
    import tensorflow as tf
    from load_data import x_test, image_dim
    from Encoder import Encoder

    latent_representation_dim = 32
    encoder = Encoder(latent_representation_dim)
    minitest = x_test[0:50, :, :]
    encoder_test = encoder(minitest)
    decoder = Decoder(image_dim)
    decoder_test = decoder(encoder_test)
    print(decoder_test.shape)