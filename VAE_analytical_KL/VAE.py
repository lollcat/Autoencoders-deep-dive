from tensorflow.keras import Model
import tensorflow as tf
#from Encoder import Encoder
fr
from Decoder import Decoder
import numpy as np

class VAE(Model):
    def __init__(self, latent_representation_dim, x_dim, layer_nodes=64):
        super(VAE, self).__init__()
        self.input_size = np.prod(x_dim)
        self.encoder = Encoder(latent_representation_dim, layer_nodes)
        self.decoder = Decoder(x_dim, layer_nodes)


    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


    def get_encoding(self, x):
        return self.encoder(x)

    @tf.function
    def train_step(self, x_data):
        with tf.GradientTape() as tape:
            reconstruction = self(x_data)
            # multiplying by input size, as because binary cross entropy divides by this
            # but we need it at the original size to balance with the KL term
            reconstruction_loss = self.loss(x_data, reconstruction)*self.input_size
            kl_loss = self.losses[0]
            ELBO = reconstruction_loss + kl_loss
        variables = self.trainable_variables
        gradients = tape.gradient(ELBO, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        return ELBO, kl_loss, reconstruction_loss

    @tf.function
    def test_step(self, x_data):
        reconstruction = self(x_data)
        reconstruction_loss = self.loss(x_data, reconstruction)*self.input_size
        kl_loss = self.losses[0]
        ELBO = reconstruction_loss + kl_loss
        return ELBO, kl_loss, reconstruction_loss


if __name__ == "__main__":
    #tf.config.run_functions_eagerly(True)
    from load_data import x_test, image_dim
    minitest = x_test[0:50, :, :]

    latent_representation_dim = 32
    vae = VAE(latent_representation_dim, image_dim)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    loss_objective = tf.keras.losses.BinaryCrossentropy()
    train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    vae.compile(optimizer, loss_objective)

    vae_reconstruction_test = vae(minitest)
    print(vae_reconstruction_test.shape)
    ELBO = vae.train_step(minitest)
    print(ELBO)