from tensorflow.keras.layers import Dense, Flatten, Conv2D, Layer
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

class Encoder(Layer):
    def __init__(self, latent_representation_dim, layer_nodes=64, name="encoder"):
        super(Encoder, self).__init__()
        #initializer = tf.keras.initializers.VarianceScaling(scale=0.0001, mode='fan_in',
        #                                                    distribution="truncated_normal")
        self.latent_representation_dim = latent_representation_dim
        distribution_type_layer = tfp.layers.IndependentNormal  #tfp.layers.MultivariateNormalTriL

        self.conv1 = Conv2D(32, 3, activation='relu')
        self.flatten = Flatten()
        self.layer1 = Dense(layer_nodes, activation="relu")
        self.layer2 = Dense(layer_nodes, activation="relu")
        self.distribution_spec = Dense(distribution_type_layer.params_size(latent_representation_dim),
                                       activation=None) #, kernel_initializer=initializer)
        self.sampler = distribution_type_layer(latent_representation_dim)
        prior = tfd.Independent(tfd.Normal(loc=tf.zeros(latent_representation_dim), scale=1),
                                        reinterpreted_batch_ndims=1)
        self.KL = tfp.layers.KLDivergenceRegularizer(prior, weight=1)
        self.variance = tf.keras.metrics.Mean(name='variance')


    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.flatten(x)
        x = self.layer1(x)
        x = self.layer2(x)
        distribution_spec = self.distribution_spec(x)
        z = self.sampler(distribution_spec)
        # make invariant to batch size
        KL_per_batch = self.KL(z)/z.batch_shape[0]
        self.add_loss(KL_per_batch)  # add KL divergence between prior and posterior distribution
        self.variance.update_state(z.variance())
        return z

if __name__ == "__main__":
    tf.config.run_functions_eagerly(True)
    from load_data import x_test
    latent_representation_dim = 32
    encoder = Encoder(latent_representation_dim)
    minitest = x_test[0:50, :, :]
    encoder_test = encoder(minitest)
    print(encoder_test.shape, encoder.losses)

