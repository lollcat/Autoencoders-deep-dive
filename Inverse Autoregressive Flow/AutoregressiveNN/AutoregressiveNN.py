from tensorflow.keras.layers import Dense, Flatten, Layer, Reshape, Concatenate
import tensorflow as tf
import numpy as np

from

class AutoRegressiveNN(Layer):
    """
    We compose this of an input layer, middle layer(s), and output layer to manage all of the shapes cleanly
    Note it is assumed that input layer and middle layer have the same number of nodes
    """
    def __init__(self, latent_representation_dim, h_dim, layer_nodes_per_latent, name="AutoRegressiveNN"):
        super(AutoRegressiveNN, self).__init__()
        concat = Concatenate()
        self.dense1 = Dense(layer_nodes)


    def call(self, inputs):
        z, h = inputs
