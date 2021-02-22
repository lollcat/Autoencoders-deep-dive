import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import initializers,activations, regularizers, constraints, layers
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import (Input, Dense, Lambda, Flatten, Reshape, BatchNormalization, Layer,
                          Activation, Dropout, Conv2D, Conv2DTranspose, Convolution2D, AveragePooling2D,
                          Concatenate, Add, Multiply, UpSampling2D)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
from tensorflow.keras import metrics
from tensorflow.keras import backend as K
from tensorflow.keras import backend as k
from tensorflow.keras.datasets import mnist
import scipy.stats as stats
from tensorflow.keras.layers import InputSpec


def mask_generator(n_dim_inp, h_dim, hidden_layers, random_input_order, seed):
    m = {}
    rng = np.random.RandomState(seed)
    if random_input_order == False:
        m[-1] = np.arange(n_dim_inp)
    else:
        m[-1] = rng.permutation(n_dim_inp)
    new_mat = np.ones([h_dim]) * -1
    m[-1] = np.array(np.concatenate((m[-1], new_mat)))
    for i in range(len(hidden_layers)):
        m[i] = rng.randint(m[i - 1].min(), n_dim_inp - 1, size=hidden_layers[i])
    masks = [tf.convert_to_tensor((m[l - 1][:, None] <= m[l][None, :]), dtype=np.float32) for l in
             range(len(hidden_layers))]
    L = len(hidden_layers)
    masks.append(tf.convert_to_tensor((m[L - 1][:, None] < m[-1][None, :]), dtype=np.float32))
    masks[-1] = tf.convert_to_tensor(masks[-1][:, 0:n_dim_inp], dtype=np.float32)
    # The last mask is the direct connection between the input and the output layer

    # for j in range(len(masks)):
    # masks[j].astype(np.float32)

    return masks


# The custom masking layer
# mask represents the mask associated with the current layer
# direct_mask represents the direct mask between the input and the output layer

class custom_masking(Layer):
    def __init__(self, units, mask,
                 # direct_mask = None,
                 random_input_order=False,
                 # is_output = False,
                 activation='relu',
                 use_bias=False,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(custom_masking, self).__init__(**kwargs)
        # Note that the input shape can also be inferred from the shape of prev_arr
        self.units = units
        self.mask = mask
        # self.direct_mask = direct_mask
        self.random_input_order = random_input_order
        # self.is_output = is_output
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(min_ndim=2)

    def build(self, input_shape):
        ## Write the code
        ## Does the direct connection between the inputs and outputs use a bias term?
        ## Same as keras.Dense
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        self.kernel = self.add_weight(shape=(input_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        # if self.is_output:
        # length = len(self.direct_mask)
        # self.direct_kernel = self.add_weight(shape = (length, length),
        # initializer = self.kernel_initializer,
        # name = 'direct_kernel',regularizer = self.kernel_regularizer,
        # constraint = self.kernel_constraint)

        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs):
        ## Modified keras.Dense to account for the mask
        pre_output = self.kernel * self.mask
        output = k.dot(inputs, pre_output)
        if self.use_bias:
            output = k.bias_add(output, self.bias, data_format='channels_last')
        if self.activation is not None:
            output = self.activation(output)
            return output

        # elif self.is_output:
        # pre_direct_output =

    def compute_output_shape(self, input_shape):
        ##Same as keras.Dense
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)


# Finally, we have the MADE object
class MADE(object):
    def __init__(self, z_in, hidden_sizes, random_input_order, h_in):
        self.z_in = z_in
        self.h_in = h_in
        self.hidden_sizes = hidden_sizes
        self.random_input_order = random_input_order
        seed1 = np.random.randint(0, 10)
        self.mask = mask_generator(self.z_in, self.h_in, self.hidden_sizes, self.random_input_order, seed=seed1)

    def build_model(self):
        a = Input(shape=(self.z_in + self.h_in,))
        x_layers = []
        # len(size_array)
        for i in range(len(self.hidden_sizes)):
            if i == 0:
                x_layers.append(custom_masking(self.hidden_sizes[i], mask=self.mask[i])(a))  # activation is relu
            else:
                x_layers.append(custom_masking(self.hidden_sizes[i], mask=self.mask[i])(x_layers[i - 1]))

        # Now write the output layer, output layer's activation is sigmoid.
        # The output mask is already present in self.mask, need the direct mask
        L = len(self.hidden_sizes)
        output_layer1 = custom_masking(self.z_in, mask=self.mask[-1], activation='linear')(x_layers[-1])
        output_layer2 = custom_masking(self.z_in, mask=self.mask[-1], activation='linear')(x_layers[-1])
        # direct_connect_mask =
        # direct_connect =
        # add the two layers
        # apply activation

        self.model = Model(inputs=a, outputs=[output_layer1, output_layer2])
        return self.model

    def summary(self):
        return self.model.summary()
