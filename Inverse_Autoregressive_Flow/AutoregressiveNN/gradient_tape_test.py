if __name__ == "__main__":
    import tensorflow as tf
    import numpy as np
    from tensorflow.keras.layers import Dense
    layers = [Dense(2), Dense(2), Dense(2)]
    x = np.array([1,2])[np.newaxis, :]
    x_list = []
    y_list = []
    with tf.GradientTape() as g:
        for i in range(3):
            y = layers[i](x)
            x = y + 2
    dy_dx = g.gradient(x, layers[0].trainable_variables)
    """
        #y_list.append(y)
        #z = y_list[0]*4
        z = y*3
        # breaker
            #y = 3
            ##z = y*z
            #k = y*z

        # also a breaker
            #z = z + 2
    dy_dx = g.gradient(z, layer.trainable_variables)
"""
# can do assignment like this
x = np.zeros(3)
y = np.zeros(3)
x[0], y[1] = (2,3)
x, y