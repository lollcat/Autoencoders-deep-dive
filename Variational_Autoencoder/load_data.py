import tensorflow as tf

mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
# Add a channels dimension
x_train = x_train[..., tf.newaxis].astype("float32")
x_test = x_test[..., tf.newaxis].astype("float32")
y_test = y_test[..., tf.newaxis].astype("float32")
y_train = y_train[..., tf.newaxis].astype("float32")

image_dim = x_train.shape[1:]
train_ds = tf.data.Dataset.from_tensor_slices(x_train).shuffle(10000).batch(256)
test_ds = tf.data.Dataset.from_tensor_slices(x_test).batch(256)
