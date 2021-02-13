import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
batch_size = 100
x_train, x_test = tfds.load("binarized_mnist", split=['train', 'test'], shuffle_files=True,
                            batch_size=-1)
x_train = x_train["image"].numpy(); x_test=x_test["image"].numpy()
x_train = x_train.astype("float32"); x_test = x_test.astype("float32")

image_dim = x_train.shape[1:]
train_ds = tf.data.Dataset.from_tensor_slices(x_train).shuffle(10000).batch(batch_size)
test_ds = tf.data.Dataset.from_tensor_slices(x_test).batch(batch_size)


if __name__ == "__main__":
    pass


