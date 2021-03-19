import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import pathlib, os

batch_size = 100
x_train, x_test = tfds.load("binarized_mnist", split=['train', 'test'], shuffle_files=False,
                                                    batch_size=-1)
x_train = x_train["image"].numpy(); x_test=x_test["image"].numpy()
x_train = x_train.astype("float32"); x_test = x_test.astype("float32")

if __name__ == "__main__":
    pathlib.Path(os.path.join(os.getcwd(), "Pytorch_VAE/Data/Binirised_MNIST/")).mkdir(parents=True, exist_ok=True)
    np.save("Pytorch_VAE/Data/Binirised_MNIST/x_train", x_train)
    np.save("Pytorch_VAE/Data/Binirised_MNIST/x_test", x_test)