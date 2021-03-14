import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import pathlib
import os
cwd_path = pathlib.Path.cwd()
assert "Utils" in os.listdir(str(cwd_path)) # check that we are calling this from the right place


x_train = np.load("Data/Binirised_MNIST/x_train.npy")
x_test = np.load("Data/Binirised_MNIST/x_test.npy")
x_train = np.squeeze(x_train)[:, np.newaxis, :, :]
x_train_4_points = x_train[0:4, :, :, :]
x_train = np.repeat(x_train_4_points , 5000, axis=0)

x_train_tensor = torch.Tensor(x_train)

train_dataset = TensorDataset(x_train_tensor)

def load_data(batch_size = 64):
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    return train_loader

if __name__ == "__main__":
    train_loader, test_loader = load_data(32)
    for (x,) in train_loader:
        pass