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
x_test = np.squeeze(x_test)[:, np.newaxis, :, :]

x_train_tensor = torch.Tensor(x_train)
x_test_tensor = torch.Tensor(x_test)

train_dataset = TensorDataset(x_train_tensor)
test_dataset = TensorDataset(x_test_tensor)

def load_data(batch_size = 64):
    train_loader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return train_loader, test_loader

if __name__ == "__main__":
    train_loader, test_loader = load_data(32)
    for (x,) in train_loader:
        pass