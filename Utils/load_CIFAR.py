import torch
import torchvision
import pathlib
import os
cwd_path = pathlib.Path.cwd()
assert "Utils" in os.listdir(str(cwd_path)) # check that we are calling this from the right place

data_path = str(cwd_path) + "/Data/" # note this is relying on this program being run from root
pathlib.Path(data_path).mkdir(parents=True, exist_ok=True)
transform = torch.nn.Sequential()

def load_data(batch_size = 64):
    train_loader = torch.utils.data.DataLoader(
      torchvision.datasets.CIFAR10(data_path, train=True, download=True,
                                 transform=torchvision.transforms.ToTensor()),
                                batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
      torchvision.datasets.CIFAR10(data_path, train=False, download=True,
                                 transform=torchvision.transforms.ToTensor()),
                                batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

if __name__ == "__main__":
    train_loader, test_loader = load_data(32)
    for x, y in train_loader:
        print("Data: ", x, y)