import torch
import torchvision
transform = torch.nn.Sequential()

def load_data(batch_size = 64):
    train_loader = torch.utils.data.DataLoader(
      torchvision.datasets.MNIST('/files/', train=True, download=True,
                                 transform=torchvision.transforms.ToTensor()),
                                batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
      torchvision.datasets.MNIST('/files/', train=False, download=True,
                                 transform=torchvision.transforms.ToTensor()),
                                batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

if __name__ == "__main__":
    train_loader, test_loader = load_data(32)
    for x, y in train_loader:
        print("Data: ", x, y)