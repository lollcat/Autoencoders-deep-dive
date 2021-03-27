import torch
import torch.nn as nn
import torch.nn.functional as F
from IAF_VAE_mnist.Resnet import ResnetBlock

class x_to_z_dim_Block(nn.Module):
    def __init__(self, n_downsample = 2):
        super(x_to_z_dim_Block, self).__init__()
        self.resnet_blocks = nn.ModuleList([])
        # first block takes in 3 filters and outputs 1
        self.resnet_blocks.append(ResnetBlock(input_filters=3, filter_size=1, stride=2, kernel_size=3, deconv=False))
        for i in range(n_downsample-1):
            self.resnet_blocks.append(ResnetBlock(input_filters=1, filter_size=1, stride=2, kernel_size=3, deconv=False))

    def forward(self, x):
        for resnet_block in self.resnet_blocks:
            x = resnet_block(x)
        return x

class z_to_x_dim_Block(nn.Module):
    def __init__(self, n_up_sample = 2):
        super(z_to_x_dim_Block, self).__init__()
        self.resnet_blocks = nn.ModuleList([])
        # first block takes in 3 filters and outputs 1
        self.resnet_blocks.append(ResnetBlock(input_filters=1, filter_size=3, stride=2, kernel_size=3, deconv=True,
                                              output_padding=1))
        for i in range(n_up_sample - 1):
            self.resnet_blocks.append(ResnetBlock(input_filters=3, filter_size=3, stride=2, kernel_size=3, deconv=True,
                                                  output_padding=1))

    def forward(self, x):
        for resnet_block in self.resnet_blocks:
            x = resnet_block(x)
        return x

if __name__ == '__main__':
    from Utils.load_CIFAR import load_data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = load_data(26)
    data = next(iter(train_loader))[0].to(device)
    to_z_block = x_to_z_dim_Block()
    to_x_block = z_to_x_dim_Block()
    z_dim_result = to_z_block(data)
    x_dim_result = to_x_block(z_dim_result)
    print(z_dim_result.shape)
    print(x_dim_result.shape)





