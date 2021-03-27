import torch
import torch.nn as nn
import torch.nn.functional as F


class UpwardBlock(nn.Module):
    def __init__(self, in_channels=3, strides=1):
        super(UpwardBlock, self).__init__()
        self.up_conv1 = torch.nn.utils.weight_norm(nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                                                             kernel_size=3, stride=strides, padding=1))
        self.up_means = torch.nn.utils.weight_norm(nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                                                             kernel_size=3, stride=strides, padding=1))
        self.up_log_stds = torch.nn.utils.weight_norm(nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                                                             kernel_size=3, stride=strides, padding=1))
        self.up_h = torch.nn.utils.weight_norm(nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                                             kernel_size=3, stride=1, padding=1)) # assume h has same dim as mean and log_std
        self.up_conv2 = torch.nn.utils.weight_norm(nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                                                             kernel_size=3, stride=strides, padding=1))

    def forward(self, x):
        up_conv1 = F.elu(self.up_conv1(x))
        up_means = self.up_means(up_conv1)
        up_log_stds = self.up_log_stds(up_conv1)
        h = F.elu(self.up_h(up_conv1))
        to_next_rung = F.elu(self.up_conv2(up_conv1)) + x
        return up_means, up_log_stds, h, to_next_rung

if __name__ == '__main__':
    from Utils.load_CIFAR import load_data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = load_data(26)
    data = next(iter(train_loader))[0].to(device)
    upblock = UpwardBlock()
    up_means, up_vars, h, to_next_rung = upblock(data)
    print(up_means.shape, up_vars.shape,h.shape, to_next_rung.shape)

