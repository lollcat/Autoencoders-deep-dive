import torch
import torch.nn as nn
import torch.nn.functional as F


class UpwardBlock(nn.Module):
    def __init__(self, in_channels=3):
        super(UpwardBlock, self).__init__()
        self.up_means = torch.nn.utils.weight_norm(nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                                                        kernel_size=3, stride=1, padding=1))
        self.up_vars = torch.nn.utils.weight_norm(nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                                                        kernel_size=3, stride=1, padding=1))
        self.up_h= torch.nn.utils.weight_norm(nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                                                        kernel_size=3, stride=1, padding=1))
        self.up_conv2 = torch.nn.utils.weight_norm(nn.Conv2d(in_channels=in_channels*3, out_channels=in_channels,
                                                             kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        up_means = F.elu(self.up_means(x))
        up_vars = F.softplus(self.up_vars(x))
        h_vars = F.elu(self.up_h(x))
        deterministic_features = torch.cat([up_means, up_vars, h_vars], dim=1)
        to_next_rung = F.elu(self.up_conv2(deterministic_features))
        return up_means, up_vars, to_next_rung

if __name__ == '__main__':
    from Utils.load_CIFAR import load_data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = load_data(26)
    data = next(iter(train_loader))[0].to(device)
    upblock = UpwardBlock()
    up_means, up_vars, to_next_rung = upblock(data)
    print(up_means.shape, up_vars.shape, to_next_rung.shape)
