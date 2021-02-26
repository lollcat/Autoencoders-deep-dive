import torch
import torch.nn as nn
import torch.nn.functional as F

class ResnetBlock(nn.Module):
    def __init__(self, input_filters, filter_size, stride, kernel_size=3, deconv=False):
        super(ResnetBlock, self).__init__()
        self.stride = stride
        if deconv is False:
            conv = nn.Conv2d
        else:
            conv = nn.ConvTranspose2d
        self.conv = torch.nn.utils.weight_norm(conv(input_filters, filter_size, kernel_size, stride,
                                                    padding=1))

        if stride == 2:
            self.identity_mapping = conv(input_filters, filter_size,
                                         kernel_size=1, stride=stride, padding=1)

    def forward(self, input):
        x = F.elu(self.conv(input))
        if self.stride == 2:
            x = x + self.identity_mapping(input)
        else:
            x = x + input
        return x

if __name__ == "__main__":
    from Utils.load_mnist_pytorch import load_data
    train_loader, test_loader = load_data(22)
    data = next(iter(train_loader))[0]
    my_nn = ResnetBlock(input_filters=32,filter_size=3, stride=2, kernel_size=3, deconv=False)
    result = my_nn(data)
    print(result)
