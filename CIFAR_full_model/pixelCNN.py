from torch import nn
import torch
import torch.nn.functional as F

class PixelCNN_MaskedConv(nn.Conv2d):
    # largely inspired by https://github.com/jzbontar/pixelcnn-pytorch/blob/master/main.py
    def __init__(self, type, *args, **kwargs):
        super(PixelCNN_MaskedConv, self).__init__(*args, **kwargs)
        assert type in ("A", "B")
        # where A is the first layer, so maks the centre pixel
        # and B are in following layers that don't mask the centre pixel
        self.register_buffer('mask', torch.ones_like(self.weight))  # copy shape of kernel from conv2D
        kernel_height = self.mask.shape[2]
        kernel_width = self.mask.shape[3]
        middle_row = kernel_height // 2
        middle_col = kernel_width // 2
        self.mask[:, :, middle_row+1:, :] = 0  # all rows after the middle one are 0
        if type == "A":
            # set middle row values to 0, including the current pixel
            self.mask[:, :, middle_row, middle_col:] = 0
        else:  # type = B
            # set middle row values to 0, not including the current pixel
            self.mask[:, :, middle_row, middle_col+1:] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super(PixelCNN_MaskedConv, self).forward(x)


class PixelCNN_Block(nn.Module):
    def __init__(self, n_hidden_layers=1):
        super(PixelCNN_Block, self).__init__()
        self.layer1 = PixelCNN_MaskedConv("A", in_channels=1, out_channels=1, kernel_size=3, stride=1,
                                                           padding=1)
        self.means = PixelCNN_MaskedConv("B", in_channels=1, out_channels=1, kernel_size=3, stride=1,
                                                           padding=1)
        self.sigma = PixelCNN_MaskedConv("B", in_channels=1, out_channels=1, kernel_size=3, stride=1,
                                                           padding=1)
        self.h_processing = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1,
                                                           padding=1)
        self.hidden_layers = torch.nn.ModuleList([])
        for i in range(n_hidden_layers):
            self.hidden_layers.append(PixelCNN_MaskedConv("B", in_channels=1, out_channels=1, kernel_size=3, stride=1,
                                                           padding=1))

    def forward(self, x, h):
        x = F.elu(self.layer1(x))
        for hidden_layer in self.hidden_layers:
            x = F.elu(hidden_layer(x))
        h = F.elu(self.h_processing(h))
        x = x + h
        means = self.means(x)
        sigma = self.sigma(x) / 10 + 1.5 # reparameterise to be near 1.5
        return means, sigma


if __name__ == '__main__':
    from Utils.load_CIFAR import load_data
    train_loader, test_loader = load_data(100)
    data_chunk = next(iter(train_loader))[0][0:1, 0:1, :, :]
    PixelCNN_masked_conv = PixelCNN_MaskedConv("B", in_channels=1, out_channels=1, kernel_size=3, stride=1,
                                                           padding=1)
    conv_block = PixelCNN_Block()
    out = conv_block(data_chunk, data_chunk)
    print(out[0].shape, out[1].shape)