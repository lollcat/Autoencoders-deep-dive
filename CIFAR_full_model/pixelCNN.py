from torch import nn
import torch
import torch.nn.functional as F

class MaskedConv2d(nn.Conv2d):
    # https://github.com/jzbontar/pixelcnn-pytorch/blob/master/main.py
    def __init__(self, mask_type, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        assert mask_type in {'A', 'B'}
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH // 2, kW // 2 + (mask_type == 'B'):] = 0
        self.mask[:, :, kH // 2 + 1:] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)


class PixelCNN_Block(nn.Module):
    def __init__(self, n_hidden_layers=1):
        super(PixelCNN_Block, self).__init__()
        self.layer1 = MaskedConv2d("A", in_channels=1, out_channels=1, kernel_size=3, stride=1,
                                                           padding=1)
        self.means = MaskedConv2d("B", in_channels=1, out_channels=1, kernel_size=3, stride=1,
                                                           padding=1)
        self.sigma = MaskedConv2d("B", in_channels=1, out_channels=1, kernel_size=3, stride=1,
                                                           padding=1)
        self.h_processing = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1,
                                                           padding=1)
        self.hidden_layers = torch.nn.ModuleList([])
        for i in range(n_hidden_layers):
            self.hidden_layers.append(MaskedConv2d("B", in_channels=1, out_channels=1, kernel_size=3, stride=1,
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
    conv_block = PixelCNN_Block()
    out = conv_block(data_chunk, data_chunk)
    print(out[0].shape, out[1].shape)