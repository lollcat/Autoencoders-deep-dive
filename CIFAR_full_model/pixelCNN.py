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

class PixelCNN_Layer(nn.Module):
    def __init__(self, type="A"):
        super(PixelCNN_Layer, self).__init__()
        self.r_conv = nn.utils.weight_norm(MaskedConv2d(type, in_channels=1, out_channels=1, kernel_size=3, stride=1,
                                                        padding=1))
        self.g_on_g_conv = nn.utils.weight_norm(MaskedConv2d(type, in_channels=1, out_channels=1, kernel_size=3,
                                                             stride=1, padding=1))
        self.g_on_r_conv = nn.utils.weight_norm(nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1,
                                                          padding=1))
        self.b_on_b_conv = nn.utils.weight_norm(MaskedConv2d(type, in_channels=1, out_channels=1, kernel_size=3,
                                                             stride=1, padding=1))
        self.b_on_rg_conv = nn.utils.weight_norm(nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, stride=1,
                                                           padding=1))

    def forward(self, x):
        # we do this weird slicing so we don't loose a dimension
        r_out = self.r_conv(x[:, 0:1, :, :])  # r only takes in r
        g_out = self.g_on_g_conv(x[:, 1:2, :, :]) + self.g_on_r_conv(x[:, 0:1, :, :])
        b_out = self.b_on_b_conv(x[:, 2:3, :, :]) + self.b_on_rg_conv(x[:, 0:2, :, :])
        out = torch.cat([r_out, g_out, b_out], dim=1)
        return out

class PixelCNN_Block(nn.Module):
    def __init__(self, n_hidden_layers=1):
        super(PixelCNN_Block, self).__init__()
        self.layer1 = PixelCNN_Layer("A")
        self.means = PixelCNN_Layer("B")
        self.sigma = PixelCNN_Layer("B")
        self.h_processing = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1,
                                                           padding=1)
        self.hidden_layers = torch.nn.ModuleList([])
        for i in range(n_hidden_layers):
            self.hidden_layers.append(PixelCNN_Layer("B"))

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
    data_chunk = next(iter(train_loader))[0]
    #conv = MaskedConv2d('A', in_channels=3, out_channels=1, kernel_size=3, stride=1, padding=1)
    #out = conv(data_chunk)
    #conv_block = PixelCNN_Layer()
    #out = conv_block(data_chunk)
    #print(out.shape)
    conv_block = PixelCNN_Block()
    out = conv_block(data_chunk, data_chunk)
    print(out[0].shape, out[1].shape)