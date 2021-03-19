import torch
import torch.nn as nn

class PixelCNN(nn.Conv2d):
    """
    https://github.com/jzbontar/pixelcnn-pytorch
    """
    def __init__(self, mask_type, *args, **kwargs):
        super(PixelCNN, self).__init__(*args, **kwargs)
        assert mask_type in {'A', 'B'}
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH // 2, kW // 2 + (mask_type == 'B'):] = 0
        self.mask[:, :, kH // 2 + 1:] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super(PixelCNN, self).forward(x)


if __name__ == '__main__':
    from Utils.load_CIFAR import load_data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = load_data(26)
    data = next(iter(train_loader))[0].to(device)
    z = torch.tensor(data, requires_grad=True)
    masked_conv = PixelCNN("A", in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
    output = masked_conv(z)
    print(output.shape, z.shape)
    filter_n = 2
    grad_sum_out = torch.autograd.grad(torch.sum(output[0, filter_n, 0, 0:5]), z, retain_graph=True)[0]
    grad_sum_out[0, filter_n, 0, 0:5]
    pass