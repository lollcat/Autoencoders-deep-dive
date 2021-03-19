import torch
import torch.nn as nn
import torch.nn.functional as F

class ResnetBlock(nn.Module):
    def __init__(self, input_filters, filter_size, stride, kernel_size=3, deconv=False,
                 output_padding=0):
        super(ResnetBlock, self).__init__()
        self.stride = stride
        if deconv is False:
            self.conv = torch.nn.utils.weight_norm(nn.Conv2d(input_filters, filter_size,
                                                             kernel_size,
                                                             stride, padding=1))
            if stride == 2:
                self.identity_mapping = nn.Conv2d(input_filters, filter_size,
                                             kernel_size=1, stride=stride)
        else:
            self.conv = torch.nn.utils.weight_norm(nn.ConvTranspose2d(input_filters,
                                                                      filter_size, kernel_size, stride,
                                                                       padding=1,
                                                                      output_padding=output_padding))
            if stride == 2:
                self.identity_mapping = nn.ConvTranspose2d(input_filters, filter_size,
                                             kernel_size=1, stride=stride, output_padding=output_padding)


    def forward(self, input):
        x = F.elu(self.conv(input))
        if self.stride == 2:
            x = x + self.identity_mapping(input)
        else:
            x = x + input
        return x  # F.elu(x)

if __name__ == "__main__":
    from Utils.load_binirised_mnist import load_data
    train_loader, test_loader = load_data(22)
    data = next(iter(train_loader))[0]
    resnet_blocks = []
    resnet_blocks.append(ResnetBlock(input_filters=1,filter_size=16, stride=2, kernel_size=3, deconv=False))
    resnet_blocks.append(ResnetBlock(input_filters=16,filter_size=16, stride=1, kernel_size=3, deconv=False))
    resnet_blocks.append(ResnetBlock(input_filters=16,filter_size=32, stride=2, kernel_size=3, deconv=False))
    resnet_blocks.append(ResnetBlock(input_filters=32, filter_size=32, stride=1, kernel_size=3, deconv=False))
    resnet_blocks.append(ResnetBlock(input_filters=32, filter_size=32, stride=2, kernel_size=3, deconv=False))

    decoder_blocks = []
    decoder_blocks.append(ResnetBlock(input_filters=32, filter_size=32, stride=2, kernel_size=3, deconv=True))
    decoder_blocks.append(ResnetBlock(input_filters=32,filter_size=32, stride=1, kernel_size=3,
                                      deconv=True))
    decoder_blocks.append(ResnetBlock(input_filters=32, filter_size=16, stride=2, kernel_size=3,
                                      deconv=True, output_padding=1))
    decoder_blocks.append(ResnetBlock(input_filters=16, filter_size=16, stride=1, kernel_size=3,
                                      deconv=True))
    decoder_blocks.append(ResnetBlock(input_filters=16, filter_size=1, stride=2, kernel_size=3,
                                      deconv=True, output_padding=1))

    x = data
    for encoder_block in resnet_blocks:
        x = encoder_block(x)
        print(x.shape)
    print("now we decode back to the same shape")
    for decoder_block in decoder_blocks:
        x = decoder_block(x)
        print(x.shape)
