import torch
import torch.nn as nn
import torch.nn.functional as F
from IAF_VAE_mnist.Resnet import ResnetBlock

class Decoder(nn.Module):
    def __init__(self, latent_dim, fc_dim=100):
        super(Decoder, self).__init__()
        self.fc1 = torch.nn.utils.weight_norm(nn.Linear(latent_dim, fc_dim))
        self.fc2 = torch.nn.utils.weight_norm(nn.Linear(fc_dim, 4*4*32))
        self.deconv_resnet_blocks = nn.ModuleList([])
        self.deconv_resnet_blocks.append(ResnetBlock(input_filters=32, filter_size=32, stride=2, kernel_size=3, deconv=True))
        self.deconv_resnet_blocks.append(ResnetBlock(input_filters=32, filter_size=32, stride=1, kernel_size=3,
                                          deconv=True))
        self.deconv_resnet_blocks.append(ResnetBlock(input_filters=32, filter_size=16, stride=2, kernel_size=3,
                                          deconv=True, output_padding=1))
        self.deconv_resnet_blocks.append(ResnetBlock(input_filters=16, filter_size=16, stride=1, kernel_size=3,
                                          deconv=True))
        self.deconv_resnet_blocks.append(ResnetBlock(input_filters=16, filter_size=1, stride=2, kernel_size=3,
                                          deconv=True, output_padding=1))

    def forward(self, x):
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = torch.reshape(x, (-1, 32, 4, 4))
        for resnet_block in self.deconv_resnet_blocks:
            x = resnet_block(x)
        return x

if __name__ == "__main__":
    from Utils.load_binirised_mnist import load_data
    from Standard_VAE.Encoder import Encoder
    train_loader, test_loader = load_data(22)
    data = next(iter(train_loader))[0]
    encoder = Encoder(latent_dim=3, fc_layer_dim=10)
    decoder = Decoder(latent_dim=3)
    z, log_q_z_given_x, log_p_z = encoder(data)
    p_x_given_z = decoder(z)
    print(p_x_given_z.shape)
