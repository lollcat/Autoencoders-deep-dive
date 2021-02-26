import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self, latent_dim, fc_dim=100):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, fc_dim)
        self.fc2 = nn.Linear(fc_dim, 4*4*32)
        self.deconv1 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, stride=2,
                                          padding=1)
        self.deconv2 = nn.ConvTranspose2d(in_channels=32, out_channels=64, kernel_size=3, stride=2,
                                          padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=3, stride=2,
                                          padding=1, output_padding=1)
        self.fc3 = nn.Linear(fc_dim, fc_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.reshape(x, (-1, 32, 4, 4))
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        return x

if __name__ == "__main__":
    from Utils.load_mnist_pytorch import load_data
    from Standard_VAE.Encoder import Encoder
    train_loader, test_loader = load_data(22)
    data = next(iter(train_loader))[0]
    encoder = Encoder(latent_dim=3, fc_layer_dim=10)
    decoder = Decoder(latent_dim=3)
    z, log_q_z_given_x, log_p_z = encoder(data)
    p_x_given_z = decoder(z)
