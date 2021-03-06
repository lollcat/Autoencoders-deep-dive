import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self, latent_dim, fc_dim=100):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, fc_dim)
        self.fc2 = nn.Linear(fc_dim, 4*4*32)
        self.deconv1 = torch.nn.utils.weight_norm(nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, stride=2,
                                          padding=1))
        self.identity_mapping1 = torch.nn.utils.weight_norm(nn.ConvTranspose2d(in_channels=32, out_channels=32,
                                          kernel_size=1, stride=2))
        self.deconv2 = torch.nn.utils.weight_norm(
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, stride=1,
                               padding=1))
        self.deconv3 = torch.nn.utils.weight_norm(nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2,
                                          padding=1, output_padding=1))
        self.identity_mapping3 = torch.nn.utils.weight_norm(nn.ConvTranspose2d(in_channels=32, out_channels=16,
                                                                               kernel_size=1, stride=2, output_padding=1))

        self.deconv4 = torch.nn.utils.weight_norm(nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=3, stride=1,
                                          padding=1))

        self.deconv5 = torch.nn.utils.weight_norm((nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=3, stride=2,
                                          padding=1, output_padding=1)))
        self.identity_mapping5 = torch.nn.utils.weight_norm(nn.ConvTranspose2d(in_channels=16, out_channels=1,
                                                                               kernel_size=1, stride=2, output_padding=1))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.reshape(x, (-1, 32, 4, 4))
        x = self.deconv1(x) #+ self.identity_mapping1(x)
        x = self.deconv2(x) #+ x
        x = self.deconv3(x) #+ self.identity_mapping3(x)
        x = self.deconv4(x) #+ x
        x = self.deconv5(x) #+ self.identity_mapping5(x)
        return x

if __name__ == "__main__":
    from Utils.load_binirised_mnist import load_data
    from IAF_VAE_mnist.Encoder import Encoder
    train_loader, test_loader = load_data(22)
    data = next(iter(train_loader))[0]
    encoder = Encoder(latent_dim=2, fc_layer_dim=32, n_IAF_steps=2, h_dim=2, IAF_node_width=32)
    decoder = Decoder(latent_dim=2)
    z, log_q_z_given_x, log_p_z = encoder(data)
    p_x_given_z = decoder(z)
    print(p_x_given_z )