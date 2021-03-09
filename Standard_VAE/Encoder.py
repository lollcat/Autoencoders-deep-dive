import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Encoder(nn.Module):
    def __init__(self, latent_dim, fc_layer_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, padding=1)
        self.fc_layer = nn.Linear(28*28*64, fc_layer_dim)
        self.mean_layer = nn.Linear(fc_layer_dim, latent_dim)
        self.log_std_layer = nn.Linear(fc_layer_dim, latent_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.flatten(x, 1)
        x = F.elu(self.fc_layer(x))
        means = self.mean_layer(x)
        log_stds = self.log_std_layer(x)
        stds = torch.exp(log_stds)
        epsilon = torch.normal(0, 1, size=means.shape)
        log_q_z_given_x = self.unit_MVG_Guassian_log_prob(epsilon)
        z = epsilon * stds + means
        log_q_z_given_x -= torch.sum(log_stds, dim=1)

        log_p_z = self.unit_MVG_Guassian_log_prob(z)
        return z, log_q_z_given_x, log_p_z

    def unit_MVG_Guassian_log_prob(self, sample):
        return -0.5*torch.sum((sample**2 + np.log(2*np.pi)), dim=1)


if __name__ == "__main__":
    from Utils.load_mnist_pytorch import load_data

    train_loader, test_loader = load_data(26)
    data = next(iter(train_loader))[0]
    my_nn = Encoder(latent_dim=3, fc_layer_dim=10)
    print(my_nn)
    z, log_q_z_given_x, log_p_z = my_nn(data)
    print(z)