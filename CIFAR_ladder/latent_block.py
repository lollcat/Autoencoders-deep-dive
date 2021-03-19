import torch
import torch.nn as nn
import torch.nn.functional as F
from IAF_VAE_mnist.AutoregressiveNN.AutoregressiveNN import IAF_NN
import numpy as np


class LatentBlock(nn.Module):
    def __init__(self, latent_dim, n_IAF_steps=1, IAF_node_width=450):
        super(LatentBlock, self).__init__()
        self.epsilon_mean = nn.Parameter(torch.tensor(0.0), requires_grad=False) # need to do this to ensure epsilon on cuda
        self.epsilon_std = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.epsilon_sample_layer = torch.distributions.normal.Normal(self.epsilon_mean, self.epsilon_std)

        self.IAF_steps = nn.ModuleList([])
        for i in range(n_IAF_steps):
            # assume h dim same dim for latent dim for simplicity
            self.IAF_steps.append(
                IAF_NN(latent_dim=latent_dim, h_dim=latent_dim, IAF_node_width=IAF_node_width)
            )
        self.register_buffer("normalisation_constant", torch.tensor(np.log(2*np.pi)))

    def forward(self, means, log_stds, h):
        stds = torch.exp(log_stds)
        epsilon = self.epsilon_sample_layer.rsample(means.shape)
        log_q_z_given_x = self.unit_MVG_Guassian_log_prob(epsilon)
        z = epsilon * stds + means
        log_q_z_given_x -= torch.sum(log_stds, dim=1)

        for IAF in self.IAF_steps:
            m, s = IAF(z, h)
            sigma = torch.sigmoid(s)
            z = sigma * z + (1 - sigma) * m
            log_q_z_given_x = log_q_z_given_x - torch.sum(torch.log(sigma), dim=1)
        return z, log_q_z_given_x


    def unit_MVG_Guassian_log_prob(self, sample):
        return -0.5 * torch.sum((sample ** 2 + self.normalisation_constant), dim=1)

if __name__ == '__main__':
    batch_size = 2; latent_dim = 32;
    means = torch.ones((batch_size, latent_dim))
    log_stds = torch.ones((batch_size, latent_dim)) + 0.1
    h = torch.ones((batch_size, latent_dim)) + 0.2
    latent_block = LatentBlock(latent_dim=latent_dim)
    z, log_q_z_given_x = latent_block(means, log_stds, h)
    print(z.shape, log_q_z_given_x.shape)

