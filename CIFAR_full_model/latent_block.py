import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from CIFAR_full_model.pixelCNN import PixelCNN_Block


class LatentBlock(nn.Module):
    def __init__(self, with_IAF=True, n_hidden_layers=1):
        # this assumes CIFAR input shapes
        super(LatentBlock, self).__init__()
        self.epsilon_mean = nn.Parameter(torch.tensor(0.0), requires_grad=False) # need to do this to ensure epsilon on cuda
        self.epsilon_std = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.epsilon_sample_layer = torch.distributions.normal.Normal(self.epsilon_mean, self.epsilon_std)
        self.register_buffer("normalisation_constant", torch.tensor(np.log(2*np.pi)))

        self.with_IAF = with_IAF
        if with_IAF:
            self.PixelCNN_IAF = PixelCNN_Block(n_hidden_layers=n_hidden_layers)

    def forward(self, means, log_stds, h):
        log_stds = log_stds/10  # reparameterise to start of low
        stds = torch.exp(log_stds)
        epsilon = self.epsilon_sample_layer.rsample(means.shape)
        log_q_z_given_x = self.unit_MVG_Guassian_log_prob(epsilon)
        z = epsilon * stds + means
        log_q_z_given_x -= torch.sum(log_stds, dim=[1,2,3])
        if self.with_IAF:
            m, s = self.PixelCNN_IAF(z, h)
            sigma = torch.sigmoid(s)
            z = sigma * z + (1 - sigma) * m
            log_q_z_given_x = log_q_z_given_x - torch.sum(torch.log(sigma), dim=[1,2,3])
        return z, log_q_z_given_x


    def unit_MVG_Guassian_log_prob(self, sample):
        return -0.5 * torch.sum((sample ** 2 + self.normalisation_constant), dim=[1,2,3])

if __name__ == '__main__':
    from Utils.load_CIFAR import load_data
    train_loader, test_loader = load_data(100)
    data_chunk = next(iter(train_loader))[0][0:1, 0:1, :, :]
    latent_block = LatentBlock()
    means = data_chunk; log_stds = data_chunk-1; h = data_chunk + 3
    z, log_q_z_given_x = latent_block(means, log_stds, h)
    print(z.shape, log_q_z_given_x.shape)

