import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from IAF_VAE_mnist.Resnet import ResnetBlock


class Encoder(nn.Module):
    def __init__(self, latent_dim, fc_layer_dim, n_IAF_steps, h_dim, IAF_node_width, constant_sigma=False):
        super(Encoder, self).__init__()
        self.constant_sigma = constant_sigma
        if constant_sigma is False:
            from IAF_VAE_mnist.AutoregressiveNN.AutoregressiveNN import IAF_NN
        else:
            from IAF_VAE_mnist.AutoregressiveNN.AutoregressiveNN_constant_sigma import IAF_NN

        self.resnet_blocks = nn.ModuleList([])
        self.resnet_blocks.append(ResnetBlock(input_filters=1, filter_size=16, stride=2, kernel_size=3, deconv=False))
        self.resnet_blocks.append(ResnetBlock(input_filters=16, filter_size=16, stride=1, kernel_size=3, deconv=False))
        self.resnet_blocks.append(ResnetBlock(input_filters=16, filter_size=32, stride=2, kernel_size=3, deconv=False))
        self.resnet_blocks.append(ResnetBlock(input_filters=32, filter_size=32, stride=1, kernel_size=3, deconv=False))
        self.resnet_blocks.append(ResnetBlock(input_filters=32, filter_size=32, stride=2, kernel_size=3, deconv=False))

        self.fc_layer = torch.nn.utils.weight_norm(nn.Linear(4*4*32, fc_layer_dim))
        self.mean_layer = nn.Linear(fc_layer_dim, latent_dim)
        self.log_std_layer = nn.Linear(fc_layer_dim, latent_dim)

        self.epsilon_mean = nn.Parameter(torch.tensor(0.0), requires_grad=False) # need to do this to ensure epsilon on cuda
        self.epsilon_std = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.epsilon_sample_layer = torch.distributions.normal.Normal(self.epsilon_mean, self.epsilon_std)
        self.h_layer = nn.Linear(fc_layer_dim, h_dim)

        self.IAF_steps = nn.ModuleList([])
        for i in range(n_IAF_steps):
            self.IAF_steps.append(
                IAF_NN(latent_dim=latent_dim, h_dim=h_dim, IAF_node_width=IAF_node_width)
            )
        self.register_buffer("normalisation_constant", torch.tensor(np.log(2*np.pi)))

    def forward(self, x):
        for resnet_block in self.resnet_blocks:
            x = resnet_block(x)
        x = torch.flatten(x, 1)
        x = F.elu(self.fc_layer(x))
        means = self.mean_layer(x)
        log_stds = self.log_std_layer(x)/10  # reparameterise to start with low std deviation
        h = F.elu(self.h_layer(x))

        stds = torch.exp(log_stds)
        epsilon = self.epsilon_sample_layer.rsample(means.shape)
        log_q_z_given_x = self.unit_MVG_Guassian_log_prob(epsilon)
        z = epsilon * stds + means
        log_q_z_given_x -= torch.sum(log_stds, dim=1)

        for IAF in self.IAF_steps:
            m, s = IAF(z, h)
            if self.constant_sigma is False:
                sigma = torch.sigmoid(s)
                z = sigma * z + (1 - sigma) * m
                log_q_z_given_x = log_q_z_given_x - torch.sum(torch.log(sigma), dim=1)
            else:
                z = z + m  # sigma = 1, and we don't have to re-estimate log_q_z_given_x

        log_p_z = self.unit_MVG_Guassian_log_prob(z)
        return z, log_q_z_given_x, log_p_z

    def unit_MVG_Guassian_log_prob(self, sample):
        return -0.5*torch.sum((sample**2 + self.normalisation_constant), dim=1)


if __name__ == "__main__":
    from Utils.load_binirised_mnist import load_data

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = load_data(26)
    data = next(iter(train_loader))[0].to(device)
    my_nn = Encoder(latent_dim=3, fc_layer_dim=10, n_IAF_steps=10, h_dim=4, IAF_node_width=10).to(device)
    z, log_q_z_given_x, log_p_z = my_nn(data)
    print(z.shape)
