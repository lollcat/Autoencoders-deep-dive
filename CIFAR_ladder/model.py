import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from CIFAR_ladder.latent_block import latent_block
from CIFAR_ladder.updward_block import upblock

class VAE_ladder(nn.Module):
    def __init__(self, latent_dim=32, n_rungs=4, n_IAF_steps=1):
        super(VAE_ladder, self).__init__()
        self.n_rungs = n_rungs
        self.upward_blocks = nn.ModuleList([])
        self.latent_blocks = nn.ModuleList([])
        self.generative_block_conv1 = nn.ModuleList([])
        self.generative_block_mean_prior = nn.ModuleList([])
        self.generative_block_log_std_prior = nn.ModuleList([])
        self.generative_block_fc_before_concat = nn.ModuleList([]) # gives us features that we can conv
        self.generative_block_conv2 = nn.ModuleList([])
        for rung in range(n_rungs):
            self.upward_blocks.append(upblock(latent_dim=latent_dim))
            self.generative_block_conv1.append(
                torch.nn.utils.weight_norm(nn.Conv2d(in_channels=3, out_channels=3,
                                                             kernel_size=3, stride=1, padding=1))
            )
            self.generative_block_mean_prior.append(nn.Linear(32**2*3, latent_dim))
            self.generative_block_log_std_prior.append(nn.Linear(32**2*3, latent_dim))
            self.generative_block_fc_before_concat.append(nn.Linear(latent_dim, 32**2*3))
            # we get 3 channels from conv1 and 3 channels from fc processing of latent
            self.generative_block_conv2.append(torch.nn.utils.weight_norm(nn.Conv2d(in_channels=6, out_channels=3,
                                                             kernel_size=3, stride=1, padding=1)))

    def forward(self, x):
        # deterministic upwards
        up_variables = []
        for i in range(self.n_rungs):
            up_mean, up_log_std, h, to_next_rung = self.upward_blocks(x)
            up_variables.append((up_mean, up_log_std, h))
            x = to_next_rung

        generative_conv_starting = torch.ones(x.shape)
        for j in reversed(range(self.n_rungs)):
            generative_conv1 = F.elu(self.generative_block_conv1(generative_conv_starting))
            flat_conv1 = torch.flatten(generative_conv1, start_dim=1, end_dim=3)
            prior_mean = self.generative_block_mean_prior[j](flat_conv1)
            prior_log_std = self.generative_block_log_std_prior[j](flat_conv1)
            posterior_mean = prior_mean + up_variables[j][0]
            posterior_log_var = prior_log_std + up_variables[j][1]
            iaf_h = up_variables[j][2]
            z, log_q_z_given_x = self.latent_blocks[j](posterior_mean, posterior_log_var, iaf_h)
            prior_std = torch.exp(prior_log_std)
            log_p_z = self.unit_MVG_Guassian_log_prob(z, prior_mean, prior_std)
            process_stochastic_feat = self.generative_block_fc_before_concat[j](z)
            process_stochastic_feat = torch.reshape(process_stochastic_feat, (-1, 3, 32, 32))
            generative_concat_features = torch.cat([process_stochastic_feat, generative_conv1], dim=1)
            generative_conv2 = F.elu(self.generative_block_conv2[j](generative_concat_features))
            generative_conv_starting = generative_conv2 + generative_conv_starting  # now loop to next rung

            # TODO keep track of ELBO stuff and get output

    def unit_MVG_Guassian_log_prob(self, sample, mu=0, sigma=1):
        return -0.5*torch.sum((sample - mu)**2/sigma**2 + np.log(2*np.pi) + 2*np.log(sigma), dim=1)
