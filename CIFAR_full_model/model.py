import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from CIFAR_full_model.latent_block import LatentBlock
from CIFAR_full_model.updward_block import UpwardBlock

class VAE_ladder_model(nn.Module):
    """
    We have written Mnist version a bit more generally
    """
    def __init__(self, n_rungs=4, lambda_free_bits=0.25):
        super(VAE_ladder_model, self).__init__()
        self.lambda_free_bits = lambda_free_bits
        self.n_rungs = n_rungs
        self.upward_blocks = nn.ModuleList([])
        self.latent_blocks = nn.ModuleList([])
        self.generative_block_conv1 = nn.ModuleList([])
        self.generative_block_mean_prior = nn.ModuleList([])
        self.generative_block_log_std_prior = nn.ModuleList([])
        self.generative_block_conv2 = nn.ModuleList([])
        for rung in range(n_rungs):
            self.upward_blocks.append(UpwardBlock())
            self.latent_blocks.append(LatentBlock())
            self.generative_block_conv1.append(torch.nn.utils.weight_norm(nn.Conv2d(in_channels=3, out_channels=3,
                                                             kernel_size=3, stride=1, padding=1)))
            self.generative_block_mean_prior.append(torch.nn.utils.weight_norm(nn.Conv2d(in_channels=3, out_channels=3,
                                                             kernel_size=3, stride=1, padding=1)))
            self.generative_block_log_std_prior.append(torch.nn.utils.weight_norm(nn.Conv2d(in_channels=3, out_channels=3,
                                                             kernel_size=3, stride=1, padding=1)))
            # we get 3 channels from conv1 and 3 channels from fc processing of latent
            self.generative_block_conv2.append(torch.nn.utils.weight_norm(nn.Conv2d(in_channels=6, out_channels=3,
                                                             kernel_size=3, stride=1, padding=1)))
        self.reconstruction_mu = torch.nn.utils.weight_norm(nn.Conv2d(in_channels=3, out_channels=3,
                                                                      kernel_size=3, stride=1, padding=1))
        self.reconstruction_log_sigma = torch.nn.utils.weight_norm(nn.Conv2d(in_channels=3, out_channels=3,
                                                                             kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        # deterministic upwards
        up_variables = []
        for i in range(self.n_rungs):
            up_mean, up_log_std, h, to_next_rung = self.upward_blocks[i](x)
            up_variables.append((up_mean, up_log_std, h))
            x = to_next_rung

        KL_q_p = 0
        KL_free_bits_term = 0
        generative_conv_starting = torch.ones_like(x)
        for j in reversed(range(self.n_rungs)):
            generative_conv1 = F.elu(self.generative_block_conv1[j](generative_conv_starting))
            prior_mean = self.generative_block_mean_prior[j](generative_conv1)
            prior_log_std = self.generative_block_log_std_prior[j](generative_conv1)
            posterior_mean = prior_mean + up_variables[j][0]
            posterior_log_std = prior_log_std + up_variables[j][1]
            iaf_h = up_variables[j][2]
            z, log_q_z_given_x = self.latent_blocks[j](posterior_mean, posterior_log_std, iaf_h)
            prior_std = torch.exp(prior_log_std)
            log_p_z = self.unit_MVG_Guassian_log_prob(z, prior_mean, prior_std)
            generative_concat_features = torch.cat([z, generative_conv1], dim=1)
            generative_conv2 = F.elu(self.generative_block_conv2[j](generative_concat_features))
            generative_conv_starting = generative_conv2 + generative_conv_starting  # now loop to next rung
            rung_KL = log_q_z_given_x - log_p_z
            KL_q_p += rung_KL
            KL_free_bits_term -= torch.maximum(torch.mean(rung_KL),
                                                self.lambda_free_bits*torch.ones(1, device=rung_KL.device)*0)
        reconstruction_mu = torch.sigmoid(self.reconstruction_mu(generative_conv_starting))
        reconstruction_log_sigma = self.reconstruction_log_sigma(generative_conv_starting)

        # note KL is KL(q(z|x) | p(z) )
        return reconstruction_mu, reconstruction_log_sigma, KL_free_bits_term,  KL_q_p


    def unit_MVG_Guassian_log_prob(self, sample, mu=0, sigma=1):
        return -0.5*torch.sum((sample - mu)**2/sigma**2 + np.log(2*np.pi) + 2*torch.log(sigma), dim=[1,2,3])


if __name__ == '__main__':
    from Utils.load_CIFAR import load_data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = load_data(26)
    data = next(iter(train_loader))[0].to(device)
    test_model = VAE_ladder_model()
    reconstruction_mu, reconstruction_log_sigma, KL_free_bits_term,  KL_q_p = test_model(data)
    print(reconstruction_mu.shape, reconstruction_log_sigma.shape, KL_free_bits_term.shape, KL_q_p.shape)
