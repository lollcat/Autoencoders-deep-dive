import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from IAF_VAE_mnist.AutoregressiveNN.AutoregressiveNN import IAF_NN
from IAF_VAE_mnist.Decoder_old import Decoder
#from IAF_VAE_mnist.Decoder_new import Decoder


class VAE_ladder_model(nn.Module):
    """
    We assume constant dim of mean and var at each layer to make things easy
    # top down (x -> z)
    # bottom up (z -> x)
    latent_dim, n_IAF_steps, h_dim, IAF_node_width, encoder_fc_dim, decoder_fc_dim
    """
    def __init__(self, latent_dim, n_IAF_steps, h_dim, IAF_node_width, encoder_fc_dim, decoder_fc_dim=450):
        super(VAE_ladder_model, self).__init__()
        # first let's define the TopDown part
        Top_down_resnet_channels = 16
        self.TopDown1_conv = torch.nn.utils.weight_norm(nn.Conv2d(in_channels=1, out_channels = Top_down_resnet_channels,
                                                                  kernel_size=3, stride=1, padding=1))
        self.TopDown1_conv2 = torch.nn.utils.weight_norm(nn.Conv2d(in_channels=Top_down_resnet_channels, out_channels=16,
                                                                  kernel_size=3, stride=1, padding=1))

        self.TopDown1_fc_layer = nn.Linear(28 * 28 * Top_down_resnet_channels, encoder_fc_dim)
        self.TopDown1_mean_layer = nn.Linear(encoder_fc_dim, latent_dim)  # to be combined with the final prior param
        self.TopDown1_log_std_layer = nn.Linear(encoder_fc_dim, latent_dim)
        self.TopDown1_h_layer = nn.Linear(encoder_fc_dim, h_dim)


        self.TopDown2_conv = self.conv = torch.nn.utils.weight_norm(nn.Conv2d(in_channels=Top_down_resnet_channels ,
                                                                              out_channels = Top_down_resnet_channels ,
                                                                         kernel_size=3, stride=1, padding=1))
        self.TopDown2_fc_layer = nn.Linear(28 * 28 * Top_down_resnet_channels, encoder_fc_dim)
        self.TopDown2_mean_layer = nn.Linear(encoder_fc_dim, latent_dim)  # to be combined with the final prior param
        self.TopDown2_log_std_layer = nn.Linear(encoder_fc_dim, latent_dim)
        self.TopDown2_h_layer = nn.Linear(encoder_fc_dim, h_dim)

        # Now the bottom up part
        # the numbering corresponds to the same rungs of the ladder as topdown
        self.BottomUp2_mean_layer = torch.nn.Parameter(torch.zeros(latent_dim))
        self.BottomUp2_log_std_layer = torch.nn.Parameter(torch.ones(latent_dim))
        # takes in concat of prior mu, sigma and z from IAF
        self.BottomUp2_processing_layer = nn.Linear(latent_dim*3, latent_dim)

        self.BottomUp1_mean_layer = nn.Linear(latent_dim, latent_dim)
        self.BottomUp1_log_std_layer = nn.Linear(latent_dim, latent_dim)
        self.BottomUp1_processing_layer = nn.Linear(latent_dim * 3, latent_dim)

        # let's start with just one IAF step
        self.IAF_rung1 = IAF_NN(latent_dim=latent_dim, h_dim=h_dim, IAF_node_width=IAF_node_width)
        self.IAF_rung2 = IAF_NN(latent_dim=latent_dim, h_dim=h_dim, IAF_node_width=IAF_node_width)

        self.epsilon_mean = nn.Parameter(torch.tensor(0.0), requires_grad=False) # need to do this to ensure epsilon on cuda
        self.epsilon_std = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.epsilon_sample_layer = torch.distributions.normal.Normal(self.epsilon_mean, self.epsilon_std)

        self.decoder = Decoder(latent_dim=latent_dim, fc_dim=decoder_fc_dim)


    def forward(self, x):
        TopDown1_conv = F.elu(self.TopDown1_conv(x))
        TopDown1_fc_layer = F.elu(self.TopDown1_fc_layer(torch.flatten(TopDown1_conv, 1)))
        TopDown1_mean = self.TopDown1_mean_layer(TopDown1_fc_layer)
        TopDown1_log_std = self.TopDown1_log_std_layer(TopDown1_fc_layer)
        TopDown1_h = F.elu(self.TopDown1_h_layer(TopDown1_fc_layer))
        TopDown1_conv2 = F.elu(self.TopDown1_conv2(TopDown1_conv))

        x = TopDown1_conv2 + x
        TopDown2_conv = F.elu(self.TopDown2_conv(x))
        TopDown2_fc_layer = F.elu(self.TopDown2_fc_layer(torch.flatten(TopDown2_conv, 1)))
        TopDown2_mean = self.TopDown2_mean_layer(TopDown2_fc_layer)
        TopDown2_log_std= self.TopDown1_log_std_layer(TopDown2_fc_layer)
        TopDown2_h = F.elu(self.TopDown2_h_layer(TopDown2_fc_layer))

        # prior
        BottomUp2_mean = self.BottomUp2_mean_layer
        BottomUp2_log_std = self.BottomUp2_log_std_layer

        # now can merge ladder rung 2
        rung_2_mean = TopDown2_mean + BottomUp2_mean
        rung_2_log_std = TopDown2_log_std + BottomUp2_log_std
        rung_2_epsilon = self.epsilon_sample_layer.rsample(rung_2_mean.shape)
        rung_2_log_q_z_given_x = self.unit_MVG_Guassian_log_prob(rung_2_epsilon)
        run_2_z = rung_2_epsilon * torch.exp(rung_2_log_std)  + rung_2_mean
        rung_2_log_q_z_given_x -= torch.sum(rung_2_log_std, dim=1)
        # rung 2 IAF step
        rung_2_m, rung_2_s = self.IAF_rung2(run_2_z, TopDown2_h)
        rung_2_sigma = torch.sigmoid(rung_2_s)
        run_2_z = rung_2_sigma * run_2_z + (1 - rung_2_sigma) * rung_2_m
        rung_2_log_q_z_given_x = rung_2_log_q_z_given_x - torch.sum(torch.log(rung_2_sigma), dim=1)
        rung_2_log_p_z = self.unit_MVG_Guassian_log_prob(run_2_z, mu=BottomUp2_mean, sigma=BottomUp2_log_std)

        rung2_concat = self.BottomUp2_processing_layer(torch.cat((run_2_z, rung_2_mean, rung_2_log_std), 1))

        rung2_to_rung1_top_down = rung2_concat # there is no earlier part to connect (resnet) as 2 is highest
        BottomUp1_mean = self.BottomUp1_mean_layer(rung2_to_rung1_top_down)
        BottomUp1_log_std =  self.BottomUp1_log_std_layer(rung2_to_rung1_top_down)

        # now can merge ladder rung 2
        rung_1_mean = TopDown1_mean + BottomUp1_mean
        rung_1_log_std = TopDown1_log_std + BottomUp1_log_std
        rung_1_epsilon = self.epsilon_sample_layer.rsample(rung_1_mean.shape)
        rung_1_log_q_z_given_x = self.unit_MVG_Guassian_log_prob(rung_1_epsilon)
        run_1_z = rung_1_epsilon * torch.exp(rung_1_log_std)  + rung_1_mean
        rung_1_log_q_z_given_x  -= torch.sum(rung_1_log_std, dim=1)
        # rung 1 IAF step
        rung_1_m, rung_1_s = self.IAF_rung1(run_1_z, TopDown1_h)
        rung_1_sigma = torch.sigmoid(rung_1_s)
        run_1_z = rung_1_sigma * run_1_z + (1 - rung_1_sigma) * rung_1_m
        rung_1_log_q_z_given_x = rung_1_log_q_z_given_x - torch.sum(torch.log(rung_1_sigma), dim=1)
        rung_1_log_p_z = self.unit_MVG_Guassian_log_prob(run_1_z, mu=BottomUp1_mean, sigma=BottomUp1_log_std)

        rung1_concat = self.BottomUp1_processing_layer(torch.cat((run_1_z, rung_1_mean, rung_1_log_std), 1))

        rung1_out = rung2_to_rung1_top_down + rung1_concat

        reconstruction_logits = self.decoder(rung1_out)

        # compute overall posterior and prior vals
        log_q_z_given_x = rung_1_log_q_z_given_x + rung_2_log_q_z_given_x
        log_p_z = rung_1_log_p_z + rung_2_log_q_z_given_x
        return reconstruction_logits, log_q_z_given_x, log_p_z

    def unit_MVG_Guassian_log_prob(self, sample, mu=0, sigma=1):
        return -0.5*torch.sum((sample - mu)**2/sigma**2 + np.log(2*np.pi) + 2*np.log(sigma), dim=1)



if __name__ == '__main__':
    from Utils.load_binirised_mnist import load_data
    train_loader, test_loader = load_data(26)
    data = next(iter(train_loader))[0]
    model = VAE_ladder_model(latent_dim=10, encoder_fc_dim=10, n_IAF_steps=1, h_dim=10,
                             IAF_node_width=16, decoder_fc_dim=5)
    print(model(data)[0].shape, model(data)[1].shape, model(data)[2].shape)
