import torch
import torch.nn as nn
import torch.nn.functional as F

from IAF_VAE_mnist.AutoregressiveNN.AutoregressiveNN import IAF_NN
from IAF_VAE_mnist.Resnet import ResnetBlock


class VAE_ladder_model(nn.Module):
    """
    We assume constant dim of mean and var at each layer to make things easy
    # top down (x -> z)
    # bottom up (z -> x)
    """
    def __init__(self, latent_dim, fc_layer_dim, n_IAF_steps, h_dim, IAF_node_width):
        super(VAE_ladder_model, self).__init__()
        # first let's define the TopDown part
        self.TopDown1_conv = torch.nn.utils.weight_norm(nn.Conv2d(in_channels=1, out_channels = 16,
                                                                  kernel_size=3, stride=1, padding=1))
        self.TopDown1_conv2 = torch.nn.utils.weight_norm(nn.Conv2d(in_channels=1, out_channels=16,
                                                                  kernel_size=3, stride=1, padding=1))

        self.TopDown1_fc_layer = nn.Linear(28 * 28 * 16, fc_layer_dim)
        self.TopDown1_mean_layer = nn.Linear(fc_layer_dim, latent_dim)  # to be combined with the final prior param
        self.TopDown1_log_std_layer = nn.Linear(fc_layer_dim, latent_dim)
        self.TopDown1_h_layer = nn.Linear(fc_layer_dim, h_dim)


        self.TopDown2_conv = self.conv = torch.nn.utils.weight_norm(nn.Conv2d(in_channels=1, out_channels = 16,
                                                                         kernel_size=3, stride=1, padding=1))
        self.TopDown2_fc_layer = nn.Linear(4 * 4 * 28, fc_layer_dim)
        self.TopDown2_mean_layer = nn.Linear(fc_layer_dim, latent_dim)  # to be combined with the final prior param
        self.TopDown2_log_std_layer = nn.Linear(fc_layer_dim, latent_dim)
        self.TopDown2_h_layer = nn.Linear(fc_layer_dim, h_dim)

        # Now the bottom up part
        self.BottomUp2_mean_layer = torch.nn.Parameter(torch.zeros(latent_dim))
        self.BottomUp2_log_std_layer = torch.nn.Parameter(torch.ones(latent_dim))
        self.BottomUp2_processing_layer = nn.Linear(latent_dim*2, latent_dim) # takes in






    def forward(self, x):
        TopDown1_conv = F.elu(self.TopDown1_conv(x))
        TopDown1_fc_layer = F.elu(self.TopDown1_fc_layer(torch.flatten(TopDown1_conv, 1)))
        TopDown1_mean_layer = self.TopDown1_mean_layer(TopDown1_fc_layer)
        TopDown1_log_std_layer = self.TopDown1_log_std_layer(TopDown1_fc_layer)
        TopDown1_h_layer = self.TopDown1_h_layer(TopDown1_fc_layer)


        x = TopDown1_conv + x
        TopDown2_conv = self.TopDown2_conv(x)
        TopDown2_fc_layer = self.TopDown2_fc_layer(torch.flatten(TopDown2_conv, 1))
        TopDown2_mean_layer = self.TopDown2_mean_layer(TopDown2_fc_layer)
        TopDown2_log_std_layer = self.TopDown1_log_std_layer(TopDown2_fc_layer)
        TopDown2_h_layer = self.TopDown2_h_layer(TopDown2_fc_layer)

        return x






if __name__ == '__main__':
    from Utils.load_binirised_mnist import load_data
    train_loader, test_loader = load_data(26)
    data = next(iter(train_loader))[0]
    model = VAE_ladder_model(latent_dim=10, fc_layer_dim=10, n_IAF_steps=1, h_dim=10, IAF_node_width=16)
    print(model(data).shape)
