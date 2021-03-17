import torch
import torch.nn as nn
import torch.nn.functional as F
from IAF_VAE_mnist.AutoregressiveNN import AutoregressiveNN
from CIFAR_fancy.pixelcnn import PixelCNN


class LatentBlock(nn.Module):
    def __init__(self, in_channels=3, width=32, n_IAF_steps=1, IAF_node_width=32*32*3):
        super(LatentBlock, self).__init__()

        self.IAF_steps = nn.ModuleList([])
        for i in range(n_IAF_steps):
            self.IAF_steps.append(
                IAF_NN(latent_dim=width**2*3, h_dim=h_dim, IAF_node_width=IAF_node_width)
            )