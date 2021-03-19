import torch
import torch.nn as nn
import torch.nn.functional as F
from IAF_VAE_mnist.AutoregressiveNN import AutoregressiveNN
from CIFAR_fancy.pixelcnn import PixelCNN


class LatentBlock(nn.Module):
    def __init__(self, in_channels=3, width=32, n_IAF_steps=1, IAF_node_width=32*32*3):
        super(LatentBlock, self).__init__()
