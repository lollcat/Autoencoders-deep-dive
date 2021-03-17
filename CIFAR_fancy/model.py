import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from IAF_VAE_mnist.AutoregressiveNN.AutoregressiveNN import IAF_NN

class VAE_ladder(nn.Module):
    def __init__(self, in_dim):
        super(VAE_ladder, self).__init__()
