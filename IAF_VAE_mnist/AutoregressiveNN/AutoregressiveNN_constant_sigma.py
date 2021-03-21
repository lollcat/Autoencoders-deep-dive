import torch
import torch.nn as nn
import torch.nn.functional as F
from IAF_VAE_mnist.AutoregressiveNN.First_layer import FirstLayer
from IAF_VAE_mnist.AutoregressiveNN.Middle_layer import MiddleLayer
from IAF_VAE_mnist.AutoregressiveNN.Final_layer_constant_s import FinalLayer
from IAF_VAE_mnist.AutoregressiveNN.skip_layer import SkipLayer


class IAF_NN(nn.Module):
    def __init__(self, latent_dim, h_dim, IAF_node_width):
        super(IAF_NN, self).__init__()
        self.FirstLayer = FirstLayer(latent_dim, h_dim, layer_width=IAF_node_width)
        self.MiddleLayer = MiddleLayer(latent_dim=latent_dim, layer_width=IAF_node_width)
        self.FinalLayer = FinalLayer(latent_dim=latent_dim, layer_width=IAF_node_width)
        self.m_SkipLayer = SkipLayer(latent_dim=latent_dim)

    def forward(self, z, h):
        x = self.FirstLayer(z, h)
        x = self.MiddleLayer(x)
        m, s = self.FinalLayer(x)
        m = m + self.m_SkipLayer(z)
        return m, s


if __name__ == "__main__":
    import numpy as np
    # do some checks to ensure autoregressive property
    z_test_tensor = torch.tensor([[1.5, 4.7, 5, 76]], requires_grad=True)
    h_test_tensor = torch.tensor([[1.5, 4.7, 5, 22]])
    autoNN = IAF_NN(z_test_tensor.shape[1], h_test_tensor.shape[1], 20)
    m, s = autoNN(z_test_tensor, h_test_tensor)
    gradient_w_r_t_first_element = torch.autograd.grad(m[:, 0], z_test_tensor, only_inputs=True, retain_graph=True)[0]\
        .detach().numpy()
    assert np.sum(gradient_w_r_t_first_element) == 0

    gradient_w_r_t_last_element = torch.autograd.grad(m[:, -1], z_test_tensor, only_inputs=True, retain_graph=True,)[0]\
        .detach().numpy()
    # final element must be dependent on all previous one (no zeros derivatives present)
    assert np.sum(gradient_w_r_t_last_element[:, 0:-1] == 0) == 0
    # final element of output not dependent of final element of input
    assert np.sum(gradient_w_r_t_last_element[:, -1] != 0) == 0

    jacobian = np.zeros((z_test_tensor.shape[1], m.shape[1]))
    for i in range(m.shape[1]):
        jacobian[:, i] = torch.autograd.grad(m[:, i], z_test_tensor, only_inputs=True, retain_graph=True)[0].detach().numpy()
    print(jacobian)
    assert np.allclose(jacobian, np.triu(jacobian, 1))