import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from IAF_VAE_mnist.AutoregressiveNN.First_layer import FirstLayer
from IAF_VAE_mnist.AutoregressiveNN.Middle_layer import MiddleLayer
from IAF_VAE_mnist.AutoregressiveNN.Final_layer import FinalLayer
from IAF_VAE_mnist.AutoregressiveNN.skip_layer import SkipLayer
from CIFAR_fancy.pixelcnn import PixelCNN


class IAF_NN(nn.Module):
    def __init__(self, in_channels=3, latent_dim=32*32*3, h_dim=32*32*3, IAF_node_width=32*32*3):
        super(IAF_NN, self).__init__()
        self.output_shape = [3, 32, 32]
        self.PixelCNN = PixelCNN("A", in_channels=in_channels, out_channels=in_channels,
                                 kernel_size=3, stride=1, padding=1)
        self.FirstLayer = FirstLayer(latent_dim, h_dim, layer_width=IAF_node_width)
        #self.MiddleLayer = MiddleLayer(latent_dim=latent_dim, layer_width=IAF_node_width)
        self.FinalLayer = FinalLayer(latent_dim=latent_dim, layer_width=IAF_node_width)
        self.m_SkipLayer = SkipLayer(latent_dim=latent_dim)
        self.s_SkipLayer = SkipLayer(latent_dim=latent_dim)


    def forward(self, z, h):
        batch_size = z.shape[0]
        z = self.PixelCNN(z)
        x_flat = z.flatten(start_dim=1, end_dim=-1)
        h = torch.flatten(h, start_dim=1, end_dim=-1)
        x = self.FirstLayer(x_flat, h)
        #x = self.MiddleLayer(x)
        m, s = self.FinalLayer(x)
        m = m + self.m_SkipLayer(x_flat)
        s = s + self.s_SkipLayer(x_flat)
        s = s/10 + 1.5  # reparameterise to be about +1 to +2
        m = m.unflatten(1, self.output_shape)
        s = s.unflatten(1, self.output_shape)
        return m, s

    # TODO
    def forward(self, x):
        # for testing jacobian of pixelCNN
        return self.PixelCNN(x)


if __name__ == "__main__":
    import numpy as np
    from Utils.load_CIFAR import load_data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = load_data(26)
    data = next(iter(train_loader))[0].to(device)
    z = torch.tensor(data, requires_grad=True)
    iaf_nn = IAF_NN()
    h = torch.ones(data.shape, requires_grad=True)
    m, s = iaf_nn(z, h)
    i = 10
    filter_n = 1    # if we set this to 0, then gradient asserts below work
    # we want to flatten filter_n last
    height = 5
    grad_h_all = torch.autograd.grad(torch.sum(m), h, retain_graph=True)
    grad_z_all = torch.autograd.grad(torch.sum(m), z, retain_graph=True)
    grad_z_some = torch.autograd.grad(torch.sum(m[0, filter_n, height, 0:i]), z, retain_graph=True)[0]

    assert torch.sum(grad_z_some[0, filter_n, height, i:]) == 0
    assert torch.sum(grad_z_some[0, filter_n, height, 0:i]) != 0
    assert torch.sum(grad_z_some[0, filter_n, height, 0:i]) != 0


    """
    print(m.shape, s.shape)
    n = 5
    z_slice = z[0, 0, 0, 0:n]
    m_slice = m[0, 0, 0, 0:n]
    
    jacobian = np.zeros((n, n))
    for i in range(m.shape[1]):
        jacobian[:, i] = torch.autograd.grad(m_slice[i], z_slice, only_inputs=True, retain_graph=True,
                                             allow_unused=True)[0].detach().numpy()
    print(jacobian)
    assert np.allclose(jacobian, np.triu(jacobian, 1))

    """
    """
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
    """