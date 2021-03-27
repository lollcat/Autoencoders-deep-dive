import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from CIFAR_ladder.model import VAE_ladder_model
from CIFAR_base_class.BaseClass import CIFAR_BASE


class VAE_ladder(CIFAR_BASE):
    # note this class inherits important functions from CIFAR_basic_IAF\VAE such as bits per dim
    def __init__(self, latent_dim=32, n_rungs=4, n_IAF_steps=1, IAF_node_width=450, use_GPU = True, name="",
                 constant_sigma=False, lambda_free_bits=0.25):
        super(VAE_ladder, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = VAE_ladder_model(latent_dim=latent_dim, n_rungs=n_rungs, n_IAF_steps=n_IAF_steps,
                                      IAF_node_width=IAF_node_width, constant_sigma=constant_sigma,
                                      lambda_free_bits=lambda_free_bits).to(self.device)
        self.optimizer = torch.optim.Adamax(self.model.parameters())
        current_time = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
        self.save_NN_path = f"Results_and_trained_models/CIFAR_ladder/{name}__latent_dim_{latent_dim}" \
                            f"__n_IAF_steps_{n_IAF_steps}__n_rungs_{n_rungs}__constant_sigma_{constant_sigma}__" \
                            f"IAF_node_width_{IAF_node_width}/{current_time}/"

if __name__ == '__main__':
    from Utils.load_CIFAR import load_data
    from Utils.mnist_plotting import plot_train_test
    from Utils.CIFAR_plotting import plot_original_and_reconstruction
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = load_data(100)
    x_data = next(iter(train_loader))[0].to(device)
    experiment_dict = {"latent_dim": 3, "n_IAF_steps": 1, "IAF_node_width": 10, "n_rungs": 4}
    test_model = VAE_ladder(**experiment_dict)
    print(test_model.get_bits_per_dim(test_loader, n_samples=1))
    train_history, test_history, bits_per_dim = test_model.train(2, train_loader, test_loader, save=False,
              lr_decay=True, validation_based_decay=True, early_stopping=True,
              early_stopping_criterion=40)
    fig_original, axs_original, fig_reconstruct, axs_reconstruct = \
        plot_original_and_reconstruction(test_model, test_loader)
    import matplotlib.pyplot as plt
    plt.show()
    figure, axs = plot_train_test(train_history, test_history)
