from Utils.running_mean import running_mean
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datetime import datetime
import pathlib, os
from tqdm import tqdm
from MNIST_ladder.model import VAE_ladder_model
from IAF_VAE_mnist.VAE import VAE


class VAE_ladder(VAE):
    def __init__(self, latent_dim=32, n_rungs=4, n_IAF_steps=1, IAF_node_width=450, use_GPU = True, name="",
                 constant_sigma=False):
        super(VAE_ladder, self).__init__()
        if use_GPU is True:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = "cpu"
        print(f"running using {self.device}")
        self.VAE_model = VAE_ladder_model(latent_dim=latent_dim, n_rungs=n_rungs, n_IAF_steps=n_IAF_steps,
                                          IAF_node_width=IAF_node_width, constant_sigma=constant_sigma).to(self.device)

        current_time = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
        self.save_NN_path = f"Results_and_trained_models/IAF_VAE_mnist/saved_models/{name}__latent_dim_{latent_dim}" \
                            f"__n_IAF_steps_{n_IAF_steps}__n_rungs_{n_rungs}__constant_sigma_{constant_sigma}__" \
                            f"IAF_node_width_{IAF_node_width}/{current_time}/"
        self.BCE_loss = torch.nn.BCEWithLogitsLoss(reduction="none")
        self.optimizer = torch.optim.Adamax(self.VAE_model.parameters(), lr=0.001)


if __name__ == "__main__":
    from Utils.load_binirised_mnist import load_data
    train_loader, test_loader = load_data(100)
    data = next(iter(train_loader))[0]
    print(data.shape)
    vae = VAE_ladder(latent_dim=3, n_rungs=2, n_IAF_steps=1, IAF_node_width=8)
    vae.train(EPOCHS = 1, train_loader=train_loader, save_model=False, test_loader=test_loader)