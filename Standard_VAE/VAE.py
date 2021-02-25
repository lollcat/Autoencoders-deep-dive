from Standard_VAE.Encoder import Encoder
from Standard_VAE.Decoder import Decoder
from Utils.running_mean import running_mean

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class VAE_model(nn.Module):
    def __init__(self, latent_dim, encoder_fc_dim=32, decoder_fc_dim=32):
        self.encoder = Encoder(latent_dim=latent_dim, fc_layer_dim=encoder_fc_dim)
        self.decoder = Decoder(latent_dim=latent_dim, fc_dim=decoder_fc_dim)

    def forward(self, x):
        z, log_q_z_given_x, log_p_z = self.encoder(x)
        reconstruction_logits = self.decoder(z)
        return reconstruction_logits, log_q_z_given_x, log_p_z

class VAE:
    def __init(self, latent_dim=32, encoder_fc_dim=32, decoder_fc_dim=32, use_gpu=False):
        use_gpu = False
        device = torch.device("cuda" if use_gpu else "cpu")
        self.VAE_model = VAE_model(latent_dim, encoder_fc_dim=encoder_fc_dim,
                                   decoder_fc_dim=decoder_fc_dim).to(device)
        self.BCE_loss = torch.nn.BCEWithLogitsLoss(reduction="none")
        self.optimizer = torch.optim.Adamax(self.VAE_model.parameters())



    def loss_function(self, reconstruction_logits, log_q_z_given_x, log_p_z, x_target):
        log_p_x_given_z = - torch.sum(self.BCE_loss(reconstruction_logits, x_target), dim=[1,2,3])
        log_p_x_given_z_per_batch = torch.mean(log_p_x_given_z)
        log_q_z_given_x_per_batch = torch.mean(log_q_z_given_x)
        log_p_z_per_batch = torch.mean(log_p_z)
        ELBO = log_p_x_given_z_per_batch + log_p_z_per_batch - log_q_z_given_x_per_batch
        loss = -ELBO
        return loss, log_p_x_given_z_per_batch, log_q_z_given_x_per_batch, log_p_z_per_batch

    def train(self, EPOCHS, train_loader, test_loader=None):
        for EPOCH in EPOCHS:
            running_loss = 0
            running_log_q_z_given_x = 0
            running_log_p_x_given_z = 0
            running_log_p_z = 0
            n_train_batches = len(train_loader)
            for i, (x, _) in enumerate(train_loader):
                self.optimizer.zero_grad()
                reconstruction_logits, log_q_z_given_x, log_p_z = self.VAE_model(x)
                loss, log_p_x_given_z_per_batch, log_q_z_given_x_per_batch, log_p_z_per_batch = \
                    self.loss_function(reconstruction_logits, log_q_z_given_x, log_p_z, x)
                loss.backward()
                self.optimizer.step()
                running_loss = running_mean(loss, running_loss, i)
                running_log_q_z_given_x = running_mean(log_q_z_given_x_per_batch, running_log_q_z_given_x, i)
                running_log_p_x_given_z = running_mean(log_p_x_given_z_per_batch, running_log_p_x_given_z, i)
                running_log_p_z = running_mean(log_p_z_per_batch, running_log_p_z, i)

                if i % (n_train_batches/10 + 1) == 0:
                    print(f"Epoch: {EPOCH + 1}"
                          f"running loss: {running_loss} "
                          f"running_log_q_z_given_x: {running_log_q_z_given_x}"
                          f"running_log_p_x_given_z: {running_log_p_x_given_z}"
                          f"running_log_p_z: {running_log_p_z}")



if __name__ == "__main__":
    from Utils.load_mnist_pytorch import load_data
    from Standard_VAE.Encoder import Encoder
    train_loader, test_loader = load_data(22)
    vae = VAE()
    print(
        vae.VAE_model(next(iter(train_loader))[0])
    )
    #vae.train(EPOCHS = 10, train_loader=train_loader)



