from CIFAR_basic_IAF.Encoder import Encoder
from CIFAR_basic_IAF.Decoder import Decoder
from Utils.running_mean import running_mean
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np


class VAE_model(nn.Module):
    def __init__(self, latent_dim, n_IAF_steps, h_dim, IAF_node_width=320, encoder_fc_dim=450, decoder_fc_dim=450):
        super(VAE_model, self).__init__()
        self.encoder = Encoder(latent_dim=latent_dim, fc_layer_dim=encoder_fc_dim, n_IAF_steps=n_IAF_steps,
                               h_dim=h_dim, IAF_node_width=IAF_node_width)
        self.decoder = Decoder(latent_dim=latent_dim, fc_dim=decoder_fc_dim)

    def forward(self, x):
        z, log_q_z_given_x, log_p_z = self.encoder(x)
        reconstruction_means, reconstruction_log_sigma = self.decoder(z)
        return reconstruction_means, reconstruction_log_sigma, log_q_z_given_x, log_p_z

class VAE:
    def __init__(self, latent_dim=2, n_IAF_steps=2, h_dim=32, IAF_node_width=32, encoder_fc_dim=32, decoder_fc_dim=32
                 , use_GPU = True):
        if use_GPU is True:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = "cpu"
        print(f"running using {self.device}")
        self.VAE_model = VAE_model(latent_dim, n_IAF_steps, h_dim, IAF_node_width, encoder_fc_dim, decoder_fc_dim)\
            .to(self.device)
        self.BCE_loss = torch.nn.BCEWithLogitsLoss(reduction="none")
        self.optimizer = torch.optim.Adamax(self.VAE_model.parameters())
        """
        base_distribution = torch.distributions.Uniform(0, 1)
        transforms = [torch.distributions.SigmoidTransform().inv, torch.distributions.AffineTransform(loc=a, scale=b)]
        logistic = torch.distributions.TransformedDistribution(base_distribution, transforms)
        """

    def get_reconstruction(self, x_data):
        # for visualisation
        return self.VAE_model(x_data)[0].cpu().detach().numpy()

    def get_latent_encoding(self, x_data):
        # for visualisation
        return self.VAE_model.encoder(x_data)[0].cpu().detach().numpy()


    def get_bits_per_dim(self, test_loader):
        num_pixels = 32*32
        ELBO = 0
        for i, (x, _) in enumerate(test_loader):
            x = x.to(self.device)
            reconstruction_mu, reconstruction_log_sigma, log_q_z_given_x, log_p_z = self.VAE_model(x)
            batch_ELBO = - self.loss_function(reconstruction_mu, reconstruction_log_sigma,
                                              log_q_z_given_x, log_p_z, x)[0].item()
            ELBO += (batch_ELBO - ELBO)/(i+1)
        return - ELBO/num_pixels/np.log(2)

    def discretized_log_lik(self, sample, mu, log_sigma, binsize=1/256.0):
        # see https://github.com/openai/iaf/blob/ad33fe4872bf6e4b4f387e709a625376bb8b0d9d/tf_utils/distributions.py#L28
        scale = torch.exp(log_sigma)
        sample = (torch.floor(sample / binsize) * binsize - mu) / scale
        log_p_x_given_z = torch.log(torch.sigmoid(sample + binsize / scale) - torch.sigmoid(sample) + 1e-7)
        return torch.sum(log_p_x_given_z, dim=[1, 2, 3])


    def loss_function(self, reconstruction_means, reconstruction_log_sigmas
                      , log_q_z_given_x, log_p_z, x_target):
        log_p_x_given_z = self.discretized_log_lik(x_target, reconstruction_means, reconstruction_log_sigmas)
        log_p_x_given_z_per_batch = torch.mean(log_p_x_given_z)
        log_q_z_given_x_per_batch = torch.mean(log_q_z_given_x)
        log_p_z_per_batch = torch.mean(log_p_z)
        ELBO = log_p_x_given_z_per_batch + log_p_z_per_batch - log_q_z_given_x_per_batch
        loss = -ELBO
        return loss, log_p_x_given_z_per_batch, log_q_z_given_x_per_batch, log_p_z_per_batch

    def train(self, EPOCHS, train_loader, test_loader=None):
        epoch_per_info = max(min(100, round(EPOCHS / 10)), 1)
        n_train_batches = len(train_loader)
        train_history = {"loss": [],
                         "log_p_x_given_z": [],
                         "log_q_z_given_x": [],
                         "log_p_z ": []}
        if test_loader is not None:
            test_history = {"loss": [],
                         "log_p_x_given_z": [],
                         "log_q_z_given_x": [],
                         "log_p_z ": []}
        else:
            test_history = None

        for EPOCH in range(EPOCHS):
            running_loss = 0
            running_log_q_z_given_x = 0
            running_log_p_x_given_z = 0
            running_log_p_z = 0

            test_running_loss = 0
            test_running_log_q_z_given_x = 0
            test_running_log_p_x_given_z = 0
            test_running_log_p_z = 0


            for i, (x, _) in enumerate(train_loader):
                x = x.to(self.device)
                #with torch.cuda.amp.autocast():
                reconstruction_means, reconstruction_log_vars, log_q_z_given_x, log_p_z = self.VAE_model(x)
                loss, log_p_x_given_z_per_batch, log_q_z_given_x_per_batch, log_p_z_per_batch = \
                    self.loss_function(reconstruction_means, reconstruction_log_vars, log_q_z_given_x, log_p_z, x)
                if torch.isnan(loss).item():
                    raise Exception("NAN loss encountered")
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                running_loss = running_mean(loss.item(), running_loss, i)
                running_log_q_z_given_x = running_mean(log_q_z_given_x_per_batch.item(), running_log_q_z_given_x, i)
                running_log_p_x_given_z = running_mean(log_p_x_given_z_per_batch.item(), running_log_p_x_given_z, i)
                running_log_p_z = running_mean(log_p_z_per_batch.item(), running_log_p_z, i)

            train_history["loss"].append(running_loss)
            train_history["log_p_x_given_z"].append(running_log_p_x_given_z)
            train_history["log_q_z_given_x"].append(running_log_q_z_given_x)
            train_history["log_p_z "].append(running_log_p_z)
            if EPOCH % epoch_per_info == 0:
                print(f"Epoch: {EPOCH + 1} \n"
                      f"running loss: {running_loss} \n"
                      f"running_log_p_x_given_z: {running_log_p_x_given_z} \n"
                      f"running_log_q_z_given_x: {running_log_q_z_given_x} \n"
                      f"running_log_p_z: {running_log_p_z} \n")


            for i, (x, _) in enumerate(test_loader):
                x = x.to(self.device)
                #with torch.cuda.amp.autocast():
                reconstruction_means, reconstruction_log_vars, log_q_z_given_x, log_p_z = self.VAE_model(x)
                loss, log_p_x_given_z_per_batch, log_q_z_given_x_per_batch, log_p_z_per_batch = \
                    self.loss_function(reconstruction_means, reconstruction_log_vars, log_q_z_given_x, log_p_z, x)
                test_running_loss = running_mean(loss.item(), test_running_loss, i)
                test_running_log_q_z_given_x = running_mean(log_q_z_given_x_per_batch.item(), test_running_log_q_z_given_x, i)
                test_running_log_p_x_given_z = running_mean(log_p_x_given_z_per_batch.item(), test_running_log_p_x_given_z, i)
                test_running_log_p_z = running_mean(log_p_z_per_batch.item(), test_running_log_p_z, i)

            test_history["loss"].append(test_running_loss)
            test_history["log_p_x_given_z"].append(test_running_log_p_x_given_z)
            test_history["log_q_z_given_x"].append(test_running_log_q_z_given_x)
            test_history["log_p_z "].append(test_running_log_p_z)

            if EPOCH % epoch_per_info == 0:
                print(f"Epoch: {EPOCH + 1} \n"
                      f"test running loss: {test_running_loss} \n"
                      f"test running_log_p_x_given_z: {test_running_log_p_x_given_z} \n"
                      f"test running_log_q_z_given_x: {test_running_log_q_z_given_x} \n"
                      f"test running_log_p_z: {test_running_log_p_z} \n")

        bits_per_dim = self.get_bits_per_dim(test_loader=test_loader)
        return train_history, test_history, bits_per_dim



if __name__ == "__main__":
    from Utils.load_CIFAR import load_data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = load_data(200)
    data_chunk = next(iter(train_loader))[0]
    vae = VAE()
    #reconstruction_means, reconstruction_log_vars, log_q_z_given_x, log_p_z = vae.VAE_model(data_chunk)
    #log_probs = vae.discretized_log_lik(data_chunk, reconstruction_means, reconstruction_log_vars)
    print(vae.get_bits_per_dim(test_loader=test_loader))
    vae.train(EPOCHS = 2, train_loader=train_loader, test_loader=test_loader)
    print(vae.get_bits_per_dim(test_loader=test_loader))

    n = 5
    data_chunk = next(iter(train_loader))[0][0:n**2, :, :, :]
    fig, axs = plt.subplots(n, n)
    for i in range(n * n):
        row = int(i / n)
        col = i % n
        axs[row, col].imshow(np.moveaxis(data_chunk[i, :, :, :].detach().numpy(), source=0, destination=-1), cmap="gray")
        axs[row, col].axis('off')
    plt.show()

    n = 5
    prediction = vae.VAE_model(data_chunk)[0].detach().numpy()
    fig, axs = plt.subplots(n, n)
    for i in range(n * n):
        row = int(i / n)
        col = i % n
        axs[row, col].imshow(np.moveaxis(prediction[i, :, :, :], source=0, destination=-1), cmap="gray")
        axs[row, col].axis('off')
    plt.show()










