from IAF_VAE_mnist.Encoder import Encoder
#from IAF_VAE_mnist.Decoder import Decoder
from IAF_VAE_mnist.Decoder_new import Decoder
from Utils.running_mean import running_mean
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class VAE_model(nn.Module):
    def __init__(self, latent_dim, n_IAF_steps, h_dim, IAF_node_width=320, encoder_fc_dim=450, decoder_fc_dim=450):
        super(VAE_model, self).__init__()
        self.encoder = Encoder(latent_dim=latent_dim, fc_layer_dim=encoder_fc_dim, n_IAF_steps=n_IAF_steps,
                               h_dim=h_dim, IAF_node_width=IAF_node_width)
        self.decoder = Decoder(latent_dim=latent_dim, fc_dim=decoder_fc_dim)

    def forward(self, x):
        z, log_q_z_given_x, log_p_z = self.encoder(x)
        reconstruction_logits = self.decoder(z)
        return reconstruction_logits, log_q_z_given_x, log_p_z

class VAE:
    def __init__(self, latent_dim=32, n_IAF_steps=2, h_dim=200, IAF_node_width=320, encoder_fc_dim=450, decoder_fc_dim=450
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

    def get_reconstruction(self, x_data):
        # for visualisation
        return torch.sigmoid(self.VAE_model(x_data)[0]).cpu().detach().numpy()

    def get_latent_encoding(self, x_data):
        # for visualisation
        return self.VAE_model.encoder(x_data)[0].cpu().detach().numpy()


    def loss_function(self, reconstruction_logits, log_q_z_given_x, log_p_z, x_target):
        log_p_x_given_z = - torch.sum(self.BCE_loss(reconstruction_logits, x_target), dim=[1,2,3])
        log_p_x_given_z_per_batch = torch.mean(log_p_x_given_z)
        log_q_z_given_x_per_batch = torch.mean(log_q_z_given_x)
        log_p_z_per_batch = torch.mean(log_p_z)
        ELBO = log_p_x_given_z_per_batch + log_p_z_per_batch - log_q_z_given_x_per_batch
        loss = -ELBO
        return loss, log_p_x_given_z_per_batch, log_q_z_given_x_per_batch, log_p_z_per_batch

    def get_marginal_batch(self, x_batch, n_samples = 128):
        """
       This first calculates the estimate of the marginal p(x) for each datapoint in the test set, using importance sampling
       and then returns the mean p(x) across the different points
        """
        running_mean = 0
        for n in range(n_samples):
            reconstruction_logits, log_q_z_given_x, log_p_z = self.VAE_model(x_batch)
            log_p_x_given_z = - torch.sum(self.BCE_loss(reconstruction_logits, x_batch), dim=[1, 2, 3])
            log_monte_carlo_sample = log_p_x_given_z  + log_p_z - log_q_z_given_x
            monte_carlo_sample = torch.exp(log_monte_carlo_sample.type(torch.double)).cpu().detach().numpy()
            running_mean = running_mean + (monte_carlo_sample - running_mean)/(n + 1)
        return np.mean(np.log(running_mean))

    def get_marginal(self, test_loader, n_samples=128):
        marginal_running_mean = 0
        for i, (x,) in enumerate(test_loader):
            x = x.to(self.device)
            with torch.cuda.amp.autocast():
                marginal_batch = self.get_marginal_batch(x, n_samples=n_samples)
            marginal_running_mean = marginal_running_mean + (marginal_batch - marginal_running_mean)/(i + 1)
        return marginal_running_mean


    def train(self, EPOCHS, train_loader, test_loader=None):
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


            for i, (x,) in enumerate(train_loader):
                x = x.to(self.device)
                self.optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    reconstruction_logits, log_q_z_given_x, log_p_z = self.VAE_model(x)
                loss, log_p_x_given_z_per_batch, log_q_z_given_x_per_batch, log_p_z_per_batch = \
                    self.loss_function(reconstruction_logits, log_q_z_given_x, log_p_z, x)
                loss.backward()
                self.optimizer.step()
                running_loss = running_mean(loss.cpu().detach().numpy(), running_loss, i)
                running_log_q_z_given_x = running_mean(log_q_z_given_x_per_batch.cpu().detach().numpy(), running_log_q_z_given_x, i)
                running_log_p_x_given_z = running_mean(log_p_x_given_z_per_batch.cpu().detach().numpy(), running_log_p_x_given_z, i)
                running_log_p_z = running_mean(log_p_z_per_batch.cpu().detach().numpy(), running_log_p_z, i)


            if EPOCH % (round(EPOCHS/10) + 1) == 0 or EPOCH == EPOCHS - 1:
                print(f"Epoch: {EPOCH + 1} \n"
                      f"running loss: {running_loss} \n"
                      f"running_log_p_x_given_z: {running_log_p_x_given_z} \n"
                      f"running_log_q_z_given_x: {running_log_q_z_given_x} \n"
                      f"running_log_p_z: {running_log_p_z} \n")

                train_history["loss"].append(running_loss)
                train_history["log_p_x_given_z"].append(running_log_p_x_given_z)
                train_history["log_q_z_given_x"].append(running_log_q_z_given_x)
                train_history["log_p_z "].append(running_log_p_z)

            if test_loader is not None:
                for i, (x,) in enumerate(test_loader):
                    x = x.to(self.device)
                    with torch.cuda.amp.autocast():
                        reconstruction_logits, log_q_z_given_x, log_p_z = self.VAE_model(x)
                    loss, log_p_x_given_z_per_batch, log_q_z_given_x_per_batch, log_p_z_per_batch = \
                        self.loss_function(reconstruction_logits, log_q_z_given_x, log_p_z, x)
                    test_running_loss = running_mean(loss.cpu().detach().numpy(), test_running_loss, i)
                    test_running_log_q_z_given_x = running_mean(log_q_z_given_x_per_batch.cpu().detach().numpy(), test_running_log_q_z_given_x, i)
                    test_running_log_p_x_given_z = running_mean(log_p_x_given_z_per_batch.cpu().detach().numpy(), test_running_log_p_x_given_z, i)
                    test_running_log_p_z = running_mean(log_p_z_per_batch.cpu().detach().numpy(), test_running_log_p_z, i)

                if EPOCH % (round(EPOCHS/10) + 1) == 0 or EPOCH == EPOCHS - 1:
                    print(f"Epoch: {EPOCH + 1} \n"
                          f"test running loss: {test_running_loss} \n"
                          f"test running_log_p_x_given_z: {test_running_log_p_x_given_z} \n"
                          f"test running_log_q_z_given_x: {test_running_log_q_z_given_x} \n"
                          f"test running_log_p_z: {test_running_log_p_z} \n")
                    test_history["loss"].append(test_running_loss)
                    test_history["log_p_x_given_z"].append(test_running_log_p_x_given_z)
                    test_history["log_q_z_given_x"].append(test_running_log_q_z_given_x)
                    test_history["log_p_z "].append(test_running_log_p_z)

                if EPOCH % (round(EPOCHS/3) + 1) == 0:
                    p_x = self.get_marginal(test_loader, n_samples=20)
                    print(f"marginal log likelihood is {p_x}")

        p_x = self.get_marginal(test_loader, n_samples=128)
        print(f"marginal log likelihood is {p_x}")
        return train_history, test_history, p_x




if __name__ == "__main__":
    from Utils.load_binirised_mnist import load_data
    train_loader, test_loader = load_data(256)
    vae = VAE()
    vae.train(EPOCHS = 1, train_loader=train_loader, test_loader=test_loader)

    p_x = vae.get_marginal(test_loader, n_samples=20)
    print(f"marginal log likelihood is {p_x}")

    n = 5
    data_chunk = next(iter(train_loader))[0][0:n**2, :, :, :]
    fig, axs = plt.subplots(n, n)
    for i in range(n * n):
        row = int(i / n)
        col = i % n
        axs[row, col].imshow(np.squeeze(data_chunk [i, :, :, :]), cmap="gray")
        axs[row, col].axis('off')
    plt.show()

    n = 5
    prediction = vae.VAE_model(data_chunk)[0].detach().numpy()
    fig, axs = plt.subplots(n, n)
    for i in range(n * n):
        row = int(i / n)
        col = i % n
        axs[row, col].imshow(np.squeeze(prediction[i, :, :, :]), cmap="gray")
        axs[row, col].axis('off')
    plt.show()










