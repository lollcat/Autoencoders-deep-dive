from IAF_VAE_mnist.Encoder import Encoder
#from IAF_VAE_mnist.Decoder import Decoder
from IAF_VAE_mnist.Decoder_new import Decoder
from Utils.running_mean import running_mean
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datetime import datetime
import pathlib, os

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

        current_time = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
        self.save_NN_path = f"IAF_VAE_mnist/saved_models/latent_dim_{latent_dim}__n_IAF_steps_{n_IAF_steps}__h_dim_{200}_" \
                            f"IAF_node_width_{IAF_node_width}/{current_time}/"
        self.BCE_loss = torch.nn.BCEWithLogitsLoss(reduction="none")
        self.optimizer = torch.optim.Adamax(self.VAE_model.parameters())

    def get_reconstruction(self, x_data):
        # for visualisation
        return torch.sigmoid(self.VAE_model(x_data)[0]).cpu().detach().numpy()

    def save_NN_model(self, epochs_trained_for = 0):
        model_path = self.save_NN_path + f"epochs_{epochs_trained_for}__model"
        pathlib.Path(os.path.join(os.getcwd(), model_path)).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.VAE_model.state_dict(), model_path)

    def load_NN_model(self, path):
        self.VAE_model.load_state_dict(torch.load(path, map_location=torch.device(self.device)))

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


    def train(self, EPOCHS, train_loader, test_loader=None, save_model=True, decay_lr=True,
              save_info_during_training=True):
        """
        :param EPOCHS: number of epochs
        :param train_loader: train data loader
        :param test_loader: test data loader
        :param save_model: whether to save model during training
        :param decay_lr: whether to decay learning rate
        :param save_info_during_training: whether to save & display loss throughout training
        :return: train_history, test_history, p_x (depends whichever of these exist based on function settings
        """
        if decay_lr is True:  # number of decay steps
            epoch_per_decay = int(EPOCHS / 10)

        if save_info_during_training is True:
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
            if save_info_during_training is True:
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
                if save_info_during_training is True:
                    running_loss = running_mean(loss.cpu().detach().numpy(), running_loss, i)
                    running_log_q_z_given_x = running_mean(log_q_z_given_x_per_batch.cpu().detach().numpy(), running_log_q_z_given_x, i)
                    running_log_p_x_given_z = running_mean(log_p_x_given_z_per_batch.cpu().detach().numpy(), running_log_p_x_given_z, i)
                    running_log_p_z = running_mean(log_p_z_per_batch.cpu().detach().numpy(), running_log_p_z, i)

            if decay_lr is True and EPOCH % epoch_per_decay == 0 and EPOCH > 5:
                print("learning rate decayed")
                self.optimizer.param_groups[0]["lr"] *= 0.5

            if save_info_during_training is True:
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

            if test_loader is not None and save_info_during_training is True:
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

            if save_model is True and EPOCH % (round(EPOCHS/10) + 1) == 0 and EPOCH > 10:
                print(f"saving checkpoint model at epoch {EPOCH}")
                self.save_NN_model(EPOCH)

            if save_info_during_training is False and EPOCH % (round(EPOCHS/10) + 1) == 0:
                print(f"EPOCH {EPOCH}")

        if save_model is True:
            print("model saved")
            self.save_NN_model(EPOCHS)

        if test_loader is not None:
            p_x = self.get_marginal(test_loader, n_samples=128)
            print(f"marginal log likelihood is {p_x}")
            return train_history, test_history, p_x
        elif save_info_during_training is True
            return train_history
        else
            return


if __name__ == "__main__":
    from Utils.load_binirised_mnist import load_data
    train_loader, test_loader = load_data(256)
    vae = VAE(latent_dim=2, n_IAF_steps=2, h_dim=2, IAF_node_width=32, encoder_fc_dim=45, decoder_fc_dim=45)
    vae.train(EPOCHS = 1, train_loader=train_loader) #, test_loader=test_loader)

    """
    vae.save_NN_model(0)
    vae.save_NN_model(1)
    vae.load_NN_model("IAF_VAE_mnist/saved_models/latent_dim_2__n_IAF_steps_2__h_dim_200_IAF_node_width_32/2021_03_15-03_52_50_PM/epochs_1__model")

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
    """










