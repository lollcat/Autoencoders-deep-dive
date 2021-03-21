from Utils.running_mean import running_mean
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datetime import datetime
import pathlib, os
from tqdm import tqdm

from CIFAR_ladder.model import VAE_ladder_model


class VAE_ladder:
    def __init__(self, latent_dim=32, n_rungs=4, n_IAF_steps=1, IAF_node_width=450, use_GPU = True, name=""):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = VAE_ladder_model(latent_dim=latent_dim, n_rungs=n_rungs, n_IAF_steps=n_IAF_steps,
                                      IAF_node_width=IAF_node_width).to(self.device)
        self.optimizer = torch.optim.Adamax(self.model.parameters(), lr=0.001)
        current_time = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
        self.save_NN_path = f"Results_and_trained_models/IAF_VAE_mnist/saved_models/{name}__latent_dim_{latent_dim}" \
                            f"__n_IAF_steps_{n_IAF_steps}__n_rungs_{n_rungs}__" \
                            f"IAF_node_width_{IAF_node_width}/{current_time}/"

    def save_NN_model(self, epochs_trained_for = 0, additional_name_info=""):
        model_path = self.save_NN_path + f"epochs_{epochs_trained_for}__model__" + additional_name_info
        pathlib.Path(os.path.join(os.getcwd(), model_path)).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), model_path)

    def load_NN_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=torch.device(self.device)))
        print(f"loaded model from {path}")

    def discretized_log_lik(self, sample, mu, log_sigma, binsize=1/256.0):
        # see https://github.com/openai/iaf/blob/ad33fe4872bf6e4b4f387e709a625376bb8b0d9d/tf_utils/distributions.py#L28
        scale = torch.exp(log_sigma)
        sample = (torch.floor(sample / binsize) * binsize - mu) / scale
        log_p_x_given_z = torch.log(torch.sigmoid(sample + binsize / scale) - torch.sigmoid(sample) + 1e-7)
        return torch.sum(log_p_x_given_z, dim=[1, 2, 3])

    def get_bits_per_dim(self, test_loader):
        num_pixels = 32*32
        ELBO = 0
        for i, (x, _) in enumerate(test_loader):
            x = x.to(self.device)
            reconstruction_mu, reconstruction_log_sigma, KL_ELBO_term = self.model(x)
            batch_ELBO = - self.loss_function(reconstruction_mu, reconstruction_log_sigma, KL_ELBO_term, x)[0].item()
            ELBO += (batch_ELBO - ELBO)/(i+1)
        return - ELBO/num_pixels/np.log(2)

    def loss_function(self, reconstruction_means, reconstruction_log_sigmas
                      , KL_ELBO_term, x_target):
        log_p_x_given_z = self.discretized_log_lik(x_target, reconstruction_means, reconstruction_log_sigmas)
        log_p_x_given_z_per_batch = torch.mean(log_p_x_given_z)
        ELBO = log_p_x_given_z_per_batch + KL_ELBO_term
        loss = -ELBO
        return loss, log_p_x_given_z_per_batch, KL_ELBO_term

    def train(self, EPOCHS, train_loader, test_loader, lr_schedule=False, n_lr_cycles=1, epoch_per_info_min=50,
              save_model=False):
        epoch_per_info = max(min(100, round(EPOCHS / 10)), 1)
        n_train_batches = len(train_loader)
        train_history = {"loss": [],
                         "log_p_x_given_z": [],
                         "KL_ELBO_term": []}
        test_history = {"loss": [],
                     "log_p_x_given_z": [],
                     "KL_ELBO_term": []}
        if lr_schedule is True:  # number of decay steps
            n_decay_steps = 5
            epoch_per_decay = max(int(EPOCHS/n_lr_cycles / n_decay_steps), 1)
            epoch_per_cycle = int(EPOCHS/n_lr_cycles) + 2
            original_lr = self.optimizer.param_groups[0]["lr"]
        epoch_per_info = max(min(epoch_per_info_min, round(EPOCHS / 10)), 1)

        for EPOCH in tqdm(range(EPOCHS)):
            running_loss = 0
            running_log_p_x_given_z = 0
            running_KL_ELBO_term = 0

            test_running_loss = 0
            test_running_log_p_x_given_z = 0
            test_running_KL_ELBO_term = 0
            for i, (x, _) in enumerate(train_loader):
                x = x.to(self.device)
                reconstruction_mu, reconstruction_log_sigma, KL_ELBO_term = self.model(x)
                loss, log_p_x_given_z_per_batch, KL_ELBO_term = \
                    self.loss_function(reconstruction_mu, reconstruction_log_sigma, KL_ELBO_term, x)
                loss, log_p_x_given_z_per_batch, KL_ELBO_term = \
                    self.loss_function(reconstruction_mu, reconstruction_log_sigma, KL_ELBO_term, x)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                running_loss = running_mean(loss.item(), running_loss, i)
                running_log_p_x_given_z = running_mean(log_p_x_given_z_per_batch.item(), running_log_p_x_given_z, i)
                running_KL_ELBO_term = running_mean(-KL_ELBO_term.item(), running_KL_ELBO_term, i)

            if lr_schedule is True and EPOCH > 50: # use max lr for first 50 epoch
                if EPOCH % epoch_per_cycle == 0:
                    print("learning rate reset")
                    self.optimizer.param_groups[0]["lr"] = original_lr
                elif EPOCH % epoch_per_decay == 0:
                    print("learning rate decayed")
                    self.optimizer.param_groups[0]["lr"] *= 0.5



            for i, (x, _) in enumerate(test_loader):
                torch.no_grad()
                x = x.to(self.device)
                reconstruction_mu, reconstruction_log_sigma, KL_ELBO_term = self.model(x)
                loss, log_p_x_given_z_per_batch, KL_ELBO_term = \
                    self.loss_function(reconstruction_mu, reconstruction_log_sigma, KL_ELBO_term, x)
                loss, log_p_x_given_z_per_batch, KL_ELBO_term = \
                    self.loss_function(reconstruction_mu, reconstruction_log_sigma, KL_ELBO_term, x)
                test_running_loss = running_mean(loss.item(), test_running_loss, i)
                test_running_log_p_x_given_z = running_mean(log_p_x_given_z_per_batch.item(), test_running_log_p_x_given_z, i)
                test_running_KL_ELBO_term = running_mean(KL_ELBO_term.item(), -test_running_KL_ELBO_term, i)


            train_history["loss"].append(running_loss)
            train_history["log_p_x_given_z"].append(running_log_p_x_given_z)
            train_history["KL_ELBO_term"].append(running_KL_ELBO_term)
            test_history["loss"].append(test_running_loss)
            test_history["log_p_x_given_z"].append(test_running_log_p_x_given_z)
            test_history["KL_ELBO_term"].append(test_running_KL_ELBO_term)

            if EPOCH % epoch_per_info == 0 or EPOCH == EPOCHS - 1:
                print(f"Epoch: {EPOCH + 1} \n"
                      f"running loss: {running_loss} \n"
                      f"running_log_p_x_given_z: {running_log_p_x_given_z} \n"
                      f"running_KL_ELBO_term: {running_KL_ELBO_term} \n")
                print(f"test running loss: {test_running_loss} \n"
                      f"test running_log_p_x_given_z: {test_running_log_p_x_given_z} \n"
                      f"test running_KL_ELBO_term: {test_running_KL_ELBO_term} \n")

        if save_model is True:
            print("model saved")
            self.save_NN_model(EPOCHS)

        bits_per_dim = self.get_bits_per_dim(test_loader)
        print(bits_per_dim)
        return train_history, test_history, bits_per_dim


if __name__ == '__main__':
    from Utils.load_CIFAR import load_data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = load_data(26)
    x_data = next(iter(train_loader))[0].to(device)
    test_model = VAE_ladder(latent_dim=2, n_rungs=2, n_IAF_steps=1, IAF_node_width=45)
    reconstruction_mu, reconstruction_log_sigma, KL_ELBO_term = test_model.model(x_data)
    loss, log_p_x_given_z_per_batch, KL_ELBO_term = test_model.loss_function(reconstruction_mu, reconstruction_log_sigma, KL_ELBO_term, x_data)
    print(loss, log_p_x_given_z_per_batch, KL_ELBO_term)
    train_history, test_history, bits_per_dim = test_model.train(2, train_loader, test_loader,
                                                                  lr_schedule=False, n_lr_cycles=1,
                                                                  epoch_per_info_min=50,
                                                                  save_model=False)
