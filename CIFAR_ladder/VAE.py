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
    def __init__(self, latent_dim=32, n_rungs=4, n_IAF_steps=1, IAF_node_width=450, use_GPU = True, name="",
                 constant_sigma=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = VAE_ladder_model(latent_dim=latent_dim, n_rungs=n_rungs, n_IAF_steps=n_IAF_steps,
                                      IAF_node_width=IAF_node_width, constant_sigma=constant_sigma).to(self.device)
        self.optimizer = torch.optim.Adamax(self.model.parameters())
        current_time = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
        self.save_NN_path = f"Results_and_trained_models/IAF_VAE_mnist/saved_models/{name}__latent_dim_{latent_dim}" \
                            f"__n_IAF_steps_{n_IAF_steps}__n_rungs_{n_rungs}__constant_sigma_{constant_sigma}__" \
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

    def get_marginal_batch(self, x_batch, n_samples = 128):
        """
       This first calculates the estimate of the marginal p(x) for each datapoint in the test set, using importance sampling
       and then returns the mean p(x) across the different points
        """
        running_mean = 0
        for n in range(n_samples):
            reconstruction_mu, reconstruction_log_sigma, KL_ELBO_term = self.model(x_batch)
            log_monte_carlo_sample = - self.loss_function(reconstruction_mu, reconstruction_log_sigma, KL_ELBO_term, x_batch)[0].item()
            monte_carlo_sample = torch.exp(log_monte_carlo_sample.type(torch.double)).cpu().detach().numpy()
            running_mean = running_mean + (monte_carlo_sample - running_mean)/(n + 1)
        return np.mean(np.log(running_mean))

    def get_marginal(self, test_loader, n_samples=128):
        marginal_running_mean = 0
        for i, (x, _) in enumerate(test_loader):
            x = x.to(self.device)
            marginal_batch = self.get_marginal_batch(x, n_samples=n_samples)
            marginal_running_mean = marginal_running_mean + (marginal_batch - marginal_running_mean)/(i + 1)
        return marginal_running_mean

    def get_bits_per_dim(self, test_loader, n_samples=128):
        num_pixels = 32*32
        log_p_x = self.get_marginal(test_loader=test_loader, n_samples=n_samples)
        return - log_p_x/num_pixels/np.log(2)

    def loss_function(self, reconstruction_means, reconstruction_log_sigmas
                      , KL_free_bits_term, x_target):
        log_p_x_given_z = self.discretized_log_lik(x_target, reconstruction_means, reconstruction_log_sigmas)
        log_p_x_given_z_per_batch = torch.mean(log_p_x_given_z)
        ELBO = log_p_x_given_z_per_batch + KL_free_bits_term
        loss = -ELBO
        return loss, log_p_x_given_z_per_batch

    def train(self, EPOCHS, train_loader, test_loader, lr_schedule=False, n_lr_cycles=1, epoch_per_info_min=50,
              save_model=False):
        if lr_schedule is True:  # number of decay steps
            n_decay_steps = 5
            epoch_per_decay = max(int(EPOCHS/n_lr_cycles / n_decay_steps), 1)
            epoch_per_cycle = int(EPOCHS/n_lr_cycles) + 2
            original_lr = self.optimizer.param_groups[0]["lr"]
        epoch_per_info = max(min(epoch_per_info_min, round(EPOCHS / 10)), 1)

        train_history = {"loss": [],
                         "log_p_x_given_z": [],
                         "KL_ELBO_term": [],
                         "KL_free_bits": []}
        test_history = {"loss": [],
                     "log_p_x_given_z": [],
                     "KL_ELBO_term": [],
                     "KL_free_bits": []}

        for EPOCH in tqdm(range(EPOCHS)):
            running_loss = 0
            running_log_p_x_given_z = 0
            running_KL_ELBO_term = 0
            running_KL_free_bits = 0

            test_running_loss = 0
            test_running_log_p_x_given_z = 0
            test_running_KL_ELBO_term = 0
            test_running_KL_free_bits = 0
            for i, (x, _) in enumerate(train_loader):
                x = x.to(self.device)
                reconstruction_mu, reconstruction_log_sigma, KL_free_bits_term,  KL_q_p = self.model(x)
                loss, log_p_x_given_z_per_batch = self.loss_function(reconstruction_mu, reconstruction_log_sigma, KL_free_bits_term, x)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                running_loss = running_mean(loss.item(), running_loss, i)
                running_log_p_x_given_z = running_mean(log_p_x_given_z_per_batch.item(), running_log_p_x_given_z, i)
                running_KL_ELBO_term = running_mean(KL_ELBO_term.item(), running_KL_ELBO_term, i)
                running_KL_free_bits = running_mean(-KL_free_bits_term.item(), running_KL_ELBO_term, i)

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
                reconstruction_mu, reconstruction_log_sigma, KL_free_bits_term, KL_q_p = self.model(x)
                loss, log_p_x_given_z_per_batch = \
                    self.loss_function(reconstruction_mu, reconstruction_log_sigma, KL_free_bits_term, x)
                test_running_loss = running_mean(loss.item(), test_running_loss, i)
                test_running_KL_ELBO_term = running_mean(KL_ELBO_term.item(), test_running_KL_ELBO_term, i)
                test_running_KL_free_bits = running_mean(-KL_free_bits_term.item(), test_running_KL_free_bits, i)


            train_history["loss"].append(running_loss)
            train_history["log_p_x_given_z"].append(running_log_p_x_given_z)
            train_history["KL_ELBO_term"].append(running_KL_ELBO_term)
            train_history["KL_free_bits"].append(running_KL_free_bits)
            test_history["loss"].append(test_running_loss)
            test_history["log_p_x_given_z"].append(test_running_log_p_x_given_z)
            test_history["KL_ELBO_term"].append(test_running_KL_ELBO_term)
            test_history["KL_free_bits"].append(test_running_KL_free_bits)

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
    reconstruction_mu, reconstruction_log_sigma, KL_free_bits_term,  KL_q_p = test_model.model(x_data)
    loss, log_p_x_given_z_per_batch, KL_ELBO_term = \
        test_model.loss_function(reconstruction_mu, reconstruction_log_sigma, KL_free_bits_term, x_data)
    print(loss, log_p_x_given_z_per_batch, KL_ELBO_term)
    train_history, test_history, bits_per_dim = test_model.train(3, train_loader, test_loader,
                                                                  lr_schedule=False, n_lr_cycles=1,
                                                                  epoch_per_info_min=50,
                                                                  save_model=False)
