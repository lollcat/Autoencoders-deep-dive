from Utils.running_mean import running_mean
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datetime import datetime
import pathlib, os
#from tqdm import tqdm
from tqdm.notebook import tqdm
from CIFAR_ladder.model import VAE_ladder_model
from CIFAR_basic_IAF.VAE import VAE
from Utils.epoch_manager import EpochManager


class VAE_ladder(VAE):
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
        self.save_NN_path = f"Results_and_trained_models/CIFAR_ladder/saved_models/{name}__latent_dim_{latent_dim}" \
                            f"__n_IAF_steps_{n_IAF_steps}__n_rungs_{n_rungs}__constant_sigma_{constant_sigma}__" \
                            f"IAF_node_width_{IAF_node_width}/{current_time}/"

    @torch.no_grad()
    def get_marginal_batch(self, x_batch, n_samples = 128):
        """
       This first calculates the estimate of the marginal p(x) for each datapoint in the test set, using importance sampling
       and then returns the mean p(x) across the different points
       Note inner operating is aggregating across samples for each point, outer operation aggregates over different points
       Use log sum x trick described in https://blog.feedly.com/tricks-of-the-trade-logsumexp/, using pytorches library
        """
        samples = []
        for n in range(n_samples):
            reconstruction_mu, reconstruction_log_sigma, KL_free_bits_term,  KL_q_p = self.model(x_batch)
            log_p_x_given_z = self.discretized_log_lik(x_batch, reconstruction_mu, reconstruction_log_sigma)
            # dim batch_size
            log_monte_carlo_sample = log_p_x_given_z - KL_q_p
            samples.append(torch.unsqueeze(log_monte_carlo_sample, dim=0))
        log_monte_carlo_sample_s = torch.cat(samples, dim=0)
        log_p_x_per_sample = torch.logsumexp(log_monte_carlo_sample_s, dim=0) - torch.log(torch.tensor(n_samples).float())
        mean_log_p_x = torch.mean(log_p_x_per_sample)
        return mean_log_p_x.item()

    def loss_function(self, reconstruction_means, reconstruction_log_sigmas
                      , KL_free_bits_term, x_target):
        log_p_x_given_z = self.discretized_log_lik(x_target, reconstruction_means, reconstruction_log_sigmas)
        log_p_x_given_z_per_batch = torch.mean(log_p_x_given_z)
        ELBO = log_p_x_given_z_per_batch + KL_free_bits_term
        loss = -ELBO
        return loss, log_p_x_given_z_per_batch

    def train(self, EPOCHS, train_loader, test_loader=None, save_model=True,
              lr_decay=True, validation_based_decay = True, early_stopping=True,
              early_stopping_criterion=40):
        epoch_manager = EpochManager(self.optimizer, EPOCHS, lr_decay=lr_decay,
                                     early_stopping=early_stopping,
                                     early_stopping_criterion=early_stopping_criterion,
                                     validation_based_decay=validation_based_decay)
        epoch_per_info = max(round(EPOCHS / 10), 1)
        train_history = {"loss": [],
                         "log_p_x_given_z": [],
                         "KL": [],
                         "KL_free_bits": []}
        test_history = {"loss": [],
                     "log_p_x_given_z": [],
                     "KL": [],
                     "KL_free_bits": []}

        for EPOCH in tqdm(range(EPOCHS)):
            running_loss = 0
            running_log_p_x_given_z = 0
            running_KL = 0
            running_KL_free_bits = 0

            test_running_loss = 0
            test_running_log_p_x_given_z = 0
            test_running_KL = 0
            test_running_KL_free_bits = 0
            for i, (x, _) in enumerate(train_loader):
                x = x.to(self.device)
                reconstruction_mu, reconstruction_log_sigma, KL_free_bits_term,  KL_q_p = self.model(x)
                KL_mean = torch.mean(KL_q_p)
                loss, log_p_x_given_z_per_batch = self.loss_function(reconstruction_mu, reconstruction_log_sigma, KL_free_bits_term, x)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                running_loss = running_mean(loss.item(), running_loss, i)
                running_log_p_x_given_z = running_mean(log_p_x_given_z_per_batch.item(), running_log_p_x_given_z, i)
                running_KL = running_mean(KL_mean.item(), running_KL, i)
                running_KL_free_bits = running_mean(-KL_free_bits_term.item(), running_KL_free_bits, i)


            for i, (x, _) in enumerate(test_loader):
                torch.no_grad()
                x = x.to(self.device)
                reconstruction_mu, reconstruction_log_sigma, KL_free_bits_term, KL_q_p = self.model(x)
                KL_mean = torch.mean(KL_q_p)
                loss, log_p_x_given_z_per_batch = \
                    self.loss_function(reconstruction_mu, reconstruction_log_sigma, KL_free_bits_term, x)
                test_running_loss = running_mean(loss.item(), test_running_loss, i)
                test_running_log_p_x_given_z = running_mean(log_p_x_given_z_per_batch.item(), test_running_log_p_x_given_z, i)
                test_running_KL = running_mean(KL_mean.item(), test_running_KL, i)
                test_running_KL_free_bits = running_mean(-KL_free_bits_term.item(), test_running_KL_free_bits, i)

            train_history["loss"].append(running_loss)
            train_history["log_p_x_given_z"].append(running_log_p_x_given_z)
            train_history["KL"].append(running_KL)
            train_history["KL_free_bits"].append(running_KL_free_bits)
            test_history["loss"].append(test_running_loss)
            test_history["log_p_x_given_z"].append(test_running_log_p_x_given_z)
            test_history["KL"].append(test_running_KL)
            test_history["KL_free_bits"].append(test_running_KL_free_bits)

            if EPOCH % epoch_per_info == 0 or EPOCH == EPOCHS - 1:
                print(f"Epoch: {EPOCH + 1} \n"
                      f"running loss: {running_loss} \n"
                      f"running_log_p_x_given_z: {running_log_p_x_given_z} \n"
                      f"running_KL: {running_KL} \n"
                      f"running KL free bits {running_KL_free_bits}")
                print(f"test running loss: {test_running_loss} \n"
                      f"test running_log_p_x_given_z: {test_running_log_p_x_given_z} \n"
                      f"test running_KL: {test_running_KL} \n"
                      f"test running KL free bits {test_running_KL_free_bits}")

            halt_training = epoch_manager.manage(EPOCH, test_history["loss"])
            if halt_training:
                break

        if save_model is True:
            print("model saved")
            self.save_NN_model(EPOCHS)

        bits_per_dim = self.get_bits_per_dim(test_loader)
        print(bits_per_dim)
        return train_history, test_history, bits_per_dim


if __name__ == '__main__':
    from Utils.load_CIFAR import load_data
    from Utils.mnist_plotting import plot_train_test
    from Utils.CIFAR_plotting import plot_original_and_reconstruction
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = load_data(100)
    x_data = next(iter(train_loader))[0].to(device)
    experiment_dict = {"latent_dim": 3, "n_IAF_steps": 1, "IAF_node_width": 10, "n_rungs": 4}
    test_model = VAE_ladder(**experiment_dict)
    print(test_model.get_bits_per_dim(test_loader, n_samples=3))
    train_history, test_history, bits_per_dim = test_model.train(2, train_loader, test_loader,
                                                                  lr_schedule=False, n_lr_cycles=1,
                                                                  epoch_per_info_min=50,
                                                                  save_model=False)
    fig_original, axs_original, fig_reconstruct, axs_reconstruct = \
        plot_original_and_reconstruction(test_model, test_loader)
    import matplotlib.pyplot as plt
    plt.show()
    figure, axs = plot_train_test(train_history, test_history)
