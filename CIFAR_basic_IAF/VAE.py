from CIFAR_basic_IAF.Encoder import Encoder
from CIFAR_basic_IAF.Decoder import Decoder
from Utils.running_mean import running_mean
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
#from tqdm import tqdm
from tqdm.notebook import tqdm
import pathlib, os
from datetime import datetime
from Utils.epoch_manager import EpochManager
import pandas as pd


class VAE_model(nn.Module):
    def __init__(self, latent_dim, n_IAF_steps, h_dim, IAF_node_width=320, encoder_fc_dim=450, decoder_fc_dim=450,
                 constant_sigma=False):
        super(VAE_model, self).__init__()
        self.encoder = Encoder(latent_dim=latent_dim, fc_layer_dim=encoder_fc_dim, n_IAF_steps=n_IAF_steps,
                               h_dim=h_dim, IAF_node_width=IAF_node_width, constant_sigma=constant_sigma)
        self.decoder = Decoder(latent_dim=latent_dim, fc_dim=decoder_fc_dim)

    def forward(self, x):
        z, log_q_z_given_x, log_p_z = self.encoder(x)
        reconstruction_means, reconstruction_log_sigma = self.decoder(z)
        return reconstruction_means, reconstruction_log_sigma, log_q_z_given_x, log_p_z

class VAE:
    def __init__(self, latent_dim=32, n_IAF_steps=2, h_dim=200, IAF_node_width=320, encoder_fc_dim=450, decoder_fc_dim=450
                 , use_GPU = True, name="", constant_sigma=False):
        if use_GPU is True:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = "cpu"
        print(f"running using {self.device}")
        self.model = VAE_model(latent_dim, n_IAF_steps, h_dim, IAF_node_width, encoder_fc_dim, decoder_fc_dim,
                               constant_sigma=constant_sigma)\
            .to(self.device)
        self.BCE_loss = torch.nn.BCEWithLogitsLoss(reduction="none")
        self.optimizer = torch.optim.Adamax(self.model.parameters())
        current_time = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
        self.save_NN_path = f"Results_and_trained_models/CIFAR_basic_IAF/{name}__latent_dim_{latent_dim}" \
                            f"__n_IAF_steps_{n_IAF_steps}__constant_sigma_{constant_sigma}__" \
                            f"IAF_node_width_{IAF_node_width}/{current_time}/"

    def get_reconstruction(self, x_data):
        # for visualisation
        return self.model(x_data)[0].cpu().detach().numpy()

    def get_latent_encoding(self, x_data):
        # for visualisation
        return self.model.encoder(x_data)[0].cpu().detach().numpy()

    def save_NN_model(self, epochs_trained_for = 0, additional_name_info=""):
        base_dir = pathlib.Path.cwd().parent
        model_path = base_dir / self.save_NN_path / f"epochs_{epochs_trained_for}__model__{additional_name_info}"
        pathlib.Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), str(model_path))

    def save_training_info(self, numpy_dicts, single_value_dict):
        base_dir = pathlib.Path.cwd().parent
        main_path = base_dir / self.save_NN_path
        pathlib.Path(main_path).mkdir(parents=True, exist_ok=True)
        for numpy_dict in numpy_dicts: # train and test info
            df = pd.DataFrame(numpy_dict)
            df_path = main_path / f'{numpy_dict["name"]}.csv'
            df.to_csv(df_path)

        summary_results = ""
        for key in single_value_dict:
            summary_results += f"{key} : {single_value_dict[key]}\n"
        summary_results_path = str(main_path / "summary_results.txt")
        with open(summary_results_path, "w") as g:
            g.write(summary_results)

    def load_NN_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=torch.device(self.device)))
        print(f"loaded model from {path}")

    @torch.no_grad()
    def get_marginal_batch(self, x_batch, n_samples = 128):
        """
       This first calculates the estimate of the marginal p(x) for each datapoint in the test set, using importance sampling
       and then returns the mean p(x) across the different points
       Note inner operating is aggregating across samples for each point, outer operation aggregates over different points
       Use log sum x trick dessribed in https://blog.feedly.com/tricks-of-the-trade-logsumexp/, using pytorches library
        """
        samples = []
        for n in range(n_samples):
            reconstruction_means, reconstruction_log_sigmas, log_q_z_given_x, log_p_z = self.model(x_batch)
            log_p_x_given_z = self.discretized_log_lik(x_batch, reconstruction_means, reconstruction_log_sigmas)
            # dim batch_size
            log_monte_carlo_sample = log_p_x_given_z - log_q_z_given_x + log_p_z
            samples.append(torch.unsqueeze(log_monte_carlo_sample, dim=0))
        log_monte_carlo_sample_s = torch.cat(samples, dim=0)
        log_p_x_per_sample = torch.logsumexp(log_monte_carlo_sample_s, dim=0) - torch.log(torch.tensor(n_samples).float())
        mean_log_p_x = torch.mean(log_p_x_per_sample)
        return mean_log_p_x.item()

    @torch.no_grad()
    def get_marginal(self, test_loader, n_samples=128):
        marginal_running_mean = 0
        for i, (x,_) in enumerate(test_loader):
            x = x.to(self.device)
            marginal_batch = self.get_marginal_batch(x, n_samples=n_samples)
            marginal_running_mean = marginal_running_mean + (marginal_batch - marginal_running_mean)/(i + 1)
        return marginal_running_mean

    def get_bits_per_dim(self, test_loader, n_samples=128):
        num_pixels = 32*32*3
        log_p_x = self.get_marginal(test_loader=test_loader, n_samples=n_samples)
        return - log_p_x/num_pixels/np.log(2)

    def discretized_log_lik(self, sample, mu, log_sigma, binsize=1/256.0):
        # based on https://github.com/openai/iaf/blob/ad33fe4872bf6e4b4f387e709a625376bb8b0d9d/tf_utils/distributions.py#L28
        sample = sample - 0.5
        mu = torch.clip(mu - 0.5, -0.5 + 1 / 512., 0.5 - 1 / 512)
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

    def train(self, EPOCHS, train_loader, test_loader, save=True,
              lr_decay=True, validation_based_decay = True, early_stopping=True,
              early_stopping_criterion=40):
        epoch_manager = EpochManager(self.optimizer, EPOCHS, lr_decay=lr_decay,
                                     early_stopping=early_stopping,
                                     early_stopping_criterion=early_stopping_criterion,
                                     validation_based_decay=validation_based_decay)
        epoch_per_info = max(round(EPOCHS / 10), 1)
        train_history = {"name": "train",
                        "loss": [],
                         "log_p_x_given_z": [],
                         "log_q_z_given_x": [],
                         "log_p_z ": []}
        test_history = {"name": "test",
                        "loss": [],
                     "log_p_x_given_z": [],
                     "log_q_z_given_x": [],
                     "log_p_z ": []}


        for EPOCH in tqdm(range(EPOCHS)):
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
                reconstruction_means, reconstruction_log_vars, log_q_z_given_x, log_p_z = self.model(x)
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
                torch.no_grad()
                reconstruction_means, reconstruction_log_vars, log_q_z_given_x, log_p_z = self.model(x)
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

            halt_training = epoch_manager.manage(EPOCH, test_history["loss"])
            if halt_training:
                break

            if save is True and EPOCH % (round(EPOCHS / 3) + 1) == 0 and EPOCH > 10:
                print(f"saving checkpoint model at epoch {EPOCH}")
                self.save_NN_model(EPOCH)

        bits_per_dim = self.get_bits_per_dim(test_loader=test_loader)
        bits_per_dim_lower_bound = self.get_bits_per_dim(test_loader, n_samples=1)  # 1 sample gives us lower bound
        print(f"{bits_per_dim} bits per dim \n {bits_per_dim_lower_bound} bits per dim lower bound")
        if save is True:
            print("model saved")
            self.save_NN_model(EPOCHS)
            self.save_training_info(numpy_dicts=[train_history, test_history],
                                    single_value_dict={"bits per dim": bits_per_dim,
                                                       "bits per dim lower bound": bits_per_dim_lower_bound,
                                                       "test loss": -test_history['loss'][-1],
                                                       "train loss": -train_history['loss'][-1],
                                                       "EPOCHS MAX": EPOCHS,
                                                       "EPOCHS Actual": EPOCH + 1,
                                                       "lr_decay": lr_decay,
                                                       "early_stopping": early_stopping,
                                                       "early_stopping_criterion": early_stopping_criterion,
                                                       "validation_based_decay": validation_based_decay})
        return train_history, test_history, bits_per_dim



if __name__ == "__main__":
    from Utils.load_CIFAR import load_data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = load_data(200)
    data_chunk = next(iter(train_loader))[0]
    vae = VAE()
    print(vae.get_bits_per_dim(test_loader=test_loader, n_samples=3))
    vae.train(EPOCHS = 1, train_loader=train_loader, test_loader=test_loader, save=False)
    """
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
    """










