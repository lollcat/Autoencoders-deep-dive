import pandas as pd
from Utils.running_mean import running_mean
import numpy as np
import torch
from Utils.epoch_manager import EpochManager
import pathlib, os
#from tqdm import tqdm
from tqdm.notebook import tqdm

class CIFAR_BASE:
    # base class used for CIFAR models
    def __init__(self):
        self.model = None # NN model
        self.save_NN_path = None # path where results are saved to
        self.device = None # gpu or cpu
        self.optimizer = None

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
        scale = torch.exp(log_sigma)
        sample = (torch.floor(sample / binsize) * binsize - mu) / scale
        log_p_x_given_z = torch.log(torch.sigmoid(sample + binsize / scale) - torch.sigmoid(sample) + 1e-7)
        return torch.sum(log_p_x_given_z, dim=[1, 2, 3])

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

    def train(self, EPOCHS, train_loader, test_loader=None, save=True,
              lr_decay=True, validation_based_decay=True, early_stopping=True,
              early_stopping_criterion=40):
        epoch_manager = EpochManager(self.optimizer, EPOCHS, lr_decay=lr_decay,
                                     early_stopping=early_stopping,
                                     early_stopping_criterion=early_stopping_criterion,
                                     validation_based_decay=validation_based_decay)
        epoch_per_info = max(round(EPOCHS / 10), 1)
        train_history = {"name": "train",
                         "loss": [],
                         "log_p_x_given_z": [],
                         "KL": [],
                         "KL_free_bits": []}
        test_history = {"name": "test",
                        "loss": [],
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
                if torch.isnan(loss).item():
                    raise Exception("NAN loss encountered")
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
                print(f"\n\ntest running loss: {test_running_loss} \n"
                      f"test running_log_p_x_given_z: {test_running_log_p_x_given_z} \n"
                      f"test running_KL: {test_running_KL} \n"
                      f"test running KL free bits {test_running_KL_free_bits}")

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