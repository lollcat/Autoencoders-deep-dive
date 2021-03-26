from IAF_VAE_mnist.Encoder import Encoder
from IAF_VAE_mnist.Decoder_new import Decoder
from Utils.running_mean import running_mean
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datetime import datetime
import pathlib, os
#from tqdm import tqdm
from tqdm.notebook import tqdm
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
        reconstruction_logits = self.decoder(z)
        return reconstruction_logits, log_q_z_given_x, log_p_z


class VAE:
    def __init__(self, latent_dim=32, n_IAF_steps=2, h_dim=200, IAF_node_width=320, encoder_fc_dim=450, decoder_fc_dim=450
                 , use_GPU = True, name="", constant_sigma=False):
        if use_GPU is True:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = "cpu"
        print(f"running using {self.device}")
        self.model = VAE_model(latent_dim, n_IAF_steps, h_dim, IAF_node_width, encoder_fc_dim,
                               decoder_fc_dim, constant_sigma=constant_sigma)\
            .to(self.device)

        current_time = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
        self.save_NN_path = f"Results_and_trained_models/IAF_VAE_mnist/{name}__latent_dim_{latent_dim}" \
                            f"__n_IAF_steps_{n_IAF_steps}__h_dim_{h_dim}_constant_sigma_{constant_sigma}__" \
                            f"IAF_node_width_{IAF_node_width}/{current_time}/"
        self.BCE_loss = torch.nn.BCEWithLogitsLoss(reduction="none")
        self.optimizer = torch.optim.Adamax(self.model.parameters())

    def get_reconstruction(self, x_data):
        # for visualisation
        return torch.sigmoid(self.model(x_data)[0]).cpu().detach().numpy()

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

    def get_latent_encoding(self, x_data):
        # for visualisation
        return self.model.encoder(x_data)[0].cpu().detach().numpy()

    def get_reconstruction_from_latent(self, z):
        return torch.sigmoid(self.model.decoder(z)).cpu().detach().numpy()


    def loss_function(self, reconstruction_logits, log_q_z_given_x, log_p_z, x_target):
        log_p_x_given_z = - torch.sum(self.BCE_loss(reconstruction_logits, x_target), dim=[1,2,3])
        log_p_x_given_z_per_batch = torch.mean(log_p_x_given_z)
        log_q_z_given_x_per_batch = torch.mean(log_q_z_given_x)
        log_p_z_per_batch = torch.mean(log_p_z)
        ELBO = log_p_x_given_z_per_batch + log_p_z_per_batch - log_q_z_given_x_per_batch
        loss = -ELBO
        #loss = torch.clamp_max(loss, 1e4)
        return loss, log_p_x_given_z_per_batch, log_q_z_given_x_per_batch, log_p_z_per_batch

    def get_marginal_batch(self, x_batch, n_samples = 128):
        """
       This first calculates the estimate of the marginal p(x) for each datapoint in the test set, using importance sampling
       and then returns the mean p(x) across the different points
        """
        running_mean = 0
        for n in range(n_samples):
            reconstruction_logits, log_q_z_given_x, log_p_z = self.model(x_batch)
            log_p_x_given_z = - torch.sum(self.BCE_loss(reconstruction_logits, x_batch), dim=[1, 2, 3])
            log_monte_carlo_sample = log_p_x_given_z  + log_p_z - log_q_z_given_x
            monte_carlo_sample = torch.exp(log_monte_carlo_sample.type(torch.double)).cpu().detach().numpy()
            running_mean = running_mean + (monte_carlo_sample - running_mean)/(n + 1)
        return np.mean(np.log(running_mean))

    def get_marginal(self, test_loader, n_samples=128):
        marginal_running_mean = 0
        for i, (x,) in enumerate(test_loader):
            x = x.to(self.device)
            marginal_batch = self.get_marginal_batch(x, n_samples=n_samples)
            marginal_running_mean = marginal_running_mean + (marginal_batch - marginal_running_mean)/(i + 1)
        return marginal_running_mean


    def train(self, EPOCHS, train_loader, test_loader=None, save=True,
              lr_decay=True, validation_based_decay = True, early_stopping=True,
              early_stopping_criterion=40):
        """
        :param EPOCHS: number of epochs
        :param train_loader: train data loader
        :param test_loader: test data loader
        :param save: whether to save model during training
        :param lr_decay: whether to decay learning rate
        :param save_info_during_training: whether to save & display loss throughout training
        :return: train_history, test_history, p_x (depends whichever of these exist based on function settings
        """
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
        if test_loader is not None:
            test_history = {"name":"test",
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


            for i, (x,) in enumerate(train_loader):
                x = x.to(self.device)
                reconstruction_logits, log_q_z_given_x, log_p_z = self.model(x)
                loss, log_p_x_given_z_per_batch, log_q_z_given_x_per_batch, log_p_z_per_batch = \
                    self.loss_function(reconstruction_logits, log_q_z_given_x, log_p_z, x)
                if torch.isnan(loss).item():
                    raise Exception("NAN loss encountered")
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                running_loss = running_mean(loss.item(), running_loss, i)
                running_log_q_z_given_x = running_mean(log_q_z_given_x_per_batch.item(), running_log_q_z_given_x, i)
                running_log_p_x_given_z = running_mean(log_p_x_given_z_per_batch.item(), running_log_p_x_given_z, i)
                running_log_p_z = running_mean(log_p_z_per_batch.item(), running_log_p_z, i)
                if i > 2:
                    break



            train_history["loss"].append(running_loss)
            train_history["log_p_x_given_z"].append(running_log_p_x_given_z)
            train_history["log_q_z_given_x"].append(running_log_q_z_given_x)
            train_history["log_p_z "].append(running_log_p_z)
            if EPOCH % epoch_per_info == 0 or EPOCH == EPOCHS - 1:
                print(f"Epoch: {EPOCH + 1} \n"
                      f"running loss: {running_loss} \n"
                      f"running_log_p_x_given_z: {running_log_p_x_given_z} \n"
                      f"running_log_q_z_given_x: {running_log_q_z_given_x} \n"
                      f"running_log_p_z: {running_log_p_z} \n")


            if test_loader is not None:
                for i, (x,) in enumerate(test_loader):
                    torch.no_grad()
                    x = x.to(self.device)
                    with torch.cuda.amp.autocast():
                        reconstruction_logits, log_q_z_given_x, log_p_z = self.model(x)
                    loss, log_p_x_given_z_per_batch, log_q_z_given_x_per_batch, log_p_z_per_batch = \
                        self.loss_function(reconstruction_logits, log_q_z_given_x, log_p_z, x)
                    test_running_loss = running_mean(loss.item(), test_running_loss, i)
                    test_running_log_q_z_given_x = running_mean(log_q_z_given_x_per_batch.item(), test_running_log_q_z_given_x, i)
                    test_running_log_p_x_given_z = running_mean(log_p_x_given_z_per_batch.item(), test_running_log_p_x_given_z, i)
                    test_running_log_p_z = running_mean(log_p_z_per_batch.item(), test_running_log_p_z, i)
                    if i > 2:
                        break


                test_history["loss"].append(test_running_loss)
                test_history["log_p_x_given_z"].append(test_running_log_p_x_given_z)
                test_history["log_q_z_given_x"].append(test_running_log_q_z_given_x)
                test_history["log_p_z "].append(test_running_log_p_z)

                if EPOCH % epoch_per_info == 0 or EPOCH == EPOCHS - 1:
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

        if test_loader is not None:
            log_p_x = self.get_marginal(test_loader, n_samples=1)
            print(f"marginal log likelihood is {log_p_x}")
            if save is True:
                print("model saved")
                self.save_NN_model(EPOCHS)
                self.save_training_info(numpy_dicts=[train_history, test_history],
                                        single_value_dict={"log_p_x" : log_p_x,
                                                           "test ELBO" : -test_history['loss'][-1],
                                                           "train ELBO" : -train_history['loss'][-1],
                                                           "EPOCHS MAX": EPOCHS,
                                                           "EPOCHS Actual" : EPOCH+1,
                                                           "lr_decay":lr_decay,
                                                           "early_stopping": early_stopping,
                                     "early_stopping_criterion":early_stopping_criterion,
                                     "validation_based_decay":validation_based_decay})
            return train_history, test_history, log_p_x
        else:
            self.save_training_info(numpy_dicts=[train_history, test_history],
                                    single_value_dict={})
            return train_history


if __name__ == "__main__":
    from Utils.load_binirised_mnist import load_data
    train_loader, test_loader = load_data(100)
    data = next(iter(train_loader))[0]
    print(data.shape)
    vae = VAE(latent_dim=32, n_IAF_steps=2, h_dim=20,
              IAF_node_width=450, encoder_fc_dim=450,
              decoder_fc_dim=450, constant_sigma=True)
    #vae.model(data)
    # vae.VAE_model.encoder.IAF_steps[0].FinalLayer.state_dict() # useful for checking sigma constant
    vae.train(EPOCHS = 2, train_loader=train_loader, save=True, test_loader=test_loader)
