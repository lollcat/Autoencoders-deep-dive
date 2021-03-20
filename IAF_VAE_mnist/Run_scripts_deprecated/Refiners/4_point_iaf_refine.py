import pathlib, os, sys
if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())
from Utils.load_4_point import load_data, x_train_4_points
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
import numpy as np
from IAF_VAE_mnist.VAE import VAE
import torch
from datetime import datetime

import time
import pandas as pd


def run_experiment(vae_kwargs, epochs=100, batch_size=256, experiment_name="", save_model=True, lr_schedule=False,
                              save_info_during_training=True):
    current_time = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    if experiment_name != "":
        assert experiment_name[-1] == "/"
    name = experiment_name
    for key in vae_kwargs:
        name += f"{key}_{vae_kwargs[key]}__"
    name += f"{current_time}"
    train_loader= load_data(batch_size=batch_size)
    vae = VAE(**vae_kwargs)
    model_path = "Results_and_trained_models/IAF_VAE_mnist/saved_models/latent_dim_2__n_IAF_steps_8__h_dim_200_IAF_node_width_320/2021_03_18-07_47_26_AM/epochs_2000__model"
    vae.load_NN_model(model_path)
    start_time = time.time()
    train_history = vae.train(EPOCHS=epochs, train_loader=train_loader, test_loader=None, save_model=save_model,
                              lr_schedule=lr_schedule,
                              save_info_during_training=save_info_during_training)
    run_time = time.time() - start_time
    print(f"runtime for training (with marginal estimation) is {round(run_time/3600, 2)} hours")

if __name__ == '__main__':
    # python -m IAF_VAE_mnist.experiment_4_point # to run in command line
    experiment_name = "IAF" + "4_point_refine" + "/"
    print(f"running experiment {experiment_name}")
    n_epoch = 2000
    experiment_dict = {"latent_dim": 2, "n_IAF_steps": 8, "IAF_node_width" : 320}
    print(f"running 4 point with config {experiment_dict} for {n_epoch} epoch")
    run_experiment(experiment_dict, epochs=n_epoch, experiment_name=experiment_name,
                   save_info_during_training=True, save_model=True, lr_schedule=True)
