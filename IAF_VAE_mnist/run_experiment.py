from Utils.load_binirised_mnist import load_data
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
import numpy as np
from IAF_VAE_mnist.VAE import VAE
import torch
from datetime import datetime
import pathlib, os
import time
import pandas as pd


def run_experiment(vae_kwargs, epochs=2000, batch_size=32):
    current_time = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    name = ""
    for key in vae_kwargs:
        name += f"{key}_{vae_kwargs[key]}__"
    name += f"{current_time}"
    fig_path = f"IAF_VAE_mnist/Experiment_results/{name}/"
    save = True
    use_GPU = True
    train_loader, test_loader = load_data(batch_size=batch_size)

    vae = VAE(**vae_kwargs)
    start_time = time.time()
    train_history, test_history, p_x = vae.train(EPOCHS=epochs, train_loader=train_loader, test_loader=test_loader)
    run_time = time.time() - start_time
    print(f"runtime for training (with marginal estimation) is {round(run_time/3600, 2)} hours")

    pathlib.Path(os.path.join(os.getcwd(), fig_path)).mkdir(parents=True, exist_ok=True)
    if use_GPU is True:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = "cpu"
    train_df = pd.DataFrame(train_history)
    train_df.to_csv(f"{fig_path}_train_df.csv")
    test_df = pd.DataFrame(test_history)
    test_df.to_csv(f"{fig_path}_test_df.csv")

    with open(f"{fig_path}_final_results", "w") as g:
        g.write("\n".join([f"marginal likelihood: {p_x} \n",
                           f"test ELBO:     {-test_history['loss'][-1]}",
                           f"train ELBO:     {-train_history['loss'][-1]}",
                           f"runtime: {round(run_time/3600, 2)} hours"]))

    figure, axs = plt.subplots(len(train_history), 1, figsize=(6, 10))
    for i, key in enumerate(train_history):
        axs[i].plot(train_history[key])
        axs[i].plot(test_history[key])
        axs[i].legend([key + " train", key + " test"])
        # if i == 0:
        #    axs[i].set_yscale("log")
    plt.tight_layout()
    if save is True:
        plt.savefig(f"{fig_path}train_test_info.png")
    plt.show()

    n = 5
    data_chunk = next(iter(train_loader))[0][0:n ** 2, :, :, :]
    fig, axs = plt.subplots(n, n)
    for i in range(n * n):
        row = int(i / n)
        col = i % n
        axs[row, col].imshow(np.moveaxis(data_chunk[i, :, :, :].detach().numpy(), source=0, destination=-1),
                             cmap="gray")
        axs[row, col].axis('off')
    if save is True:
        plt.savefig(f"{fig_path}original.png")
    plt.show()

    n = 5
    prediction = vae.get_reconstruction(data_chunk.to(device))
    fig, axs = plt.subplots(n, n)
    for i in range(n * n):
        row = int(i / n)
        col = i % n
        axs[row, col].imshow(np.squeeze(prediction[i, :, :, :]), cmap="gray")
        axs[row, col].axis('off')
    if save is True:
        plt.savefig(f"{fig_path}reconstruction.png")
    plt.show()

if __name__ == '__main__':
    # python -m IAF_VAE_mnist.run_experiment # to run in command line
    from IAF_VAE_mnist.Experiment_dicts import experiment_dicts
    # How many epoch?
    for i, experiment_dict in enumerate(experiment_dicts):
        print(f"running experiment {experiment_dict}")
        run_experiment(experiment_dict, epochs=2000)
        print(f"\n experiment {i} complete \n\n\n")
