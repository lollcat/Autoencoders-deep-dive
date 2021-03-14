from Utils.load_4_point import load_data, x_train_4_points
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


def run_experiment(vae_kwargs, epochs=100, batch_size=64):
    current_time = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    name = ""
    for key in vae_kwargs:
        name += f"{key}_{vae_kwargs[key]}__"
    name += f"{current_time}"
    fig_path = f"IAF_VAE_mnist/4_point/{name}/"
    save = True
    use_GPU = True
    train_loader= load_data(batch_size=batch_size)

    vae = VAE(**vae_kwargs)
    start_time = time.time()
    train_history = vae.train(EPOCHS=epochs, train_loader=train_loader)
    run_time = time.time() - start_time
    print(f"runtime for training (with marginal estimation) is {round(run_time/3600, 2)} hours")

    pathlib.Path(os.path.join(os.getcwd(), fig_path)).mkdir(parents=True, exist_ok=True)
    if use_GPU is True:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = "cpu"
    train_df = pd.DataFrame(train_history)
    train_df.to_csv(f"{fig_path}_train_df.csv")

    with open(f"{fig_path}_final_results", "w") as g:
        g.write("\n".join([f"train ELBO:     {-train_history['loss'][-1]}",
                           f"runtime: {round(run_time/3600, 2)} hours"]))

    figure, axs = plt.subplots(len(train_history), 1, figsize=(6, 10))
    for i, key in enumerate(train_history):
        axs[i].plot(train_history[key])
        axs[i].legend([key + " train"])
    plt.tight_layout()
    if save is True:
        plt.savefig(f"{fig_path}train_test_info.png")
    plt.show()

    n = 2
    data_chunk = torch.tensor(x_train_4_points)
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

    n = 2
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

    n_points_latent_vis = 4
    cols = mpl.cm.rainbow(np.linspace(0.1, 0.9, n_points_latent_vis))
    plt.figure()
    for point_n in range(n_points_latent_vis):
        point_repeat = np.zeros((1000, 1, 28, 28))
        point_repeat[:, :, :, :] = data_chunk[point_n, :, :, :]
        encoding_2D = vae.get_latent_encoding(torch.tensor(point_repeat, dtype=torch.float32).to(device))
        plt.scatter(encoding_2D[:, 0], encoding_2D[:, 1], color=cols[point_n], s=1, )
    plt.xlabel(r"$z_1$")
    plt.ylabel(r"$z_2$")
    if save is True:
        plt.savefig(f"{fig_path}visualise_latent_space.png")
    plt.show()


if __name__ == '__main__':
    # python -m IAF_VAE_mnist.4_point # to run in command line
    n_epoch = 1000
    experiment_dict = {"latent_dim": 2, "n_IAF_steps": 4, "IAF_node_width" : 320}
    print(f"running 4 point with config {experiment_dict} for {n_epoch} epoch")
    run_experiment(experiment_dict, epochs=n_epoch)
