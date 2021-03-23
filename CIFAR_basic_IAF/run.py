from Utils.load_CIFAR import load_data
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
import numpy as np
from CIFAR_basic_IAF.VAE import VAE
import torch
from datetime import datetime
import pathlib, os


if __name__ == "__main__":
    current_time = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    name = current_time + f"first_run"
    fig_path = f"CIFAR_basic_IAF/Figures/{name}/"
    save = True
    use_GPU = True
    if use_GPU is True:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = "cpu"

    train_loader, test_loader = load_data(32)
    latent_dim = 32
    vae = VAE(latent_dim=latent_dim, n_IAF_steps=2,
              h_dim=32, IAF_node_width=64, encoder_fc_dim=64, decoder_fc_dim=64,
              use_GPU=False)
    #vae = VAE(latent_dim=latent_dim, n_IAF_steps=8,
    #          h_dim=200, IAF_node_width=1920, encoder_fc_dim=450, decoder_fc_dim=450,
    #          use_GPU = True)
    train_history, test_history = vae.train(EPOCHS=100, train_loader=train_loader, test_loader=test_loader)

    pathlib.Path(os.path.join(os.getcwd(), fig_path)).mkdir(parents=True, exist_ok=True)
    figure, axs = plt.subplots(len(train_history), 1, figsize=(6, 10))
    for i, key in enumerate(train_history):
        axs[i].plot(train_history[key])
        axs[i].plot(test_history[key])
        axs[i].legend([key + " train", key + " test"])
        #if i == 0:
        #    axs[i].set_yscale("log")
    plt.tight_layout()
    if save is True:
        plt.savefig(f"{fig_path}train_test_info.png")
    plt.show()

    n = 5
    data_chunk = next(iter(train_loader))[0][0:n**2, :, :, :]
    fig, axs = plt.subplots(n, n)
    for i in range(n * n):
        row = int(i / n)
        col = i % n
        axs[row, col].imshow(np.moveaxis(data_chunk[i, :, :, :].detach().numpy(), source=0, destination=-1), cmap="gray")
        axs[row, col].axis('off')
    if save is True:
        plt.savefig(f"{fig_path}original.png")
    plt.show()

    n = 5
    prediction = vae.model(data_chunk)[0].detach().numpy()
    fig, axs = plt.subplots(n, n)
    for i in range(n * n):
        row = int(i / n)
        col = i % n
        axs[row, col].imshow(np.moveaxis(prediction[i, :, :, :], source=0, destination=-1), cmap="gray")
        axs[row, col].axis('off')
    if save is True:
        plt.savefig(f"{fig_path}reconstruction.png")
    plt.show()

    if latent_dim == 2:
        n_points_latent_vis = 10
        cols = mpl.cm.rainbow(np.linspace(0.1, 0.9, n_points_latent_vis))
        points = []
        for point_n in range(n_points_latent_vis):
            point_repeat = np.zeros((500, 1, 28, 28))
            point_repeat[:, :, :, :] = data_chunk[point_n, :, :, :]
            encoding_2D = vae.get_latent_encoding(torch.tensor(point_repeat, dtype=torch.float32).to(device))
            plt.scatter(encoding_2D[:, 0], encoding_2D[:, 1], color=cols[point_n], s=1, )
        if save is True:
            plt.savefig(f"{fig_path}visualise_latent_space.png")
        plt.show()


