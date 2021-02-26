from Utils.load_mnist_pytorch import load_data
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from IAF_VAE.VAE import VAE
import torch
from datetime import datetime
import pathlib, os


if __name__ == "__main__":
    current_time = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    name = current_time + f"latent_2" + "with_resnet"
    fig_path = f"IAF_VAE/Figures/{name}/"
    pathlib.Path(os.path.join(os.getcwd(), fig_path)).mkdir(parents=True, exist_ok=True)

    train_loader, test_loader = load_data(256)
    vae = VAE(latent_dim=2, encoder_fc_dim=10, decoder_fc_dim=10, n_IAF_steps=2)
    #vae = VAE(latent_dim=2, encoder_fc_dim=120, decoder_fc_dim=120, n_IAF_steps=2)
    train_history, test_history = vae.train(EPOCHS=20, train_loader=train_loader, test_loader=test_loader)

    figure, axs = plt.subplots(len(train_history), 1)
    for i, key in enumerate(train_history):
        axs[i].plot(train_history[key])
        axs[i].plot(test_history[key])
        axs[i].legend([key + " train", key + " test"])
        if i == 0:
            axs[i].set_yscale("log")
    plt.savefig(f"{fig_path}train_test_info.png")
    plt.show()

    n = 5
    data_chunk = next(iter(train_loader))[0][0:n ** 2, :, :, :]
    fig, axs = plt.subplots(n, n)
    for i in range(n * n):
        row = int(i / n)
        col = i % n
        axs[row, col].imshow(np.squeeze(data_chunk[i, :, :, :]), cmap="gray")
        axs[row, col].axis('off')
    plt.savefig(f"{fig_path}original.png")
    plt.show()

    n = 5
    prediction = vae.get_reconstruction(data_chunk)
    fig, axs = plt.subplots(n, n)
    for i in range(n * n):
        row = int(i / n)
        col = i % n
        axs[row, col].imshow(np.squeeze(prediction[i, :, :, :]), cmap="gray")
        axs[row, col].axis('off')
    plt.savefig(f"{fig_path}reconstruction.png")
    plt.show()

    n_points_latent_vis = 10
    cols = mpl.cm.rainbow(np.linspace(0.1, 0.9, n_points_latent_vis))
    points = []
    for point_n in range(n_points_latent_vis):
        point_repeat = np.zeros((500, 1, 28, 28))
        point_repeat[:, :, :, :] = data_chunk[point_n, :, :, :]
        encoding_2D = vae.get_latent_encoding(torch.tensor(point_repeat, dtype=torch.float32))
        plt.scatter(encoding_2D[:, 0], encoding_2D[:, 1], color=cols[point_n], s=1, )
    plt.savefig(f"{fig_path}visualise_latent_space.png")
    plt.show()


