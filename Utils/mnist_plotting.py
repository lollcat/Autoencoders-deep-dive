import matplotlib.pyplot as plt
import matplotlib as mpl
#mpl.rcParams['figure.dpi'] = 300
import numpy as np

def plot_train_test(train_history, test_history):
    figure, axs = plt.subplots(len(train_history), 1, figsize=(6, 10))
    for i, key in enumerate(train_history):
        axs[i].plot(train_history[key])
        axs[i].plot(test_history[key])
        axs[i].legend([key + " train", key + " test"])
    plt.tight_layout()
    return figure, axs

def plot_original_and_reconstruction(vae, train_loader):
    n = 5
    data_chunk = next(iter(train_loader))[0][0:n ** 2, :, :, :]
    prediction = vae.get_reconstruction(data_chunk.to(vae.device))
    fig_original, axs_original = plt.subplots(n, n)
    fig_reconstruct, axs_reconstruct = plt.subplots(n, n)
    for i in range(n * n):
        row = int(i / n)
        col = i % n
        axs_original[row, col].imshow(np.moveaxis(data_chunk[i, :, :, :].detach().numpy(), source=0, destination=-1),
                             cmap="gray")
        axs_original[row, col].axis('off')
        # reconstruction
        axs_reconstruct[row, col].imshow(np.squeeze(prediction[i, :, :, :]), cmap="gray")
        axs_reconstruct[row, col].axis('off')
    return fig_original, axs_original, fig_reconstruct, axs_reconstruct

def plot_4_point_overfit(vae, cols=None, n_scatter_points=10000):
    from Utils.load_4_point import x_train_4_points
    import torch
    data_chunk = torch.tensor(x_train_4_points)
    if cols is None:
        cols = mpl.cm.rainbow(np.linspace(0.1, 0.9, 4))
    fig, ax = plt.subplots(figsize=(5,5))
    n_points_latent_vis = 4
    for point_n in range(n_points_latent_vis):
        point_repeat = np.zeros((n_scatter_points, 1, 28, 28))
        point_repeat[:, :, :, :] = data_chunk[point_n, :, :, :]
        encoding_2D = vae.get_latent_encoding(torch.tensor(point_repeat, dtype=torch.float32).to(vae.device))
        ax.scatter(encoding_2D[:, 0], encoding_2D[:, 1], color=cols[point_n, :], s=2, alpha=0.8)
    ax.set_xlabel(r"$z_1$")
    ax.set_ylabel(r"$z_2$")
    lim = 3
    ax.set_ylim((-lim, lim))
    ax.set_xlim((-lim, lim))
    ax.set_axisbelow(True)
    ax.grid()
    return fig, ax

def plot_prior(c = np.array([200, 200, 200]) / 255, alpha=1):
    lim=3
    fig, ax = plt.subplots(figsize=(5, 5))
    prior = np.random.normal(size=(20000, 2))
    ax.scatter(prior[:, 0], prior[:, 1], c=c, s=2, alpha=alpha)
    ax.set_xlabel(r"$z_1$")
    ax.set_ylabel(r"$z_2$")
    ax.set_ylim((-lim, lim))
    ax.set_xlim((-lim, lim))
    ax.set_axisbelow(True)
    ax.grid()
    return fig, ax

def plot_latent_space(vae):
    import torch
    n = 10
    half_range = 2
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    # We will sample n points within [-15, 15] standard deviations
    grid_x = np.linspace(-half_range, half_range, n)
    grid_y = np.linspace(-half_range, half_range, n)
    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sample = torch.tensor(np.array([[xi, yi]])).float().to(vae.device)
            digit = torch.sigmoid(vae.model.decoder(z_sample)).cpu().detach().numpy()
            figure[i * digit_size: (i + 1) * digit_size,
            j * digit_size: (j + 1) * digit_size] = digit
    fig, ax = plt.subplots(figsize=(10, 10))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    ax.set_xlabel(r"$z_1$", fontsize=15)
    ax.set_ylabel(r"$z_2$", fontsize=15)
    ax.imshow(figure, cmap="gray")
    return fig, ax