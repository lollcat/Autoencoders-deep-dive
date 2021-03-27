import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_original_and_reconstruction(vae, test_loader, n = 4):
    torch.manual_seed(0)
    data_chunk = next(iter(test_loader))[0][0:n ** 2, :, :, :]
    # plot original
    fig_original = plt.figure(figsize=(5., 5.))
    from mpl_toolkits.axes_grid1 import ImageGrid
    grid = ImageGrid(fig_original, 111,  # similar to subplot(111)
                     nrows_ncols=(n, n),  # creates 2x2 grid of axes
                     axes_pad=0.1,  # pad between axes in inch.
                     )
    for i, ax in enumerate(grid):
        ax.imshow(np.moveaxis(data_chunk[i, :, :, :].detach().numpy(), source=0, destination=-1))
        ax.axis('off')


    prediction = vae.get_reconstruction(data_chunk.to(vae.device))
    fig_reconstruction = plt.figure(figsize=(5., 5.))
    from mpl_toolkits.axes_grid1 import ImageGrid
    grid = ImageGrid(fig_reconstruction, 111,  # similar to subplot(111)
                     nrows_ncols=(n, n),  # creates 2x2 grid of axes
                     axes_pad=0.1,  # pad between axes in inch.
                     )
    for i, ax in enumerate(grid):
        ax.imshow(np.moveaxis(prediction[i, :, :, :], source=0, destination=-1), cmap="gray")
        ax.axis('off')

    return fig_original, fig_reconstruction


