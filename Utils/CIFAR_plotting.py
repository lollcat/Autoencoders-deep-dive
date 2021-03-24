import matplotlib.pyplot as plt
import numpy as np

def plot_original_and_reconstruction(vae, train_loader, n = 4):
    data_chunk = next(iter(train_loader))[0][0:n ** 2, :, :, :]
    fig_original, axs_original = plt.subplots(n, n)
    fig_reconstruct, axs_reconstruct = plt.subplots(n, n)
    prediction = vae.get_reconstruction(data_chunk.to(vae.device))
    for i in range(n * n):
        row = int(i / n)
        col = i % n
        axs_original[row, col].imshow(np.moveaxis(data_chunk[i, :, :, :].detach().numpy(), source=0, destination=-1),
                             cmap="gray")
        axs_original[row, col].axis('off')
        axs_reconstruct[row, col].imshow(np.moveaxis(prediction[i, :, :, :], source=0, destination=-1), cmap="gray")
        axs_reconstruct[row, col].axis('off')
    return fig_original, axs_original, fig_reconstruct, axs_reconstruct

