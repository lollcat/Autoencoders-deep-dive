figure, axs = plt.subplots(len(train_history), 1, figsize=(6, 10))
for i, key in enumerate(train_history):
    axs[i].plot(train_history[key])
    axs[i].plot(test_history[key])
    axs[i].legend([key + " train", key + " test"])
    # if i == 0:
    #    axs[i].set_yscale("log")
plt.tight_layout()
if save is True:
    plt.savefig(f"{results_path}train_test_info.png")
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
    plt.savefig(f"{results_path}original.png")
plt.show()

n = 5
prediction = vae.get_reconstruction(data_chunk.to(vae.device))
fig, axs = plt.subplots(n, n)
for i in range(n * n):
    row = int(i / n)
    col = i % n
    axs[row, col].imshow(np.moveaxis(prediction[i, :, :, :], source=0, destination=-1), cmap="gray")
    axs[row, col].axis('off')
plt.show()