import pathlib, os, sys
if not os.getcwd() in sys.path: # need this check for running without pycharm
    sys.path.append(os.getcwd())


from Utils.load_binirised_mnist import load_data
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
import numpy as np
from IAF_VAE_mnist.VAE import VAE
import torch
from datetime import datetime
import time
import pandas as pd


def run_experiment(vae_kwargs, epochs=2000, batch_size=256, experiment_name="", save_model=True,
                   save_figures=True):
    current_time = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    if experiment_name != "":
        assert experiment_name[-1] == "/"
    name = experiment_name
    for key in vae_kwargs:
        name += f"{key}_{vae_kwargs[key]}__"
    name += f"{current_time}"
    results_path = f"Results_and_trained_models/IAF_VAE_mnist/Experiment_results/{name}/"
    save = save_figures
    use_GPU = True
    train_loader, test_loader = load_data(batch_size=batch_size)

    vae = VAE(**vae_kwargs)
    start_time = time.time()
    train_history, test_history, p_x = vae.train(EPOCHS=epochs, train_loader=train_loader, test_loader=test_loader,
                                                 save_model=save_model)
    run_time = time.time() - start_time
    print(f"runtime for training (with marginal estimation) is {round(run_time/3600, 2)} hours")

    pathlib.Path(os.path.join(os.getcwd(), results_path)).mkdir(parents=True, exist_ok=True)
    if use_GPU is True:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = "cpu"
    train_df = pd.DataFrame(train_history)
    train_df.to_csv(f"{results_path}_train_df.csv")
    test_df = pd.DataFrame(test_history)
    test_df.to_csv(f"{results_path}_test_df.csv")

    with open(f"{results_path}_final_results", "w") as g:
        g.write("\n".join([f"marginal likelihood: {p_x} \n",
                           f"test ELBO:     {-test_history['loss'][-1]}",
                           f"train ELBO:     {-train_history['loss'][-1]}",
                           f"runtime: {round(run_time/3600, 2)} hours",
                           f"trained for : {epochs} EPOCH"]))

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
    prediction = vae.get_reconstruction(data_chunk.to(device))
    fig, axs = plt.subplots(n, n)
    for i in range(n * n):
        row = int(i / n)
        col = i % n
        axs[row, col].imshow(np.squeeze(prediction[i, :, :, :]), cmap="gray")
        axs[row, col].axis('off')
    if save is True:
        plt.savefig(f"{results_path}reconstruction.png")
    plt.show()


if __name__ == '__main__':
    # python -m IAF_VAE_mnist.run_experiment # to run in command line
    from IAF_VAE_mnist.Experiment_dicts import experiment_dicts
    experiment_name = "new_decoder_decay/"
    epoch = 2000
    #for i, experiment_dict in enumerate(experiment_dicts):
    i = 1; experiment_dict = experiment_dicts[i]
    print(f"running experiment {experiment_name}, number {i} {experiment_dict} for {epoch} epoch")
    run_experiment(experiment_dict, epochs=epoch, experiment_name=experiment_name)
    print(f"\n experiment {i} complete \n\n\n")
