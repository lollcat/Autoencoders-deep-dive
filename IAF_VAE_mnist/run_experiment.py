import pathlib, os, sys
if not os.getcwd() in sys.path: # need this check for running without pycharm
    sys.path.append(os.getcwd())
from Utils.load_binirised_mnist import load_data
from IAF_VAE_mnist.VAE import VAE
from datetime import datetime
import time
import pandas as pd


def run_experiment(vae_kwargs, epochs=2000, batch_size=100, experiment_name="", save=True, lr_decay=True):
    """
    :param vae_kwargs:
    :param epochs:
    :param batch_size:
    :param experiment_name:
    :param save: save model and training data
    :return:
    """
    current_time = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    if experiment_name != "":
        if experiment_name[-1] == "/": # must end in / for use as folder
            experiment_name += "/"
    name = experiment_name
    vae_setting = ""
    for key in vae_kwargs:
        vae_setting += f"{key}_{vae_kwargs[key]}__"
    name += vae_setting +  f"{current_time}"
    results_path = f"Results_and_trained_models/IAF_VAE_mnist/Experiment_results/{name}/"
    train_loader, test_loader = load_data(batch_size=batch_size)
    vae = VAE(**vae_kwargs)
    start_time = time.time()
    train_history, test_history, p_x = vae.train(EPOCHS=epochs, train_loader=train_loader, test_loader=test_loader,
                                                 save_model=save, lr_decay=lr_decay)
    run_time = time.time() - start_time
    print(f"runtime for training (with marginal estimation) is {round(run_time/3600, 2)} hours")
    if save:
        pathlib.Path(os.path.join(os.getcwd(), results_path)).mkdir(parents=True, exist_ok=True)
        train_df = pd.DataFrame(train_history)
        train_df.to_csv(f"{results_path}_train_df.csv")
        test_df = pd.DataFrame(test_history)
        test_df.to_csv(f"{results_path}_test_df.csv")

        with open(f"{results_path}_final_results", "w") as g:
            g.write("\n".join([f"{vae_setting} \n\n"
                               f"n_lr_cycles={n_lr_cycles}, lr_schedule={lr_decay} \n\n"
                               f"marginal likelihood: {p_x}",
                               f"test ELBO:     {-test_history['loss'][-1]}",
                               f"train ELBO:     {-train_history['loss'][-1]}",
                               f"runtime: {round(run_time/3600, 2)} hours",
                               f"trained for : {epochs} EPOCH"]))
    return vae, train_history, test_history, p_x


if __name__ == '__main__':
    """
    import sys
    experiment_number = int(sys.argv[1])
    print(f"experiment argument {experiment_number}")
    dict_number = int(sys.argv[2])
    print(f"dict argument {dict_number}")
        if experiment_number == 0:
        from IAF_VAE_mnist.Experiment_dicts import experiment_dicts_paper as experiment_dicts
        experiment_name = "paper_table_replication/"
    elif experiment_number == 1:
        from IAF_VAE_mnist.Experiment_dicts import experiment_dicts_no_sigma as experiment_dicts
        experiment_name = "no_sigma/"
    else:
        raise Exception("experiment incorrectly specified")
    """
    experiment_number = 1
    epoch = 2000
    experiment_dict = {"constant_sigma": True, "latent_dim": 5, "n_IAF_steps": 2, "IAF_node_width" : 10}
    experiment_name = "constant_sigma/"
    print(f"running experiment {experiment_name}, {experiment_dict} for {epoch} epoch")
    vae = run_experiment(experiment_dict, epochs=epoch, experiment_name=experiment_name,
                         save=False, n_lr_cycles=1, lr_decay=True)
    print(f"\n completed experiment {experiment_name}, {experiment_dict} for {epoch} epoch")
