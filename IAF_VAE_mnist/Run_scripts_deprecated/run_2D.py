import pathlib, os, sys
if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())

from IAF_VAE_mnist.run_experiment import run_experiment

if __name__ == '__main__':
    settings = {"latent_dim": 2, "n_IAF_steps": 4, "IAF_node_width" : 400}
    epochs = 500
    experiment_name = "2D_but_full_dataset/"
    run_experiment(settings, epochs=epochs, experiment_name=experiment_name)

