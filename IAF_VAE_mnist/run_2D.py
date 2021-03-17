import pathlib, os, sys
if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())

from IAF_VAE_mnist.run_experiment import run_experiment

if __name__ == '__main__':
    # python -m IAF_VAE_mnist.run_2D # to run in command line
    settings = {"latent_dim": 2, "n_IAF_steps": 6, "IAF_node_width" : 320}
    epochs = 1000
    experiment_name = "2D_1_azure/"
    run_experiment(settings, epochs=epochs, experiment_name=experiment_name)

