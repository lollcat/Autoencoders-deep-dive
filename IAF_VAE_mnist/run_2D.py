from IAF_VAE_mnist.run_experiment import run_experiment

if __name__ == '__main__':
    # python -m IAF_VAE_mnist.run_2D # to run in command line
    settings = {"latent_dim": 2, "n_IAF_steps": 4, "IAF_node_width" : 320}
    epochs = 500
    experiment_name = "2D_1/"
    run_experiment(settings, epochs=epochs, experiment_name=experiment_name )

