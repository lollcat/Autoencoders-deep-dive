if __name__ == '__main__':
    from IAF_VAE_mnist.VAE import VAE
    from IAF_VAE_mnist.Experiment_dicts import experiment_dicts_paper
    from Utils.load_binirised_mnist import load_data
    from Utils.mnist_plotting import plot_train_test, plot_original_and_reconstruction

    experiment_dict = experiment_dicts_paper[2]
    print(experiment_dict)

    epochs = 100
    batch_size = 64
    train_loader, test_loader = load_data(batch_size)
    vae = VAE(**experiment_dict)
    train_history, test_history, log_p_x = vae.train(epochs, train_loader, test_loader, save=True,
                  lr_decay=True, validation_based_decay = True, early_stopping=True,
                  early_stopping_criterion=40)
    print(f"results have been saved to {vae.save_NN_path} in the parent directory to the code directory")

    import matplotlib.pyplot as plt
    figure, axs = plot_train_test(train_history, test_history)
    plot_original_and_reconstruction()
    plt.show()
