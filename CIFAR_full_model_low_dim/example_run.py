if __name__ == '__main__':
    from CIFAR_full_model.VAE import CIFAR_VAE_fancy
    from Utils.load_CIFAR import load_data
    train_loader, test_loader = load_data(100)
    test_model = CIFAR_VAE_fancy(n_rungs=3)
    train_history, test_history, bits_per_dim = test_model.train(1, train_loader, test_loader, save=False,
                                                                 lr_decay=True, validation_based_decay=True,
                                                                 early_stopping=True,
                                                                 early_stopping_criterion=40)
