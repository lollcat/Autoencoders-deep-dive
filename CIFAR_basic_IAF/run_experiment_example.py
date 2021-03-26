from CIFAR_basic_IAF.VAE import VAE
from Utils.load_CIFAR import load_data
from IAF_VAE_mnist.Experiment_dicts import experiment_dicts_paper # we use the same basic archetecture that we used for mnist
from Utils.CIFAR_plotting import plot_original_and_reconstruction


batch_size = 100
EPOCHS = 100
train_loader, test_loader = load_data(batch_size)
experiment_dict = experiment_dicts_paper[3]
print(experiment_dict)
vae = VAE(**experiment_dict)
train_history, test_history, bits_per_dim = vae.train(EPOCHS, train_loader, test_loader, save=True,
              lr_decay=True, validation_based_decay = True, early_stopping=True,
              early_stopping_criterion=40)