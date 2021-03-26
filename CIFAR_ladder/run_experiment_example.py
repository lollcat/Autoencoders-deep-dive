from CIFAR_ladder.VAE import VAE_ladder as VAE
from Utils.load_CIFAR import load_data
from CIFAR_ladder.Experiment_dicts import experiment_dicts

batch_size = 100
EPOCHS = 100
train_loader, test_loader = load_data(batch_size)
experiment_dict = experiment_dicts[3]
print(experiment_dict)
vae = VAE(**experiment_dict)
train_history, test_history, bits_per_dim = vae.train(EPOCHS, train_loader, test_loader, save=True,
              lr_decay=True, validation_based_decay = True, early_stopping=True,
              early_stopping_criterion=40)