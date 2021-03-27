import torch
from datetime import datetime
from CIFAR_ladder.VAE import VAE_ladder
from CIFAR_full_model.model import VAE_ladder_model

class CIFAR_VAE_fancy(VAE_ladder):
    def __init__(self, latent_dim=32, n_rungs=4, n_IAF_steps=1, IAF_node_width=450, use_GPU = True, name="",
                 constant_sigma=False, lambda_free_bits=0.25):
        super(CIFAR_VAE_fancy, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = VAE_ladder_model()
        self.optimizer = torch.optim.Adamax(self.model.parameters())
        current_time = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
        self.save_NN_path = f"Results_and_trained_models/CIFAR_full_model/" \
                            f"{name}n_rungs_{n_rungs}__/{current_time}/"

if __name__ == '__main__':
    from Utils.load_CIFAR import load_data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = load_data(100)
    x_data = next(iter(train_loader))[0].to(device)
    experiment_dict = {"latent_dim": 3, "n_IAF_steps": 1, "IAF_node_width": 10, "n_rungs": 4}
    test_model = VAE_ladder(**experiment_dict)
    print(test_model.get_bits_per_dim(test_loader, n_samples=3))
    train_history, test_history, bits_per_dim = test_model.train(2, train_loader, test_loader, save=False,
              lr_decay=True, validation_based_decay=True, early_stopping=True,
              early_stopping_criterion=40)
