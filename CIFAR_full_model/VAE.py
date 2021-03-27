import torch
from datetime import datetime
from CIFAR_base_class.BaseClass import CIFAR_BASE
from CIFAR_full_model.model import VAE_ladder_model

class CIFAR_VAE_fancy(CIFAR_BASE):
    def __init__(self, n_rungs=4, lambda_free_bits=0.25, with_IAF=True, IAF_n_hidden_layers=1, name=""):
        super(CIFAR_VAE_fancy, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = VAE_ladder_model(n_rungs=n_rungs, lambda_free_bits=lambda_free_bits, with_IAF=with_IAF,
                                      IAF_n_hidden_layers=IAF_n_hidden_layers).to(self.device)
        self.optimizer = torch.optim.Adamax(self.model.parameters())
        current_time = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
        self.save_NN_path = f"Results_and_trained_models/CIFAR_full_model/" \
                            f"{name}n_rungs_{n_rungs}__/{current_time}/"

if __name__ == '__main__':
    from Utils.load_CIFAR import load_data
    train_loader, test_loader = load_data(100)
    test_model = CIFAR_VAE_fancy()
    print(test_model.get_bits_per_dim(test_loader, n_samples=1))
    train_history, test_history, bits_per_dim = test_model.train(1, train_loader, test_loader, save=False,
              lr_decay=True, validation_based_decay=True, early_stopping=True,
              early_stopping_criterion=40)
