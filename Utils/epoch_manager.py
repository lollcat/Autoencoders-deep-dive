import numpy as np

class EpochManager:
    def __init__(self, optimizer, EPOCHS, lr_decay=True, early_stopping=True, early_stopping_criterion=20,
                 validation_based_decay = True):
        self.optimizer = optimizer
        self.EPOCHs = EPOCHS
        self.lr_decay = lr_decay
        self.early_stopping = early_stopping
        self.early_stopping_criterion = early_stopping_criterion
        self.validation_based_decay = validation_based_decay
        self.max_decay_steps = 5
        if validation_based_decay:
            self.n_decay_steps_counter = 0
            self.decay_step_criterion = int(early_stopping_criterion/2)
            self.lr_decay_factor = 0.1
        else:
            self.epoch_per_decay = max(int(EPOCHS/self.max_decay_steps) + 1, 1)
            self.lr_decay_factor = 0.5


    def manage(self, EPOCH, test_history):
        test_history = np.array(test_history)
        halt_training = False
        if self.early_stopping and EPOCH > self.early_stopping_criterion + 1:
            if np.all(test_history[-self.early_stopping_criterion:] > test_history[-self.early_stopping_criterion - 1]):
                print(f"early stopping due to {self.early_stopping_criterion} steps without improvement")
                halt_training = True

        if self.lr_decay is True and EPOCH > 0:  # use max lr for first 50 epoch no matter what
            if self.validation_based_decay is True:
                if EPOCH % self.decay_step_criterion:
                    if EPOCH > self.decay_step_criterion: # just to make sure we don't get a slicing error
                        if np.all(test_history[-self.decay_step_criterion:] > test_history[-self.decay_step_criterion - 1]):
                            self.optimizer.param_groups[0]["lr"] *= self.lr_decay_factor
                            self.n_decay_steps_counter += 1
                            print(f"lr decay due to {self.decay_step_criterion} steps without improvement")
                            print(f"learning rate decayed to {self.optimizer.param_groups[0]['lr']}")
                            print(f"{self.n_decay_steps_counter} decay steps out of max {self.max_decay_steps}")
                            if self.n_decay_steps_counter > self.max_decay_steps:
                                print("stopping because maximum number of decay steps reached")
                                halt_training = True
            else: # fixed number of decay steps
                if EPOCH % self.epoch_per_decay == 0:
                    self.optimizer.param_groups[0]["lr"] *= self.lr_decay_factor
                    print(f"learning rate decayed to {self.optimizer.param_groups[0]['lr']}")
        return halt_training


if __name__ == '__main__':
    import torch
    from IAF_VAE_mnist.VAE import VAE
    vae = VAE()
    EPOCH = 100
    epoch_manager = EpochManager(vae.optimizer, 200)
    test_history = np.arange(100)
    halt_training = epoch_manager.manage(EPOCH, -test_history)
    assert halt_training is False
    halt_training = epoch_manager.manage(EPOCH, test_history)
    assert halt_training is True
