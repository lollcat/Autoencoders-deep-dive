import tensorflow as tf

from Utils.load_binarized_mnist import x_train, x_test, train_ds, test_ds, image_dim
from Inverse_Autoregressive_Flow.IAF_VAE import IAF_VAE
from Inverse_Autoregressive_Flow.run_experiment import run_experiment
from datetime import datetime
import pathlib, os

if __name__ == "__main__":
    #tf.config.experimental_run_functions_eagerly(True)
    """CONFIG ARGS"""
    decay_lr = True
    lr_anneal_period = 300
    latent_representation_dim = 32
    EPOCHS = 100
    current_time = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    name = current_time + f"eager_ex"
    fig_path = f"Inverse_Autoregressive_Flow/Figures/{name}/"

    pathlib.Path(os.path.join(os.getcwd(), fig_path)).mkdir(parents=True, exist_ok=True)
    # Define vae

    IAF_vae = IAF_VAE(latent_representation_dim=latent_representation_dim,
                      x_dim=image_dim,
                       n_autoregressive_units=2,
                      autoregressive_unit_layer_width=450,
                      First_Encoder_to_IAF_step_dim=200,
                      encoder_FC_layer_nodes=450)

    run_experiment(fig_path, IAF_vae, train_ds, test_ds, x_test, EPOCHS, latent_dim=2,
                   decay_lr=decay_lr,
                   lr_anneal_period=lr_anneal_period)




