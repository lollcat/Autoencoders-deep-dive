import tensorflow as tf
from Utils.load_plain_mnist import x_train, x_test, train_ds, test_ds, image_dim, y_train
from Inverse_Autoregressive_Flow.IAF_VAE import IAF_VAE
from datetime import datetime
from Inverse_Autoregressive_Flow.run_experiment import run_experiment
import numpy as np
import pathlib, os
x_train = x_train[0:4, :, :, :]
y_train = y_train[0:4]
x_train_4_points= np.repeat(x_train, 5000, axis=0)
train_ds = tf.data.Dataset.from_tensor_slices(x_train_4_points).shuffle(10000).batch(256)


if __name__ == "__main__":
    latent_representation_dim = 2
    EPOCHS = 100
    decay_lr = True
    lr_anneal_period = 300
    current_time = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    name = current_time + f"latent_representation_dim={latent_representation_dim}__" + "weight norm"
    fig_path = f"Inverse_Autoregressive_Flow/Figures/4_point/{name}/"
    pathlib.Path(os.path.join(os.getcwd(), fig_path)).mkdir(parents=True, exist_ok=True)

    # Define vae
    IAF_vae = IAF_VAE(latent_representation_dim=latent_representation_dim,
                      x_dim=image_dim,
                     n_autoregressive_units=6,
                     autoregressive_unit_layer_width=200,
                     First_Encoder_to_IAF_step_dim=200,
                     encoder_FC_layer_nodes=200)
    run_experiment(fig_path, IAF_vae, train_ds, test_ds, x_test, EPOCHS, decay_lr=decay_lr,
                   lr_anneal_period=lr_anneal_period, n_points_latent_vis=4)


