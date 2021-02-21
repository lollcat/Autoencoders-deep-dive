if __name__ == "__main__":
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if len(gpus) > 0:
        # Restrict TensorFlow to only use the first GPU
        try:
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)
    else:
        print("running without GPU")
    #tf.config.run_functions_eagerly(True)
    from Utils.load_plain_mnist import x_train, x_test, train_ds, test_ds, image_dim, y_train
    from Inverse_Autoregressive_Flow.IAF_VAE import IAF_VAE
    import matplotlib.pyplot as plt
    import datetime

    import numpy as np
    import matplotlib as mpl

    x_train = x_train[0:4, :, :, :]
    y_train = y_train[0:4]

    x_train_4_points= np.repeat(x_train, 5000, axis=0)
    train_ds = tf.data.Dataset.from_tensor_slices(x_train_4_points).shuffle(10000).batch(256)
    latent_representation_dim = 2
    EPOCHS = 400
    name = ""
    # Define vae
    IAF_vae = IAF_VAE(latent_representation_dim, x_dim=image_dim,
                     n_autoregressive_units=6, autoregressive_unit_layer_width=200,
                     First_Encoder_to_IAF_step_dim=200,
                     encoder_FC_layer_nodes=200)

    def running_mean(new_point, running_mean, i):
        return running_mean + (new_point - running_mean)/(i + 1)

    train_history = []
    ELBO_component_history = np.zeros((EPOCHS, 3))
    test_history = []
    step_counter = 0
    n_train_batches = len(list(train_ds))
    for epoch in range(EPOCHS):
        total_train_loss = 0
        for i, images in enumerate(train_ds):
            ELBO, log_prob_x_given_z_decode_batch, log_probs_z_given_x_batch, log_prob_z_prior_batch = IAF_vae.train_step(images)
            total_train_loss -= ELBO
            ELBO_component_history[epoch, 0] =  running_mean(log_prob_x_given_z_decode_batch.numpy(), ELBO_component_history[epoch, 0], i)
            ELBO_component_history[epoch, 1] =  running_mean(log_probs_z_given_x_batch.numpy(), ELBO_component_history[epoch, 1], i)
            ELBO_component_history[epoch, 2] =  running_mean(log_prob_z_prior_batch.numpy(), ELBO_component_history[epoch, 2], i)

        train_history.append(total_train_loss / n_train_batches)

        print(
            f'Epoch {epoch + 1}, '
            f'\n Loss: {total_train_loss.numpy() / n_train_batches}')

    plt.figure()
    plt.plot(train_history)
    plt.plot(test_history)
    plt.legend("ELBO")
    plt.show()

    plt.figure()
    plt.plot(ELBO_component_history[:, 1:])
    plt.legend([ "log_probs_z_given_x_batch", "log_prob_z_prior_batch"])

    plt.figure()
    plt.plot(ELBO_component_history[:, 0])
    plt.legend(["log_prob_x_given_z"])

    plt.figure()
    n_points = 4
    cols = cmap=mpl.cm.rainbow(np.linspace(0.1, 0.9, n_points))
    points = []
    for point_n in range(0, n_points):
        point_repeat = np.zeros((5000, 28, 28, 1))
        point_repeat[: :, :, :] = x_train[point_n, :, :]
        encoding_2D = IAF_vae.get_encoding(point_repeat)
        plt.scatter(encoding_2D[:, 0], encoding_2D[:, 1], color=cols[point_n], s=1, )
    plt.show()

