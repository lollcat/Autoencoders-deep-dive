if __name__ == "__main__":

    # tensorboard --logdir logs
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

    from Inverse_Autoregressive_Flow.IAF_VAE import IAF_VAE
    from Utils.running_mean import running_mean
    import matplotlib.pyplot as plt
    import datetime
    import numpy as np

    """CONFIG ARGS"""
    use_tensorboard = False
    binarized_data = True
    decay_lr = False
    latent_representation_dim = 8
    EPOCHS = 200

    if binarized_data is True:
        from Utils.load_binarized_mnist import x_train, x_test, train_ds, test_ds, image_dim
    else:
        from Utils.load_plain_mnist import x_train, x_test, train_ds, test_ds, image_dim


    # Define vae

    IAF_vae = IAF_VAE(latent_representation_dim, x_dim=image_dim,
                 n_autoregressive_units=2, autoregressive_unit_layer_width=320,
                 First_Encoder_to_IAF_step_dim=450,
                 encoder_FC_layer_nodes=450)
    """
    IAF_vae = IAF_VAE(latent_representation_dim, x_dim=image_dim,
                 n_autoregressive_units=2, autoregressive_unit_layer_width=32,
                 First_Encoder_to_IAF_step_dim=32,
                 encoder_FC_layer_nodes=32)
    """
    name = f"binarized={binarized_data}__latent_representation_dim={latent_representation_dim}"

    if use_tensorboard is True:
        # Tensorboard writer
        logdir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_file_writer = tf.summary.create_file_writer(logdir + name +  "/train/")
        test_file_writer = tf.summary.create_file_writer(logdir + name + "/test")

    train_history = []
    test_history = []
    step_counter = 0
    if decay_lr is True:
        previous_total_test_loss = 1e12  # initialise to a big number
    n_train_batches = len(list(train_ds))
    n_test_batches = len(list(test_ds))
    ELBO_component_history_train = np.zeros((EPOCHS, 3))
    ELBO_component_history_test = np.zeros((EPOCHS, 3))
    if use_tensorboard is True:
        tf.summary.experimental.set_step(step_counter)
    for epoch in range(EPOCHS):
        total_train_loss = 0
        for i, images in enumerate(train_ds):
            ELBO, log_prob_x_given_z_decode_batch, log_probs_z_given_x_batch, log_prob_z_prior_batch = IAF_vae.train_step(images)
            total_train_loss -= ELBO
            ELBO_component_history_train[epoch, 0] = running_mean(log_prob_x_given_z_decode_batch.numpy(),
                                                                  ELBO_component_history_train[epoch, 0], i)
            ELBO_component_history_train[epoch, 1] = running_mean(log_probs_z_given_x_batch.numpy(),
                                                                  ELBO_component_history_train[epoch, 1], i)
            ELBO_component_history_train[epoch, 2] = running_mean(log_prob_z_prior_batch.numpy(),
                                                                  ELBO_component_history_train[epoch, 2], i)
            if use_tensorboard is True:
                with train_file_writer.as_default():
                    tf.summary.scalar("ELBO", ELBO)
                    tf.summary.scalar("log_prob_x_given_z_decode", log_prob_x_given_z_decode_batch)
                    tf.summary.scalar("log_probs_z_given_x", log_probs_z_given_x_batch)
                    tf.summary.scalar("log_probs_z_given_x", log_probs_z_given_x_batch)
                    tf.summary.scalar("log_prob_z_prior", log_prob_z_prior_batch)
                step_counter += 1
                tf.summary.experimental.set_step(step_counter)

        total_test_loss = 0
        for i, test_images in enumerate(test_ds):
            ELBO, log_prob_x_given_z_decode_batch, log_probs_z_given_x_batch, log_prob_z_prior_batch = IAF_vae.test_step(test_images)
            total_test_loss -= ELBO
            ELBO_component_history_test[epoch, 0] = running_mean(log_prob_x_given_z_decode_batch.numpy(),
                                                                  ELBO_component_history_test[epoch, 0], i)
            ELBO_component_history_test[epoch, 1] = running_mean(log_probs_z_given_x_batch.numpy(),
                                                                  ELBO_component_history_test[epoch, 1], i)
            ELBO_component_history_test[epoch, 2] = running_mean(log_prob_z_prior_batch.numpy(),
                                                                  ELBO_component_history_test[epoch, 2], i)
        if decay_lr is True:
            if total_test_loss > previous_total_test_loss:
                IAF_vae.optimizer.lr = 0.5 * IAF_vae.optimizer.lr # half learning rate if validation loss increases
            previous_total_test_loss = total_test_loss

        if use_tensorboard is True:
            with test_file_writer.as_default():
                tf.summary.scalar("ELBO", ELBO)
                tf.summary.scalar("log_prob_x_given_z_decode", log_prob_x_given_z_decode_batch)
                tf.summary.scalar("log_probs_z_given_x", log_probs_z_given_x_batch)
                tf.summary.scalar("log_probs_z_given_x", log_probs_z_given_x_batch)
                tf.summary.scalar("log_prob_z_prior", log_prob_z_prior_batch)

        train_history.append(total_train_loss / n_train_batches)
        test_history.append(total_test_loss / n_test_batches)

        print(
            f'Epoch {epoch + 1}, '
            f'\n Loss: {total_train_loss.numpy() / n_train_batches}, '
            f'\n Test Loss: {total_test_loss.numpy() / n_test_batches}')
        if epoch % int(EPOCHS/10) == 0:
            print(f"marginal likelihood of data is {IAF_vae.get_marginal_likelihood(x_test, n_x_data_samples=5)}")

    print(f"marginal likelihood of data is {IAF_vae.get_marginal_likelihood(x_test)}")
    fig, axs = plt.subplots(1, 2)
    axs[0].plot(train_history)
    axs[1].plot(test_history)
    plt.show()

    n = 5
    fig, axs = plt.subplots(n, n)
    for i in range(n * n):
        row = int(i / n)
        col = i % n
        axs[row, col].imshow(x_test[i, :, :], cmap="gray")
        axs[row, col].axis('off')
    plt.show()

    # train
    plt.figure()
    plt.plot(ELBO_component_history_train[:, 1:])
    plt.legend(["log_probs_z_given_x_batch train", "log_prob_z_prior_batch train"])
    plt.show()
    plt.figure()
    plt.plot(ELBO_component_history_train[:, 0])
    plt.legend("log prob x given z train")
    plt.show()

    # test
    plt.figure()
    plt.plot(ELBO_component_history_test[:, 1:])
    plt.legend(["log_probs_z_given_x_batch test", "log_prob_z_prior_batch test"])
    plt.show()
    plt.figure()
    plt.plot(ELBO_component_history_test[:, 0])
    plt.legend("log prob x given z test")
    plt.show()

    # plot reconstruction
    n = 5
    reconstruction = IAF_vae(x_test[0:n * n, :, :])[0]
    reconstruction = tf.nn.sigmoid(reconstruction)
    fig, axs = plt.subplots(n, n)
    for i in range(n * n):
        row = int(i / n)
        col = i % n
        axs[row, col].imshow(reconstruction[i, :, :], cmap="gray")
        axs[row, col].axis('off')
    fig.tight_layout()
    plt.show()





