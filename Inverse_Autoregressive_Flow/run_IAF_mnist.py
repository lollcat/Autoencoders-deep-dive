if __name__ == "__main__":
    # tensorboard --logdir logs
    import tensorflow as tf
    #tf.config.run_functions_eagerly(True)

    from Inverse_Autoregressive_Flow.IAF_VAE import IAF_VAE
    import matplotlib.pyplot as plt
    import datetime
    use_tensorboard = False
    binarized_data = True
    latent_representation_dim = 16
    EPOCHS = 15
    name = f"binarized={binarized_data}__latent_representation_dim={latent_representation_dim}"
    if binarized_data is True:
        from Utils.load_binarized_mnist import x_train, x_test, train_ds, test_ds, image_dim
    else:
        from Utils.load_plain_mnist import x_train, x_test, train_ds, test_ds, image_dim


    # Define vae
    IAF_vae = IAF_VAE(latent_representation_dim, x_dim=image_dim,
                 n_autoregressive_units=3, autoregressive_unit_layer_width=64,
                 First_Encoder_to_IAF_step_dim=64,
                 encoder_FC_layer_nodes=64,
                 decoder_layer_width = 64)

    if use_tensorboard is True:
        # Tensorboard writer
        logdir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_file_writer = tf.summary.create_file_writer(logdir + name +  "/train/")
        test_file_writer = tf.summary.create_file_writer(logdir + name + "/test")

    train_history = []
    test_history = []
    step_counter = 0
    if use_tensorboard is True:
        tf.summary.experimental.set_step(step_counter)
    for epoch in range(EPOCHS):
        total_train_loss = 0
        for images in train_ds:
            ELBO, log_prob_x_given_z_decode_batch, log_probs_z_given_x_batch, log_prob_z_prior_batch = IAF_vae.train_step(images)
            total_train_loss -= ELBO
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
        for test_images in test_ds:
            ELBO, log_prob_x_given_z_decode_batch, log_probs_z_given_x_batch, log_prob_z_prior_batch = IAF_vae.test_step(test_images)
            total_test_loss -= ELBO
            if use_tensorboard is True:
                with test_file_writer.as_default():
                    tf.summary.scalar("ELBO", ELBO)
                    tf.summary.scalar("log_prob_x_given_z_decode", log_prob_x_given_z_decode_batch)
                    tf.summary.scalar("log_probs_z_given_x", log_probs_z_given_x_batch)
                    tf.summary.scalar("log_probs_z_given_x", log_probs_z_given_x_batch)
                    tf.summary.scalar("log_prob_z_prior", log_prob_z_prior_batch)

        train_history.append(total_train_loss / len(train_ds))
        test_history.append(total_test_loss / len(test_ds))

        print(
            f'Epoch {epoch + 1}, '
            f'\n Loss: {total_train_loss.numpy() / len(train_ds)}, '
            f'\n Test Loss: {total_test_loss.numpy() / len(test_ds)}')

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



