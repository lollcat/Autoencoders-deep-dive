if __name__ == "__main__":
    # tensorboard --logdir logs
    import tensorflow as tf
    #tf.config.run_functions_eagerly(True)
    from Variational_Autoencoder.VAE import VAE
    import matplotlib.pyplot as plt
    import datetime

    full_cov = False
    use_tensorboard = False
    binarized_data = True
    latent_representation_dim = 3
    EPOCHS = 3
    name = f"binarized={binarized_data}__latent_representation_dim={latent_representation_dim}"
    if binarized_data is True:
        from Utils.load_binarized_mnist import x_train, x_test, train_ds, test_ds, image_dim
    else:
        from Utils.load_plain_mnist import x_train, x_test, train_ds, test_ds, image_dim

    # Define vae
    vae = VAE(latent_representation_dim, image_dim, full_cov=full_cov)

    samples = x_test[0:9, :, :, :][:, :, :]
    example_reconstruction_hist = [vae(samples)[0].numpy()]

    if use_tensorboard:
        # Tensorboard writer
        logdir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_file_writer = tf.summary.create_file_writer(logdir + name +  "/train/")
        test_file_writer = tf.summary.create_file_writer(logdir + name + "/test")


    train_history = []
    test_history = []
    if use_tensorboard:
        step_counter = 0
        tf.summary.experimental.set_step(step_counter)
    for epoch in range(EPOCHS):
        total_train_loss = 0
        for images in train_ds:
            ELBO, log_prob_x_given_z_decode_batch, log_probs_z_given_x_batch, log_prob_z_prior_batch = vae.train_step(images)
            total_train_loss -= ELBO
            if use_tensorboard:
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
            ELBO, log_prob_x_given_z_decode_batch, log_probs_z_given_x_batch, log_prob_z_prior_batch = vae.test_step(test_images)
            total_test_loss -= ELBO
            if use_tensorboard:
                with test_file_writer.as_default():
                    tf.summary.scalar("ELBO", ELBO)
                    tf.summary.scalar("log_prob_x_given_z_decode", log_prob_x_given_z_decode_batch)
                    tf.summary.scalar("log_probs_z_given_x", log_probs_z_given_x_batch)
                    tf.summary.scalar("log_probs_z_given_x", log_probs_z_given_x_batch)
                    tf.summary.scalar("log_prob_z_prior", log_prob_z_prior_batch)

        train_history.append(total_train_loss / len(train_ds))
        test_history.append(total_test_loss / len(test_ds))
        example_reconstruction_hist.append(vae(samples)[0].numpy())

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
    reconstruction = vae(x_test[0:n * n, :, :])[0]
    reconstruction = tf.nn.sigmoid(reconstruction)
    fig, axs = plt.subplots(n, n)
    for i in range(n * n):
        row = int(i / n)
        col = i % n
        axs[row, col].imshow(reconstruction[i, :, :], cmap="gray")
        axs[row, col].axis('off')
    fig.tight_layout()
    plt.show()

