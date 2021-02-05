if __name__ == "__main__":
    # tensorboard --logdir logs
    #import tensorflow as tf
    #tf.config.run_functions_eagerly(True)
    from load_data import x_train, x_test, train_ds, test_ds, image_dim
    from VAE import VAE
    import tensorflow as tf
    import matplotlib.pyplot as plt
    import datetime

    """CONGI PARAMS"""
    name = "independent normal"
    latent_representation_dim = 32
    EPOCHS = 15

    # Define vae
    vae = VAE(latent_representation_dim, image_dim)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    loss_objective = tf.keras.losses.BinaryCrossentropy()
    train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    vae.compile(optimizer, loss_objective)

    samples = x_test[0:9, :, :, :][:, :, :]
    example_reconstruction_hist = [vae(samples).numpy()]

    # Tensorboard writer
    logdir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_file_writer = tf.summary.create_file_writer(logdir + name +  "/train/")
    test_file_writer = tf.summary.create_file_writer(logdir + name + "/test")


    train_history = []
    test_history = []
    step_counter = 0
    tf.summary.experimental.set_step(step_counter)
    for epoch in range(EPOCHS):
        total_train_loss = 0
        for images in train_ds:
            ELBO, kl_loss, reconstruction_loss = vae.train_step(images)
            total_train_loss += ELBO
            with train_file_writer.as_default():
                tf.summary.scalar("ELBO", ELBO)
                tf.summary.scalar("KL divergence", kl_loss)
                tf.summary.scalar("Reconstruction loss", reconstruction_loss)
                tf.summary.scalar("mean variance", vae.metrics[0].result())
            step_counter += 1
            tf.summary.experimental.set_step(step_counter)

        total_test_loss = 0
        for test_images in test_ds:
            ELBO_test, kl_loss_test, reconstruction_loss_test = vae.test_step(test_images)
            total_test_loss += ELBO_test
            with test_file_writer.as_default():
                tf.summary.scalar("ELBO", ELBO_test)
                tf.summary.scalar("KL divergence", kl_loss_test)
                tf.summary.scalar("Reconstruction loss", reconstruction_loss_test)

        train_history.append(total_train_loss / len(train_ds))
        test_history.append(total_test_loss / len(test_ds))
        example_reconstruction_hist.append(vae(samples).numpy())

        print(
            f'Epoch {epoch + 1}, '
            f'\n Loss: {total_train_loss.numpy() / len(train_ds)}, '
            f'\n Test Loss: {total_test_loss.numpy() / len(test_ds)}')

    fig, axs = plt.subplots(1, 2)
    axs[0].plot(train_history)
    axs[1].plot(test_history)
    plt.show()

    n = -1
    fig, axs = plt.subplots(3, 3)
    for i in range(9):
        row = int(i / 3)
        col = i % 3
        axs[row, col].imshow(example_reconstruction_hist[n][i, :, :], cmap="gray")
    plt.show()

