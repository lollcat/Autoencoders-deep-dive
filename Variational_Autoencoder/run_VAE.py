if __name__ == "__main__":
    # tensorboard --logdir logs
    import tensorflow as tf
    #tf.config.run_functions_eagerly(True)
    from Variational_Autoencoder.VAE import VAE
    import matplotlib.pyplot as plt
    from datetime import datetime

    full_cov = False
    use_tensorboard = False
    binarized_data = True
    latent_representation_dim = 32
    EPOCHS = 3
    current_time = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    name = current_time + f"binarized={binarized_data}__latent_representation_dim={latent_representation_dim}"
    if binarized_data is True:
        from Utils.load_binarized_mnist import x_train, x_test, train_ds, test_ds, image_dim
    else:
        from Utils.load_plain_mnist import x_train, x_test, train_ds, test_ds, image_dim

    # Define vae
    vae = VAE(latent_representation_dim, image_dim, full_cov=full_cov)

    samples = x_test[0:9, :, :, :][:, :, :]
    example_reconstruction_hist = [vae(samples)[0].numpy()]


    train_history = []
    test_history = []
    for epoch in range(EPOCHS):
        total_train_loss = 0
        for images in train_ds:
            ELBO, log_prob_x_given_z_decode_batch, log_probs_z_given_x_batch, log_prob_z_prior_batch = vae.train_step(images)
            total_train_loss -= ELBO
        total_test_loss = 0
        for test_images in test_ds:
            ELBO, log_prob_x_given_z_decode_batch, log_probs_z_given_x_batch, log_prob_z_prior_batch = vae.test_step(test_images)
            total_test_loss -= ELBO

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

