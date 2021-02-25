import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def running_mean(new_point, running_mean, i):
    return running_mean + (new_point - running_mean) / (i + 1)

def run_experiment(fig_path, IAF_vae, train_ds, test_ds, x_test, EPOCHS, decay_lr = False, lr_anneal_period=100,
                   n_points_latent_vis=100
                   ):
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

    train_history = []
    test_history = []
    n_train_batches = len(list(train_ds))
    n_test_batches = len(list(test_ds))
    ELBO_component_history_train = np.zeros((EPOCHS, 3))
    ELBO_component_history_test = np.zeros((EPOCHS, 3))
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
            if epoch % lr_anneal_period == 0 and epoch > 10:
                IAF_vae.optimizer.lr = IAF_vae.optimizer.lr * 0.5
                print("lowering lr by half")

        train_history.append(total_train_loss / n_train_batches)
        test_history.append(total_test_loss / n_test_batches)

        print(
            f'Epoch {epoch + 1}, '
            f'\n Loss: {total_train_loss.numpy() / n_train_batches}, '
            f'\n Test Loss: {total_test_loss.numpy() / n_test_batches}')
        if epoch % int(EPOCHS/10 + 1) == 0:
            print(f"marginal likelihood of data is {IAF_vae.get_marginal_likelihood(x_test, n_x_data_samples=5)}")

    print(f"marginal likelihood of data is {IAF_vae.get_marginal_likelihood(x_test)}")
    fig, axs = plt.subplots(1, 2)
    axs[0].plot(train_history)
    axs[1].plot(test_history)
    plt.show()
    plt.savefig(f"{fig_path}train_test_hist.png")

    n = 5
    fig, axs = plt.subplots(n, n)
    for i in range(n * n):
        row = int(i / n)
        col = i % n
        axs[row, col].imshow(x_test[i, :, :], cmap="gray")
        axs[row, col].axis('off')
    plt.savefig(f"{fig_path}test_imgs.png")
    plt.show()


    # train
    plt.figure()
    plt.plot(ELBO_component_history_train[:, 1:])
    plt.legend(["log_probs_z_given_x_batch train", "log_prob_z_prior_batch train"])
    plt.savefig(f"{fig_path}train_ELBO1.png")
    plt.show()

    plt.figure()
    plt.plot(ELBO_component_history_train[:, 0])
    plt.legend(["log prob x given z train"])
    plt.savefig(f"{fig_path}train_ELBO2.png")
    plt.show()

    # test
    plt.figure()
    plt.plot(ELBO_component_history_test[:, 1:])
    plt.legend(["log_probs_z_given_x_batch test", "log_prob_z_prior_batch test"])
    plt.savefig(f"{fig_path}test_ELBO1.png")
    plt.show()

    plt.figure()
    plt.plot(ELBO_component_history_test[:, 0])
    plt.legend(["log prob x given z train"])
    plt.savefig(f"{fig_path}test_ELBO2.png")
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
    plt.savefig(f"{fig_path}reconstruction.png")
    plt.show()

    cols = mpl.cm.rainbow(np.linspace(0.1, 0.9, n_points_latent_vis))
    points = []
    for point_n in range(n_points_latent_vis):
        point_repeat = np.zeros((500, 28, 28, 1))
        point_repeat[::, :, :] = x_test[point_n, :, :]
        encoding_2D = IAF_vae.get_encoding(point_repeat)
        plt.scatter(encoding_2D[:, 0], encoding_2D[:, 1], color=cols[point_n], s=1, )
    plt.savefig(f"{fig_path}visualise_latent_space.png")
    plt.show()