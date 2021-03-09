experiment_dicts = [{"latent_dim": 32, "n_IAF_steps": 0},
    {"latent_dim": 32, "n_IAF_steps": 2, "IAF_node_width" : 320},
    {"latent_dim": 32, "n_IAF_steps" : 2, "IAF_node_width": 1920},
    {"latent_dim": 32, "n_IAF_steps" : 4, "IAF_node_width": 1920},
    {"latent_dim": 32, "n_IAF_steps" : 8,"IAF_node_width": 1920}]

if __name__ == '__main__':
    # test
    from IAF_VAE_mnist.VAE import VAE
    vae_test = VAE(**experiment_dicts[1])