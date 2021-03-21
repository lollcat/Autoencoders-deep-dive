
experiment_dicts_paper = [{"latent_dim": 32, "n_IAF_steps": 0},  # 0
    {"latent_dim": 32, "n_IAF_steps": 2, "IAF_node_width" : 320},  # 1
    {"latent_dim": 32, "n_IAF_steps" : 2, "IAF_node_width": 1920},  # 2
    {"latent_dim": 32, "n_IAF_steps" : 4, "IAF_node_width": 1920},  # 3
    {"latent_dim": 32, "n_IAF_steps" : 8,"IAF_node_width": 1920}]  # 4

experiment_dicts_no_sigma = [{"constant_sigma": True, "latent_dim": 32, "n_IAF_steps": 2, "IAF_node_width" : 320},  # 1
    {"constant_sigma": True, "latent_dim": 32, "n_IAF_steps" : 2, "IAF_node_width": 1920},  # 2
    {"constant_sigma": True, "latent_dim": 32, "n_IAF_steps" : 4, "IAF_node_width": 1920},  # 3
    {"constant_sigma": True, "latent_dim": 32, "n_IAF_steps" : 8,"IAF_node_width": 1920}]  # 4

experiment_dicts = [{"latent_dim": 32, "n_IAF_steps": 0},
    {"latent_dim": 32, "n_IAF_steps": 2, "IAF_node_width" : 320},
    {"latent_dim": 32, "n_IAF_steps" : 4, "IAF_node_width": 320},
    {"latent_dim": 32, "n_IAF_steps" : 8,"IAF_node_width": 320}]


if __name__ == '__main__':
    # test
    from IAF_VAE_mnist.VAE import VAE
    vae_test = VAE(**experiment_dicts[1])