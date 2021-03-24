
raise Exception("for basic CIFAR we run the same architecture as mnist")

"""
latent_dim = 32
experiment_dicts_paper = [{"latent_dim": latent_dim, "n_IAF_steps": 0},  # 0
    {"latent_dim": latent_dim, "n_IAF_steps": 2, "IAF_node_width" : 320},  # 1
    {"latent_dim": latent_dim, "n_IAF_steps" : 2, "IAF_node_width": 1920},  # 2
    {"latent_dim": latent_dim, "n_IAF_steps" : 4, "IAF_node_width": 1920},  # 3
    {"latent_dim": latent_dim, "n_IAF_steps" : 8,"IAF_node_width": 1920}]  # 4

experiment_dicts_no_sigma = [{"constant_sigma": True, "latent_dim": latent_dim, "n_IAF_steps": 2, "IAF_node_width" : 320},  # 1
    {"constant_sigma": True, "latent_dim": latent_dim, "n_IAF_steps" : 2, "IAF_node_width": 1920},  # 2
    {"constant_sigma": True, "latent_dim": latent_dim, "n_IAF_steps" : 4, "IAF_node_width": 1920},  # 3
    {"constant_sigma": True, "latent_dim": latent_dim, "n_IAF_steps" : 8,"IAF_node_width": 1920}]  # 4
"""

