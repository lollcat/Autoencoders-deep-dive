latent_dim = 32
experiment_dicts_paper = [{"latent_dim": latent_dim, "n_IAF_steps": 0, "n_rungs":4},  # 0
    {"latent_dim": latent_dim, "n_IAF_steps": 1, "IAF_node_width" : 320, "n_rungs":4},  # 1
    {"latent_dim": latent_dim, "n_IAF_steps": 0, "n_rungs":8},  # 3
    {"latent_dim": latent_dim, "n_IAF_steps": 1, "IAF_node_width" : 320, "n_rungs":8}, # 4
    {"latent_dim": latent_dim, "n_IAF_steps": 0, "n_rungs": 12},  # 5
    {"latent_dim": latent_dim, "n_IAF_steps": 1, "IAF_node_width": 320, "n_rungs": 12} # 6
]  # 4


"""
experiment_dicts_no_sigma = [{"constant_sigma": True, "latent_dim": latent_dim, "n_IAF_steps": 2, "IAF_node_width" : 320},  # 1
    {"constant_sigma": True, "latent_dim": latent_dim, "n_IAF_steps" : 2, "IAF_node_width": 1920},  # 2
    {"constant_sigma": True, "latent_dim": latent_dim, "n_IAF_steps" : 4, "IAF_node_width": 1920},  # 3
    {"constant_sigma": True, "latent_dim": latent_dim, "n_IAF_steps" : 8,"IAF_node_width": 1920}]  # 4
"""
