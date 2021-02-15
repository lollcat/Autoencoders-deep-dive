# Questions
 - Doesn't the different dimensions of the prior completely change the "KL regularisation term"? I suppose we are trying to maximise the marginal probability p(x) and using the ELBO to do this, making everying "fair" as the marginal
 
 # TODO
  - implement everything exactly like the paper 
  - rewrite baseline guassian VAE to get good score
  - not using weight normalisation (currently doing batch norm and layer norm)
  - 4 point plot is looking strange (maybe a result of dropout regularisation?)
  - marginal p(x) calculation. I suppose VLB is mean so p(x) should also be?
  - why is test loss lower than train loss? (I think it is because of training=False)
  - why are we getting an error for latent dimension 32? seems like overfit
  
# Notes
    - issues of stability in encoder (specifically the autoregressive section and exp(log_stds) when generating the initial samples
    - layer norm inside the autoregressive section will break the autoregressiveness
    
    