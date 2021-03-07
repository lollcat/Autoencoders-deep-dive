# Questions
 - Doesn't the different dimensions of the prior completely change the "KL regularisation term"? I suppose we are trying to maximise the marginal probability p(x) and using the ELBO to do this, making everying "fair" as the marginal
 
 # TODO
  - marginal p(x) calculation. I suppose VLB is mean so p(x) should also be?
  - could rewrite normalisation layer
  
# Notes
    - issues of stability in encoder (specifically the autoregressive section and exp(log_stds) when generating the initial samples
    - layer norm inside the autoregressive section will break the autoregressiveness
    
    