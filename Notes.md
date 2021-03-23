# Issues
 - Perform experiment on effects of sigma for mnist
 - Resnet decoder works with python3.6, pytorch 1.6.0 from https://github.com/UNSWComputing/pytorch/releases/tag/v1.6.0
 - https://paperswithcode.com/sota/image-generation-on-binarized-mnist
 
# Code to write
- numerically stable version of marginal for ladder
- ladder make sure free bits thing is working
- less glaringly important: numerically stable version for mnist
 
# tests to do
 - does sigma have an effect (mnist and CIFAR)
 - does ladder network help?
 
# order of things to run
## Slow cluster
MNIST standard experiment (we aren't relying on this for next steps)

## Fast cluster
- MNIST
    - test adamW optimizer
- CIFAR basic 
    - maybe run 100 steps with different latent dim first to check
- CIFAR ladder
- CIFAR constant sigma

## Learning rate
    - if to big we get Nans, default adamax val of 0.002 works, 0.01 breaks things
