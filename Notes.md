# Issues
 - Perform experiment on effects of sigma for mnist
 - Resnet decoder works with python3.6, pytorch 1.6.0 from https://github.com/UNSWComputing/pytorch/releases/tag/v1.6.0
 - https://paperswithcode.com/sota/image-generation-on-binarized-mnist
 
# Code to write
- either warmup or lambda max thing for ladder
 
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
