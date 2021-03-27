# Notes
 - Resnet decoder works with python3.6, pytorch 1.6.0 from https://github.com/UNSWComputing/pytorch/releases/tag/v1.6.0
 - https://paperswithcode.com/sota/image-generation-on-binarized-mnist
 
# Code to write
- numerically stable version for mnist

# Experiments to conduct
    - dont use sigma for the mean of CIFAR
 





## Learning rate
    - if to big we get Nans, default adamax val of 0.002 works, 0.01 breaks things
