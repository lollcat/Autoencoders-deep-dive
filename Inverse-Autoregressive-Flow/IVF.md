# Inverse Autoregressive Flow

Source: [Improved Variational Inference with Inverse Autoregressive Flow](https://arxiv.org/abs/1606.04934)

Summary: Inverse autoregressive flow allow us to express flexible, rich, high dimensional latent variable posterior distributions  using variational inference. 

## Normalising Flows
Normalising flows create rich posterior distributions by starting with an initially simple distribution \\( p(\mathbf{z_{0}} \mid \mathbf{x})\\)   (e.g. diagonal covariance Gaussian) and repeatedly transforming it via a set of parameterised functions, such that the final result \\( \mathbf{z_T} \\) is a flexible distribution. 
{% raw %} 
\[
\mathbf{z_{0}} \sim q\left(\mathbf{z_{0}} \mid \mathbf{x}\right), \quad \mathbf{z_{t}}=\mathbf{f_{t}}\left(\mathbf{z_{t-1}}, \mathbf{x}\right) \quad \forall t=1 \ldots T
\]
{% endraw %} 

For variational inference to work, we need to be able to obtain samples from the final distribution, as well as the samples probability \\( p(\mathbf{z_T}) \\). The above equation tells us how to generate samples. To calculate the probability density function of the final iterate, we use the following: The resultant distribution from a change of variables is given by,
{% raw %} 
\[
p_{y}(y)=p_{x}(x)\left|\frac{\mathrm{d} x}{\mathrm{~d} y}\right|
\]
{% endraw %} 
Thus if we can compute the Jacobian determinant from the transormation, we are able to compute the probabiltiy density function. We express this using the log probability function: 
{% raw %} 
\[
\log q\left(\mathbf{z}_{T} \mid \mathbf{x}\right)=\log q\left(\mathbf{z}_{0} \mid \mathbf{x}\right)-\sum_{t=1}^{T} \log \operatorname{det}\left|\frac{d \mathbf{z}_{t}}{d \mathbf{z}_{t-1}}\right| \quad \quad \text{where, } \epsilon \sim \mathcal{N}(0, I)
\]
{% endraw %} 


## Inverse Autoregressive Flow
Inverse Autoregressive Flow utilises the transformation where we begin the chain with
{% raw %} 
\[
\mathbf{z}_{0}=\boldsymbol{\mu}_{0}+\boldsymbol{\sigma}_{0} \odot \boldsymbol{\epsilon}
\]
{% endraw %} 
