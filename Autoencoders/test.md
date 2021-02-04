{% raw %}
$$
\begin{aligned}
\log p_{\theta}(\mathrm{x})=& \mathbb{E_{q_{\phi}(\mathbf{z} \mid \mathbf{x})}}\left[\log p_{\theta}(\mathrm{x})\right] \\
=& \mathbb{E_{q_{\phi}(\mathbf{z} \mid \mathbf{x})}}\left[\log \left[\frac{p_{\theta}(\mathbf{x}, \mathbf{z})}{p_{\theta}(\mathbf{z} \mid \mathbf{x})}\right]\right] \\
=& \mathbb{E_{q_{\phi}(\mathbf{z} \mid \mathbf{x})}}\left[\log \left[\frac{p_{\theta}(\mathbf{x}, \mathbf{z})}{q_{\phi}(\mathbf{z} \mid \mathbf{x})} \frac{q_{\phi}(\mathbf{z} \mid \mathbf{x})}{p_{\theta}(\mathbf{z} \mid \mathbf{x})}\right]\right] \\
=& \underbrace{\mathbb{E_{q_{\phi}(\mathbf{z} \mid \mathbf{x})}}\left[\log \left[\frac{p_{\theta}(\mathbf{x}, \mathbf{z})}{q_{\phi}(\mathbf{z} \mid \mathbf{x})}\right]\right]}_{=\mathcal{L}_{\theta, \phi}(\mathbf{x}) = (\mathrm{ELBO})}+\underbrace{\mathbb{E_{q_{\phi}(\mathbf{z} \mid \mathbf{x})}}\left[\log \left[\frac{q_{\phi}(\mathbf{z} \mid \mathbf{x})}{p_{\theta}(\mathbf{z} \mid \mathbf{x})}\right]\right]}_{=D_{K L}\left(q_{\phi}(\mathbf{z} \mid \mathbf{x})|| p_{\boldsymbol{\theta}}(\mathbf{z} \mid \mathbf{x})\right)} \\

&
\end{aligned}
$$
{% endraw %}
