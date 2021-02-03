# Autoencoders intro
## What are autoencoders?
The goal of an autoencoder is to learn useful latent representations of data. Autoencoders are neural networks training to predict their input from their output. 
The neural network can be thought of consisiting of two major components: (1) an encoder, $$ \operatorname{f}(x), $$ that learns the latent representation and (2) a decoder, $$ \operatorname{g}(x) $$, that learns to re-create the input from the output.
They are designed to be unable to copy the data exactly. 

Some common mechansims of preventing the ability to copy inputs to outputs are:
  - **Undercomplete Autoencoders** lower dimension of latent variable representation than the input dimension. 
  In this case the loss fucntion is simple $$ \operatorname{L}(x, \operatorname{g}(\operatorname{f}(x)) $$ where $$ \operatorname{L} $$ simply measures similarity (e.g. mean squared error)
  - **Regularised Autoencoders** use a loss function that encourages the model to have properties besides copying input to output. For example: sparsity of the representation, smallness of the derivative of the representation and robustness to noise or missing outputs.
  
## What are autoencoders useful for?
  - **Dimensionality reduction**: Useful for data visualisation, can aid other tasks such as classification - information from mapping to lower-dimension aid generalisation. Reduce memory and runtime. 
  - **Information retrieval tasks** (using query to retrieve info from database): Has the usual benifits of dimensionality reduction, as well as allowing a the search to become far more efficient (if the low dimensional representation is binary). 
  If the representation is created is binary, then we can store all database entries in a hash table that maps binary code vectors to entries. 
  We can then do retrieval by returning all results from the hash table that have the same representation as a query. 
  Slightly less similar results can be efficienctly searched through by flipping bits in the encoding of the query. This approach is called **semantic hashing**. 
  To do this sigmoid activations are typically used in the final later. They need to produce values very close to 0 or 1 for the semantic hashing to work - one trick to accompish this is to inject additive noise just before the sigmoid activation during training, and increasing the amount of noise through time. 
  To fight the noise and preserve a signal from the data, the NN has to increase the magnitude of the inputs until saturation occurs. 
