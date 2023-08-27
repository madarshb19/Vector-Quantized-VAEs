# Vector-Quantized-VAEs
Short summary of what these are and code

# What are those?

Very similar to the typical VAEs with encoders and decoders but we have discrete representations of the latent spaces. Why do discretization? Reasons are the following:-
1) A lot of things we find are discrete (For example in images we might have categories like "Cat", "Car", etc. and it might not make sense to interpolate between these categories.)
2) Makes life easier to model discrete rather than continuous latent spaces.
3) Ez publication

This discretization is done by storing a codebook as the embedding space and every vector is then approximated to one of the vectors in the codebook. You could see one of the loss terms being the L2 norm between ze(x) which is the encoded vector and ej which is the embedding vector in the codebook.

# Problem with backprop

If we snap these encoded vectors to the nearest embedding in the codebook, chances are that the decoder function would become non-differentiable making it impossible to backprop gradients. Fortunately, there is a solution. The length of the encoder and decoder vectors (D) are the same. So, what we do is copypaste these gradients into the encoder vectors itself and backprop from there. (they call it the straight-through estimator)

# Weird looking loss function?

The terms in the loss function are way different from what appears in the original VAE paper.In this paper, there is a log-likelihood term in the log function which compares to the reconstruction loss in the original paper. It does eventually result in the MSE loss itself along with the other terms. This is because we assume the distribution of p is gaussian, so it contains e^-(...)^2 term.So when we take log, we do eventually end up with the square term in the loss function. The KL term from the VAE paper is replaced by these two terms instead. sg is a stop gradient. In pytorch, we basically detach these gradients from the computational graph and push these frozen encoded vectors to the codebook. Its the opposite in the third term, we freeze the codebook vectors and then push the encoded vectors into the frozen codebook vectors.

All the 3 terms in the loss function have a different role to play. The 2nd term optimizes the embedding vectors. The first reconstruction loss term is optimized for the decoder and both 1st and 3rd term are optimized by the encoder.

# Confusion surrounding prior and where does autoregressive modelling come into picture? (still dont quite understand it,ngl)

Why the KL divergence reconstruction loss was no taking into account was explained in the code part. After backprop is done, the output is again fed into the input. This is similar to the token prediction problem in language modelling where given all the previous tokens,we predict the next token.How?

We first train this model end-to-end,i.e, given 32x32x3 image,feed it thru encoder->quantizer->decoder. Do this for all images such that the quantizer learns the codebook and the different embedding representations.Now to generate new images,what we do is freeze all weights. Start with a special token and let the model prompt a value. This value is fed back into the input and the process is repeated until the 32x32 image is created and in this sense, it is considered an autoregressive model.

# The v2 of this paper

There is an improved version of VQ-VAEs where most of the theory is the same except for the latent vectors where they have a higher and lower level codebooks with different resolutions to capture both higher and lower level features in the embedding space. The same is done with the priors as well.
