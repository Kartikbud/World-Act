import torch
from torch import nn
import numpy as np

"""
This is where SIGReg (Sketched Isotropic Gaussian Regularizer) is defined:

This regulizer represents a loss in terms of how well a batch of embeddings 
form an isotropic gaussian and that's just a way of saying how spread apart
the embedding space is. This is a very important term as it helps to prevent
representation collapse of the latent space which is when all embeddings passed
through the encoder collapse to a trivial solution like 0.
"""

class SIGReg(nn.Module):
    def __init__(self, 
                 knots : int = 17, # the number of points being used to approximate the integral
                 num_proj : int = 1024): # number of random directions the vector will be projected in (M) 
        super().__init__()
        # setting up the trapezoidal approximation of the integral in epps-pulley test
        
        self.num_proj = num_proj
        # this is the tensor split up into points interpolated between 0 to 3 which will be integrated over
        # only need to 0 to 3 as the gaussian essentially goes to 0 after 3 so this serves as a strong approximate
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1) 
        
        # these are the weights based on the trapezoid approximation [f(x0) + 2f(x1) + ... 2f(xn-1) + f(xn)]
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt

        # gaussian characteristic function
        window = torch.exp(-t.square() / 2.0)

        # storing relevant values in buffer as non learnable parameters so that they move across devices with the nn Module
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights*window)
    
    def forward(self, Z): # Z : (N, B, D) or (N+1, B, D) if target embeddins is being included
        A = torch.randn(Z.size(-1), self.num_proj, device=Z.device) # tensor of random directions that the embeddings will be projected onto
        A = A.div_(A.norm(p=2, dim=0)) # normalizing the tensor to be unit vectors
        
        # computing the epps-pulley stat
        x_t = (Z @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        stat = (err @ self.weights) * Z.size(-2)

        return stat.mean() # returning the mean error across each projection