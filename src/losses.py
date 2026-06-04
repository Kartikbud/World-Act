import torch
from torch import nn
import numpy as np

"""
This is where SIGReg (Sketched Isotropic Gaussian Regularizer) is defined:

This regulizer represents a loss in terms of how well a batch of embeddings 
form an isotropic gaussian and that's just a way of saying how spread apart
the embedding space is. This is a very important term as it helps to prevent
representation collapse of the latent space.
"""

class SIGReg(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, Z):
        pass