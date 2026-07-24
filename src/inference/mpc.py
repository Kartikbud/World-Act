import torch
from torch import nn
import numpy as np

"""
This is where the latent space planner is implemented.
It essentially utilizes CEM to solve the MPC problem by optimizing
for the actions that place the state after some horizon of actions closest
to the goal state. The cost is the L2 loss between the current state embedding
and the goal embedding, this is also the MSELoss in PyTorch.

zi: initial state but this will be a window of embeddings where the last embedding
    is the true state
zg: this is the goal state and is a single embedding
"""

class Planner():
    def __init__(self,
                 enc : nn.Module, # encoder model
                 pred : nn.Module, # predictor model
                 H : int = 5, # planning horizon
                 N : int = 300, # number of samples per iteration
                 K : int = 30, # number of elites per iteration
                 T : int = 30): # number of iterations
        
        self.enc = enc
        self.pred = pred
        self.H = H
        self.N = N
        self.K = K
        self.T = T

    def CEM(self, zi, zg): 
        # zi shape: (window, embedding dim)

        current = zi[-1]

        mean = np.zeros(7) 
        covariance = np.eye(7) # identity matrix

        sampler = np.random.default_rng()

        for i in range(self.T):
            action_samples = []
            costs = []
            
            for i in range(self.N): # sampling and then rolling out the actions using the model
                actions = sampler.multivariate_normal(mean, covariance, size=self.H)
                action_samples.append(torch.from_numpy(actions))
                window = zi
                for j in range(self.H):
                    emb = self.enc(window)                 
                    pred = self.pred(emb, actions[j])[-1] # only taking the last one so shape is (192,)
                    window = torch.cat(window[1:], pred.unsqueeze(0), dim=1) # pred becomes (1, 192) and then is concatenated with the window but pops first emb
                costs.append(nn.MSELoss(window[-1], zg))

                    

