import torch
from torch import nn

"""
This is where the latent space planner is implemented.
It essentially utilizes CEM to solve the MPC problem by optimizing
for the actions that place the state after some horizon of actions closest
to the goal state.
"""

class Planner():
    def __init__(self,
                 enc : nn.Module, # encoder model
                 pred : nn.Module, # predictor model
                 H : int, # planning horizon
                 N : int, # number of samples per iteration
                 K : int, # number of elites per iteration
                 T : int): # number of iterations
        
        self.enc = enc
        self.pred = pred
        self.H = H
        self.N = N
        self.K = K
        self.T = T

    def cost(self, zx, zg):
        pass

    def CEM(self, zi, zg):
        pass