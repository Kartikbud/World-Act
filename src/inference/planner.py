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
                 zi, # initial state
                 zg, # goal state
                 f : nn.Module, # world model
                 H : int, # planning horizon
                 N : int, # number of samples per iteration
                 K : int, # number of elites per iteration
                 T : int): # number of iterations
        
        self.zi = zi
        self.zg = zg
        self.f = f
        self.H = H
        self.N = N
        self.K = K
        self.T = T

    def cost(self):
        pass

    def CEM(self):
        pass