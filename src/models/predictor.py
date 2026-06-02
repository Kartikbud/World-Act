from torch import nn
import torch
from modules import TransformerBlock


"""
This is the network definition for the predictor network as defined in the original paper:
Transformer Architecture:
- 6 layers, 16 attention heads, 10% dropout
- AdaLN at the beginning of each layer in order to incorporate the actions into the network
- the network takes a window of observation embeddings as the input 
    - for the pushT environment this window is set to 3
"""

class PredictorNetwork(nn.Module):
    def __init__(self,
                 d_model: int = 192,
                 nheads : int = 16,
                 d_action : int = 2,
                 dropout : float = 0.1,
                 nlayers : int = 6,
                 window : int = 3):
        super().__init__()

        self.d_model = d_model
        
        self.transformer_layers = nn.ModuleList([])
        for _ in range(nlayers):
            self.transformer_layers.append(TransformerBlock(d_model, 
                                           nheads, 
                                           d_action, 
                                           dropout))
        # learned positional embeddings
        self.position_emb = nn.Parameter(torch.randn(window, d_model))

        # projection layer
        self.projector = nn.Linear(in_features=d_model, out_features=d_model)
        self.batch_norm = nn.BatchNorm1d(d_model)

    def forward(self, z, a):
        #shape of z: (B, N, D), a: (B, N, D_a)
        B = z.shape[0]
        N = z.shape[1]
        # applying the positional embeddings
        z = z + self.position_emb
        for layer in (self.transformer_layers):
            z = layer(z, a)
        
        z = z.reshape(B*N, self.d_model)
        z = self.batch_norm(self.projector(z))

        z = z.reshape(B, N, self.d_model)

        return z

dummy_emb = torch.ones((512, 3, 192))
dummy_act = torch.ones((512, 3, 2))
pred = PredictorNetwork()

print(pred(dummy_emb, dummy_act).shape)