from torch import nn
import torchvision

"""
This is the network definition for the predictor network as defined in the original paper:
Transformer Architecture:
- 6 layers, 16 attention heads, 10% dropout
- AdaLN at the end of each layer in order to incorporate the actions into the network
- the network takes a window of observation embeddings as the input 
    - for the pushT environment this window is set to 3
"""

class PredictorNetwork(nn.Module):
    def __init__():
        super().__init__()

        



    def forward(x):
        pass