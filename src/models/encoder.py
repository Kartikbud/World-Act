from torch import nn

"""
This is the network definition for the encoder network which in the paper
was just ViT-Tiny which is a visual transformer with the following parameters:
- patch size of 14, 12 layers, 3 attention heads, and a hidden dimension of 192
- there is a "projection" layer added to the end of this because visual transformers
  typically have layernorms at the end but this ruins the regularization used to
  prevent representation collapse
    - this projection layer is a single layer with batch normalization to undo the
      affects of the layernorm at the end
"""

class EncoderNetwork(nn.Module):
    def __init__(self):
        super().__init__()

    
    def forward(x):
        pass