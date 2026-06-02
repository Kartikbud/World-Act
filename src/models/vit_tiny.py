from torch import nn
import torch
from modules import TransformerBlock

"""
Thought it would be fun to define the architecture of the ViT-Tiny
from scratch so here it is using the transformers I defined as well
"""

class CustomVitTiny(nn.Module):
    def __init__(self,
                 patch : int = 14,
                 nheads : int = 3,
                 nlayers : int = 12,
                 d_model : int = 192):
        super().__init__()

        self.conv_flattener = nn.Conv2d(in_channels=3,
                                        out_channels=d_model)

    def forward(x):
        pass




# TEST CODE
# dummy_conv = nn.Conv2d(in_channels=3,
#           out_channels=192,
#           kernel_size=14,
#           stride=14,
#           padding=0)

# dummy_frame = torch.ones((512, 3, 3, 224, 224))
# dummy_frame = dummy_frame.reshape(512*3, 3, 224, 224)
# emb = dummy_conv(dummy_frame)

# print(emb.shape)
# print("after reshaping")
# emb = emb.reshape((512, 3, 192, 256))
# emb = emb.permute(0, 1, 3, 2)
# print(emb.shape)