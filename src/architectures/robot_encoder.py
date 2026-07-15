from torch import nn
import torch
from architectures.encoder import EncoderNetwork

"""
This is the modified encoder network that extends it to take in
2 camera observations plus robot state information and then output
a final embedding in latent space. Also modifying it to be 256x256
instead of 224x224 which was for the original definition.
This encoder will now seperately pass through each input image through
the default encoder and then concatenate them together with the robot state
vector and then this concatenated vector will be passed through a fusion MLP
to compress it back to the original embedding space. The output of this encoder
will still be the standard dimension so it can be passed through the default
predictor
"""

class RobotEncoder(nn.Module):
    def __init__(self,
                 patch : int = 16,
                 nheads : int = 3,
                 nlayers : int = 12,
                 d_model : int = 192,
                 d_state : int = 64,
                 npatches : int = 256): # npatches is (H/patch_size)*(W/patch_size)
        super().__init__()
        self.encoder = EncoderNetwork(patch=patch,
                                      nheads=nheads,
                                      nlayers=nlayers,
                                      d_model=d_model,
                                      npatches=npatches)

        self.fusion_mlp = nn.Sequential(nn.Linear(in_features=(d_model*2 + d_state), out_features=(d_model*2)),
                                        nn.GELU(),
                                        nn.Linear(in_features=(d_model*2), out_features=d_model))
    
    def forward(self, above, wrist, state):
        # above : [Batch, Window, 3, 256, 256] | wrist : [Batch, Window, 3, 256, 256] | state : [Batch, Window, 64]
        above_emb = self.encoder(above) # [Batch, Window, 192]
        wrist_emb = self.encoder(wrist) # [Batch, Window, 192]
        fusion = torch.cat((above_emb, wrist_emb, state), dim=-1)
        return self.fusion_mlp(fusion)


# test_above = torch.ones(12, 3, 3, 256, 256)
# test_wrist = torch.ones(12, 3, 3, 256, 256)
# test_state = torch.ones(12, 3, 64)

# dummy = RobotEncoder()

# print(dummy(test_above, test_wrist, test_state).shape)


        