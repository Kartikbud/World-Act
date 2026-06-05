import torch 
from torch.utils.data import Dataset

"""
This script is for processing the data from the raw pusht files:
- observations come in the form of videos and these have a max of 246 frames
    - sometimes the episode ends early and the rest are padding and the length
      of the actual sequence for each episode is in the seq.pkl
- there are delta position based action tensors that correspond to each frame which will be used
- to make trainin
"""

class PushTDataset(Dataset):
    def __init__(self,
                 batch_size : int,
                 frame_skip : int):
        super().__init__()
