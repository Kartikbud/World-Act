import torch
from torch import Tensor
from torch.utils.data import Dataset
from pathlib import Path
import torchvision
from torchcodec.decoders import VideoDecoder

"""
This script is for processing the data from the raw pusht files:
- observations come in the form of videos and these have a max of 246 frames
    - sometimes the episode ends early and the rest of the actions are padded so they can be
	  a single tensor
- there are delta position based action tensors that correspond to each frame which will be used
- to make inference more efficient, frame skips are utilized so instead of predicting the next frame
  you predict the Nth next frame
- if frame skip was 2 this would make the data pairs:
  - x: (frame0, a0 + a1 + a2), y: frame3
"""

class PushTDataset(Dataset):
	def __init__(self,
				 data_dir : Path,
				 frame_skip : int,
				 window : int,
				 val : bool = False):
		super().__init__()
		dir_var = "val" if val else "train"

		self.action_tensor = torch.load(data_dir / "pusht_noise" / "pusht_noise" / dir_var / "rel_actions.pth")
		# shape of action tensor: [18685, 246, 2] : [episodes, action per frame + padding, ]
		obs_dir = Path(data_dir / "pusht_noise" / "pusht_noise" / dir_var / "obses")
		video_dirs = [ep for ep in obs_dir.iterdir()] # list of Path objects for each vid 
		# data will be returned like this frames : [window of inputs frames, next frame], [window of actions]
		# only valid batches of frames will be returned, not the 
		self.frame_samples = []
		for idx, ep in enumerate(video_dirs):
			decoder = VideoDecoder(ep)
			ep_len = len(decoder)
			for i in range((window - 1)*frame_skip, ep_len - 1 - frame_skip):
				batch = []
				for j in reversed(range(window)):
					batch.append(i - (j*frame_skip))
				batch.append(i + frame_skip)
				self.frame_samples.append((idx, ep, batch))
		
		"""
		frame_samples looks like this:
		[(0, ep0 Path, [0, 5, 10, 15]),
		 (0, ep0 Path, [1, 6, 11, 16]),
					...				,
		(3, ep3 Path, [5, 10, 15, 20])]
		"""
		
		self.total_len = sum(len(ep_batch) for ep_batch in self.frame_samples)
		
	
	def __len__(self):
		return self.total_len

	def __getitem__(self, idx):
		sample = self.frame_samples[idx]
		ep_num, ep_path, frames = sample
		
		decoder = VideoDecoder(ep_path)
		ep_action = self.action_tensor[ep_num]

		

data_dir = Path("/Users/kartikbudihal/leworld/data")

test_vid = VideoDecoder(data_dir / "pusht_noise" / "pusht_noise" / "train" / "obses" / "episode_1002.mp4")

print(test_vid[:].shape[0])
print(len(test_vid))

action_tensor = torch.load(data_dir / "pusht_noise" / "pusht_noise" / "train" / "rel_actions.pth")

