import torch
from torch import Tensor
from torch.linalg import inv_ex
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from functools import lru_cache
import torchvision
import h5py

import time

"""
This script is for processing the data from the raw pusht files:
- observations come in the form of videos and these have a max of 246 frames
    - sometimes the episode ends early and the rest of the actions are padded so they can be
	  a single tensor
- there are delta position based action tensors that correspond to each frame which will be used
- to make inference more efficient, frame skips are utilized so instead of predicting the next frame
  you predict the Nth next frame
- if frame skip was 2 this would make the data pairs:
  - x: (frame0, a0 + a1), y: frame3
"""

class PushTDataset(Dataset):
	def __init__(self,
				 data_dir : Path,
				 frame_skip : int,
				 window : int,
				 val : bool = False):
		super().__init__()
		dir_var = "val" if val else "train"

		self.frame_skip = frame_skip
		self.window = window
		self.action_tensor = torch.load(data_dir / "pusht_noise" / "pusht_noise" / dir_var / "rel_actions.pth")
		# shape of action tensor: [18685, 246, 2] : [episodes, action per frame + padding, ]
		split_dir = data_dir / "pusht_noise" / "pusht_noise" / dir_var
		self.h5_path = split_dir / "obs_tensors.h5"
		obs_dir = split_dir / "obses"
		episodes = [ep.stem for ep in sorted(obs_dir.iterdir())]
		# data will be returned like this frames : [window of inputs frames, next frame], [window of actions]
		# only valid batches of frames will be returned, not the 
		self.frame_samples = []
		with h5py.File(self.h5_path, "r") as f:
			for idx, ep in enumerate(episodes):
				ep_len = f[f"{ep}/frames"].shape[0]
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
		
		self.total_len = len(self.frame_samples)

	@lru_cache(maxsize=1)
	def _h5(self):
		return h5py.File(self.h5_path, "r")
		
	
	def __len__(self):
		return self.total_len

	def __getitem__(self, idx):
		sample = self.frame_samples[idx]
		ep_num, ep_name, frames = sample
		input_frames = frames[:-1]
		target_frames = frames[1:]

		dset = self._h5()[f"{ep_name}/frames"]
		input_window_tensor = torch.from_numpy(dset[input_frames]).float() / 255.0
		target_state_tensor = torch.from_numpy(dset[target_frames]).float() / 255.0
		
		ep_action = self.action_tensor[ep_num]
		action_window = []
		for i in range(len(frames) - 1):
			action = ep_action[frames[i]:frames[i + 1]].sum(dim=0)
			action_window.append(action)

		action_window_tensor = torch.stack(action_window, dim=0).float()

		return input_window_tensor, action_window_tensor, target_state_tensor

# PROJECT_DIR = Path(__file__).resolve().parent.parent.parent    
    
# data_dir = PROJECT_DIR / "data"

# test = PushTDataset(data_dir, 5, 3)
# test_dl = DataLoader(test, batch_size=128)

# print(len(test_dl))
