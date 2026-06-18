import torch
from torch import Tensor
from torch.linalg import inv_ex
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
		obs_dir = Path(data_dir / "pusht_noise" / "pusht_noise" / dir_var / "obses")
		video_dirs = [ep for ep in obs_dir.iterdir()] # list of Path objects for each vid 
		# data will be returned like this frames : [window of inputs frames, next frame], [window of actions]
		# only valid batches of frames will be returned, not the 
		self.frame_samples = []
		self.raw_vid_tensors = []
		for idx, ep in enumerate(video_dirs):
			decoder = VideoDecoder(ep)
			ep_len = len(decoder)
			self.raw_vid_tensors.append([decoder[i] for i in range(ep_len)])
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
		
	
	def __len__(self):
		return self.total_len

	def __getitem__(self, idx):
		sample = self.frame_samples[idx]
		ep_num, ep_path, frames = sample
		input_frames = frames[:-1]
		target_frames = frames[1:]

		input_window_tensor = torch.stack([self.raw_vid_tensors[ep_num][i] for i in input_frames]).float() / 255.0
		target_state_tensor = torch.stack([self.raw_vid_tensors[ep_num][i] for i in target_frames]).float() / 255.0
		
		ep_action = self.action_tensor[ep_num]
		action_window = []
		for i in range(len(frames) - 1):
			action = ep_action[frames[i]:frames[i + 1]].sum(dim=0)
			action_window.append(action)

		action_window_tensor = torch.stack(action_window, dim=0).float()

		return input_window_tensor, action_window_tensor, target_state_tensor

project_dir = Path(__file__).resolve().parent.parent.parent

data_dir = project_dir / "data"

test_dataset = PushTDataset(data_dir, 5, 3)

in_window, action, target = test_dataset.__getitem__(3)

print(in_window.shape)
print(action.shape)
print(target.shape)

print("actual tensors")

print(in_window)
print(target)



