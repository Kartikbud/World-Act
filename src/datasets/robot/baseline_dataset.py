import re
from pathlib import Path

import h5py
import torch
from torch.utils.data import Dataset

"""
This is the dataset for the behavioral cloning baseline.

Same HDF5 source as the primary dataset (mimicgen demo.hdf5) and the same
train/val mask split, but no frame skip or windowing since BC is just
supervised pairs of (observation, action) at each timestep.

Each sample returns:
  above, wrist, state, action

Proprio state is eef_pos (3) + eef_quat (4) + gripper_qpos (2) = 9D.
Images are agentview / eye-in-hand, returned as float CHW in [0, 1] at 84x84.
"""


def _demo_number(name: str) -> int:
	return int(re.search(r"(\d+)", name).group(1))

def _decode_demo_name(raw) -> str:
	if isinstance(raw, bytes):
		return raw.decode("utf-8")
	return str(raw)


class RobotBaselineDataset(Dataset):
	def __init__(self,
				 data_dir: Path,
				 val: bool = False):
		super().__init__()
		self.h5_path = Path(data_dir) / "robot" / "mimicgen" / "demo.hdf5"
		if not self.h5_path.exists():
			raise FileNotFoundError(f"Dataset not found: {self.h5_path}")

		mask_key = "valid" if val else "train"
		self.frame_samples = []
		with h5py.File(self.h5_path, "r") as f:
			episodes = [_decode_demo_name(x) for x in f["mask"][mask_key][:]]
			episodes = sorted(episodes, key=_demo_number)

			for idx, ep in enumerate(episodes):
				ep_len = int(f[f"data/{ep}"].attrs.get(
					"num_samples",
					f[f"data/{ep}/actions"].shape[0],
				))
				for t in range(ep_len):
					self.frame_samples.append((idx, ep, t))

		"""
		frame_samples looks like this:
		[(0, 'demo_0', 0),
		 (0, 'demo_0', 1),
					...	   ,
		 (3, 'demo_3', 120)]
		"""

		self.total_len = len(self.frame_samples)

	def __len__(self):
		return self.total_len

	def _load_image(self, f, ep_name, t, cam_key):
		dset = f[f"data/{ep_name}/obs/{cam_key}"]
		# HDF5 stores (T, H, W, C) uint8 → (C, H, W) float
		img = torch.from_numpy(dset[t]).float() / 255.0
		return img.permute(2, 0, 1).contiguous()

	def _load_state(self, f, ep_name, t):
		obs = f[f"data/{ep_name}/obs"]
		pos = torch.from_numpy(obs["robot0_eef_pos"][t]).float()
		quat = torch.from_numpy(obs["robot0_eef_quat"][t]).float()
		grip = torch.from_numpy(obs["robot0_gripper_qpos"][t]).float()
		return torch.cat([pos, quat, grip], dim=-1)

	def __getitem__(self, idx):
		_ep_num, ep_name, t = self.frame_samples[idx]

		with h5py.File(self.h5_path, "r") as f:
			above = self._load_image(f, ep_name, t, "agentview_image")
			wrist = self._load_image(f, ep_name, t, "robot0_eye_in_hand_image")
			state = self._load_state(f, ep_name, t)
			action = torch.from_numpy(f[f"data/{ep_name}/actions"][t]).float()

		return above, wrist, state, action


# PROJECT_DIR = Path(__file__).resolve().parents[3]
# data_dir = PROJECT_DIR / "data"
# test = RobotBaselineDataset(data_dir)
# print(len(test), [t.shape for t in test[0]])
