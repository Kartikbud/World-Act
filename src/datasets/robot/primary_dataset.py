import re
from functools import lru_cache
from pathlib import Path

import h5py
import torch
from torch.utils.data import Dataset

"""
PyTorch dataset over MimicGen Stack demos (robomimic-style HDF5).

Mirrors the PushT windowing logic:
- frame_skip: predict the Nth next frame instead of the immediate next
- window: number of context frames
- actions between skipped frames are summed (same as PushT)

Each sample returns observations for RobotEncoder plus the action window:
  above_in, wrist_in, state_in, actions, above_tgt, wrist_tgt, state_tgt

Proprio state is eef_pos (3) + eef_quat (4) + gripper_qpos (2) = 9D.
Images are agentview / eye-in-hand, returned as float CHW in [0, 1].

Train vs val is selected via the HDF5 mask group (`mask/train`, `mask/valid`).
"""


def _demo_number(name: str) -> int:
	return int(re.search(r"(\d+)", name).group(1))


def _decode_demo_name(raw) -> str:
	if isinstance(raw, bytes):
		return raw.decode("utf-8")
	return str(raw)


class RobotPrimaryDataset(Dataset):
	def __init__(self,
				 data_dir: Path,
				 frame_skip: int,
				 window: int,
				 val: bool = False):
		super().__init__()
		self.frame_skip = frame_skip
		self.window = window
		self.h5_path = Path(data_dir) / "robot" / "mimicgen" / "demo.hdf5"
		if not self.h5_path.exists():
			raise FileNotFoundError(f"Dataset not found: {self.h5_path}")

		mask_key = "valid" if val else "train"
		self.frame_samples = []
		with h5py.File(self.h5_path, "r") as f:
			if "mask" not in f or mask_key not in f["mask"]:
				raise KeyError(
					f"Expected mask/{mask_key} in {self.h5_path}. "
					"Re-run generate_dataset.py (it writes train/valid masks) "
					"or add the split manually."
				)
			episodes = [_decode_demo_name(x) for x in f["mask"][mask_key][:]]
			episodes = sorted(episodes, key=_demo_number)

			for idx, ep in enumerate(episodes):
				ep_len = int(f[f"data/{ep}"].attrs.get(
					"num_samples",
					f[f"data/{ep}/actions"].shape[0],
				))
				# same bounds as PushT: need a full context window and one target step
				for i in range((window - 1) * frame_skip, ep_len - 1 - frame_skip):
					batch = []
					for j in reversed(range(window)):
						batch.append(i - (j * frame_skip))
					batch.append(i + frame_skip)
					self.frame_samples.append((idx, ep, batch))

		"""
		frame_samples looks like this:
		[(0, 'demo_0', [0, 5, 10, 15]),
		 (0, 'demo_0', [1, 6, 11, 16]),
					...				,
		 (3, 'demo_3', [5, 10, 15, 20])]
		"""

		self.total_len = len(self.frame_samples)

	@lru_cache(maxsize=1)
	def _h5(self):
		return h5py.File(self.h5_path, "r")

	def __len__(self):
		return self.total_len

	def _load_images(self, ep_name, frame_idx, cam_key):
		dset = self._h5()[f"data/{ep_name}/obs/{cam_key}"]
		# HDF5 stores (T, H, W, C) uint8 → (W_frames, C, H, W) float
		imgs = torch.from_numpy(dset[frame_idx]).float() / 255.0
		return imgs.permute(0, 3, 1, 2).contiguous()

	def _load_state(self, ep_name, frame_idx):
		obs = self._h5()[f"data/{ep_name}/obs"]
		pos = torch.from_numpy(obs["robot0_eef_pos"][frame_idx]).float()
		quat = torch.from_numpy(obs["robot0_eef_quat"][frame_idx]).float()
		grip = torch.from_numpy(obs["robot0_gripper_qpos"][frame_idx]).float()
		return torch.cat([pos, quat, grip], dim=-1)

	def __getitem__(self, idx):
		_ep_num, ep_name, frames = self.frame_samples[idx]
		input_frames = frames[:-1]
		target_frames = frames[1:]

		above_in = self._load_images(ep_name, input_frames, "agentview_image")
		wrist_in = self._load_images(ep_name, input_frames, "robot0_eye_in_hand_image")
		state_in = self._load_state(ep_name, input_frames)

		above_tgt = self._load_images(ep_name, target_frames, "agentview_image")
		wrist_tgt = self._load_images(ep_name, target_frames, "robot0_eye_in_hand_image")
		state_tgt = self._load_state(ep_name, target_frames)

		ep_action = torch.from_numpy(
			self._h5()[f"data/{ep_name}/actions"][:]
		).float()
		action_window = []
		for i in range(len(frames) - 1):
			action = ep_action[frames[i]:frames[i + 1]].sum(dim=0)
			action_window.append(action)
		action_window_tensor = torch.stack(action_window, dim=0)

		return (
			above_in,
			wrist_in,
			state_in,
			action_window_tensor,
			above_tgt,
			wrist_tgt,
			state_tgt,
		)


# PROJECT_DIR = Path(__file__).resolve().parents[3]
# data_dir = PROJECT_DIR / "data"
# test = RobotPrimaryDataset(data_dir, 5, 3)
# print(len(test), [t.shape for t in test[0]])
