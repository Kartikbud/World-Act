import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from torch import nn
import torch
import torch.nn.functional as F

"""
This is the network definition for the baseline of my APS360 Project.
Its based on the visuomotor CNN from the VR teleop imitation learning paper
(Zhang et al.) which itself follows the GPS style architecture:
- CNN backbone over the images ending in a spatial soft-argmax
- proprioception gets concatenated near the end
- then an MLP outputs the action

The original paper used 1 RGB view + depth, here its just the two RGB
cameras (agentview and eye-in-hand) stacked as a 6 channel input instead.
Also dropping all the auxiliary prediction heads they had and just using
a simple MLP action head that outputs the 7D OSC_POSE action. Proprio is
the usual 9D state (eef pos + quat + gripper qpos). Sized to around 8-9M
params so it roughly matches the encoder + predictor combined.
"""


class SpatialSoftArgmax(nn.Module):
	"""Per-channel spatial softmax → expected (x, y) in [-1, 1]."""

	def forward(self, x):
		# x: (B, C, H, W)
		B, C, H, W = x.shape
		flat = x.view(B, C, -1)
		weights = F.softmax(flat, dim=-1).view(B, C, H, W)

		ys = torch.linspace(-1.0, 1.0, H, device=x.device, dtype=x.dtype)
		xs = torch.linspace(-1.0, 1.0, W, device=x.device, dtype=x.dtype)
		ey = (weights.sum(dim=-1) * ys.view(1, 1, H)).sum(dim=-1)  # (B, C)
		ex = (weights.sum(dim=-2) * xs.view(1, 1, W)).sum(dim=-1)  # (B, C)
		return torch.stack([ex, ey], dim=-1).reshape(B, C * 2)


class BaselinePolicy(nn.Module):
	def __init__(self,
				 d_state: int = 9,
				 d_action: int = 7,
				 hidden: int = 1536):
		super().__init__()
		# 6-channel input = two RGB images stacked on the channel dim
		self.cnn = nn.Sequential(
			nn.Conv2d(6, 128, kernel_size=7, stride=2, padding=3),
			nn.ReLU(inplace=True),
			nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
			nn.ReLU(inplace=True),
			nn.Conv2d(256, 384, kernel_size=5, stride=2, padding=2),
			nn.ReLU(inplace=True),
			nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
			nn.ReLU(inplace=True),
		)
		self.spatial_softargmax = SpatialSoftArgmax()
		feat_dim = 384 * 2  # (x, y) per channel

		self.action_head = nn.Sequential(
			nn.Linear(feat_dim + d_state, hidden),
			nn.ReLU(inplace=True),
			nn.Linear(hidden, hidden),
			nn.ReLU(inplace=True),
			nn.Linear(hidden, d_action),
		)

	def forward(self, above, wrist, state):
		# above/wrist: (B, 3, H, W)  |  state: (B, 9)  →  action: (B, 7)
		x = torch.cat([above, wrist], dim=1)  # (B, 6, H, W)
		feat = self.spatial_softargmax(self.cnn(x))
		return self.action_head(torch.cat([feat, state], dim=-1))


if __name__ == "__main__":
	dummy = BaselinePolicy()
	above = torch.randn(4, 3, 84, 84)
	wrist = torch.randn(4, 3, 84, 84)
	state = torch.randn(4, 9)
	out = dummy(above, wrist, state)
	n = sum(p.numel() for p in dummy.parameters())
	print(out.shape, f"params={n}")
