"""
Evaluate BaselinePolicy in a robosuite Stack env.

Rolls out N episodes with a trained checkpoint and reports success rate.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
	sys.path.insert(0, str(SRC))

# Repo checkout is named `robosuite/`; drop it so the installed package wins.
_repo_rs = str((ROOT / "robosuite").resolve())
sys.path = [p for p in sys.path if str(Path(p).resolve()) != _repo_rs]
sys.path = [p for p in sys.path if p not in ("", str(ROOT), str(ROOT.resolve()))]

from architectures.baseline import BaselinePolicy

import robosuite as suite
from robosuite.controllers import load_controller_config


CAM_ABOVE = "agentview"
CAM_WRIST = "robot0_eye_in_hand"
IMG_SIZE = 84
MAX_STEPS = 600
DEFAULT_CHECKPOINT = ROOT / "models" / "baseline" / "models" / "baseline_latest.pth"


def obs_to_tensors(obs, device):
	above = torch.from_numpy(obs[f"{CAM_ABOVE}_image"]).float() / 255.0
	wrist = torch.from_numpy(obs[f"{CAM_WRIST}_image"]).float() / 255.0
	above = above.permute(2, 0, 1).unsqueeze(0).to(device)
	wrist = wrist.permute(2, 0, 1).unsqueeze(0).to(device)
	state = torch.from_numpy(
		np.concatenate(
			[
				obs["robot0_eef_pos"],
				obs["robot0_eef_quat"],
				obs["robot0_gripper_qpos"],
			],
			axis=-1,
		)
	).float().unsqueeze(0).to(device)
	return above, wrist, state


def action_from_policy(policy, above, wrist, state):
	with torch.inference_mode():
		action = policy(above, wrist, state).squeeze(0).cpu().numpy()
	action[6] = 1.0 if action[6] > 0 else -1.0
	return action


def make_env():
	controller = load_controller_config(default_controller="OSC_POSE")
	return suite.make(
		"Stack",
		robots="Panda",
		controller_configs=controller,
		has_renderer=False,
		has_offscreen_renderer=True,
		use_camera_obs=True,
		camera_names=[CAM_ABOVE, CAM_WRIST],
		camera_heights=IMG_SIZE,
		camera_widths=IMG_SIZE,
		control_freq=20,
		horizon=MAX_STEPS,
	)


def run_episode(env, policy, device):
	obs = env.reset()
	for _ in range(MAX_STEPS):
		above, wrist, state = obs_to_tensors(obs, device)
		action = action_from_policy(policy, above, wrist, state)
		obs, _reward, done, _info = env.step(action)
		if done:
			break
	return bool(env._check_success())


def evaluate(checkpoint: Path, episodes: int, seed: int, device: str):
	torch.manual_seed(seed)
	np.random.seed(seed)

	if not checkpoint.exists():
		raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

	policy = BaselinePolicy().to(device)
	state_dict = torch.load(checkpoint, map_location=device, weights_only=True)
	policy.load_state_dict(state_dict)
	policy.eval()

	env = make_env()
	successes = 0

	print(f"checkpoint: {checkpoint}")
	print(f"device: {device} | episodes: {episodes} | max_steps: {MAX_STEPS}")

	for ep in range(episodes):
		success = run_episode(env, policy, device)
		successes += int(success)
		print(f"episode {ep + 1}/{episodes}: {'success' if success else 'fail'}")

	env.close()

	rate = successes / episodes
	print(f"\nsuccess rate: {successes}/{episodes} = {100 * rate:.1f}%")
	return rate


def main():
	parser = argparse.ArgumentParser(description="Evaluate BaselinePolicy on Stack")
	parser.add_argument(
		"--checkpoint",
		type=Path,
		default=DEFAULT_CHECKPOINT,
		help="Path to model checkpoint (.pth)",
	)
	parser.add_argument("--episodes", type=int, default=10)
	parser.add_argument("--seed", type=int, default=42)
	args = parser.parse_args()

	device = "cuda" if torch.cuda.is_available() else "cpu"
	evaluate(args.checkpoint, args.episodes, args.seed, device)


if __name__ == "__main__":
	main()
