import argparse
import contextlib
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
	sys.path.insert(0, str(SRC_DIR))

from architectures.baseline import BaselinePolicy
from datasets.robot.baseline_dataset import RobotBaselineDataset


def baseline_loss(pred, target, gripper_coeff: float):
	# first 6 dims: continuous OSC pose | last dim: binary gripper (-1/1 in data)
	mse = nn.functional.mse_loss(pred[:, :6], target[:, :6])
	gripper_target = (target[:, 6] > 0).float()
	gripper = nn.functional.binary_cross_entropy_with_logits(pred[:, 6], gripper_target)
	return mse + gripper_coeff * gripper, mse, gripper


def train(device,
		  data_dir: Path,
		  log_dir: Path,
		  save_dir: Path,
		  lr: float,
		  seed: int,
		  gripper_coeff: float,
		  epochs: int,
		  batch_size: int,
		  num_workers: int,
		  persistent_workers: bool,
		  pin_memory: bool,
		  optimizer_betas: tuple[float, float],
		  optimizer_weight_decay: float):

	torch.manual_seed(seed)
	if isinstance(device, str):
		on_cuda = device.startswith("cuda")
	else:
		on_cuda = device.type == "cuda"

	if on_cuda:
		torch.cuda.manual_seed(seed)
		torch.backends.cudnn.benchmark = True

	save_dir.mkdir(parents=True, exist_ok=True)
	log_dir.mkdir(parents=True, exist_ok=True)

	train_dataset = RobotBaselineDataset(data_dir=data_dir, val=False)
	val_dataset = RobotBaselineDataset(data_dir=data_dir, val=True)

	loader_kwargs = dict(
		batch_size=batch_size,
		num_workers=num_workers,
		pin_memory=pin_memory,
	)
	if num_workers > 0:
		loader_kwargs["persistent_workers"] = persistent_workers

	train_dataloader = DataLoader(dataset=train_dataset, shuffle=True, **loader_kwargs)
	val_dataloader = DataLoader(dataset=val_dataset, shuffle=False, **loader_kwargs)

	policy = BaselinePolicy().to(device)
	n_params = sum(p.numel() for p in policy.parameters())
	print(f"BaselinePolicy on {device} | params={n_params:,} ({n_params/1e6:.2f}M)")
	print(f"train samples={len(train_dataset)} | val samples={len(val_dataset)}")

	optimizer = torch.optim.AdamW(
		policy.parameters(),
		lr=lr,
		betas=optimizer_betas,
		weight_decay=optimizer_weight_decay,
	)

	writer = SummaryWriter(log_dir=log_dir)
	autocast_ctx = (
		torch.amp.autocast("cuda", dtype=torch.bfloat16)
		if on_cuda
		else contextlib.nullcontext()
	)
	non_blocking = on_cuda and pin_memory

	for epoch in tqdm(range(epochs), desc="epochs"):
		print(f"\nEpoch {epoch}\n------------")

		policy.train()
		train_loss = train_mse = train_grip = 0.0

		for above, wrist, state, action in tqdm(train_dataloader, desc=f"train {epoch}", leave=False):
			above = above.to(device, non_blocking=non_blocking)
			wrist = wrist.to(device, non_blocking=non_blocking)
			state = state.to(device, non_blocking=non_blocking)
			action = action.to(device, non_blocking=non_blocking)

			with autocast_ctx:
				pred = policy(above, wrist, state)
				loss, mse, grip = baseline_loss(pred, action, gripper_coeff)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			train_loss += loss.item()
			train_mse += mse.item()
			train_grip += grip.item()

		n_train = len(train_dataloader)
		train_loss /= n_train
		train_mse /= n_train
		train_grip /= n_train

		policy.eval()
		val_loss = val_mse = val_grip = 0.0

		with torch.inference_mode():
			for above, wrist, state, action in val_dataloader:
				above = above.to(device, non_blocking=non_blocking)
				wrist = wrist.to(device, non_blocking=non_blocking)
				state = state.to(device, non_blocking=non_blocking)
				action = action.to(device, non_blocking=non_blocking)

				with autocast_ctx:
					pred = policy(above, wrist, state)
					loss, mse, grip = baseline_loss(pred, action, gripper_coeff)

				val_loss += loss.item()
				val_mse += mse.item()
				val_grip += grip.item()

		n_val = len(val_dataloader)
		val_loss /= n_val
		val_mse /= n_val
		val_grip /= n_val

		writer.add_scalars("loss/total", {"train": train_loss, "val": val_loss}, epoch)
		writer.add_scalars("loss/mse", {"train": train_mse, "val": val_mse}, epoch)
		writer.add_scalars("loss/gripper", {"train": train_grip, "val": val_grip}, epoch)

		ckpt = save_dir / f"epoch_{epoch}_baseline.pth"
		torch.save(policy.state_dict(), ckpt)
		torch.save(policy.state_dict(), save_dir / "baseline_latest.pth")

		print(
			f"train loss={train_loss:.4f} (mse={train_mse:.4f}, grip={train_grip:.4f}) | "
			f"val loss={val_loss:.4f} (mse={val_mse:.4f}, grip={val_grip:.4f})"
		)
		print(f"saved → {ckpt}")

	writer.close()
	print(f"\nDone. logs: {log_dir} | checkpoints: {save_dir}")


if __name__ == "__main__":
	PROJECT_DIR = Path(__file__).resolve().parents[2]

	parser = argparse.ArgumentParser(description="Train behavioral cloning baseline on robot data")
	parser.add_argument("--run_name", type=str, default="baseline_run1",
						help="Subfolder under models/ for checkpoints")
	parser.add_argument("--data_dir", type=Path, default=PROJECT_DIR / "data")
	parser.add_argument("--log_dir", type=Path, default=PROJECT_DIR / "logs")
	parser.add_argument("--save_dir", type=Path, default=PROJECT_DIR / "models")
	parser.add_argument("--lr", type=float, default=3e-4)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--gripper_coeff", type=float, default=0.1)
	parser.add_argument("--epochs", type=int, default=10)
	parser.add_argument("--batch_size", type=int, default=64)
	parser.add_argument("--num_workers", type=int, default=4)
	parser.add_argument("--persistent_workers", action="store_true", default=True)
	parser.add_argument("--no_persistent_workers", action="store_false", dest="persistent_workers")
	parser.add_argument("--pin_memory", action="store_true", default=True)
	parser.add_argument("--no_pin_memory", action="store_false", dest="pin_memory")
	parser.add_argument("--weight_decay", type=float, default=0.05)
	args = parser.parse_args()

	device = "cuda" if torch.cuda.is_available() else "cpu"
	save_dir = args.save_dir / args.run_name
	log_dir = args.log_dir / args.run_name

	train(
		device=device,
		data_dir=args.data_dir,
		log_dir=log_dir,
		save_dir=save_dir,
		lr=args.lr,
		seed=args.seed,
		gripper_coeff=args.gripper_coeff,
		epochs=args.epochs,
		batch_size=args.batch_size,
		num_workers=args.num_workers,
		persistent_workers=args.persistent_workers,
		pin_memory=args.pin_memory,
		optimizer_betas=(0.9, 0.95),
		optimizer_weight_decay=args.weight_decay,
	)
