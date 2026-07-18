import contextlib

import torch


def resolve_amp_dtype(amp_dtype: str, device) -> torch.dtype | None:
	"""Resolve autocast dtype from config.

	Supported values:
	  - auto: bf16 on GPUs with native BF16 support (Ampere+), else fp16 on CUDA
	  - bf16 / bfloat16
	  - fp16 / float16
	  - off / none: disable autocast
	"""
	if isinstance(device, str):
		on_cuda = device.startswith("cuda")
	else:
		on_cuda = device.type == "cuda"

	if not on_cuda:
		return None

	key = amp_dtype.lower()
	if key == "auto":
		return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
	if key in ("bf16", "bfloat16"):
		return torch.bfloat16
	if key in ("fp16", "float16"):
		return torch.float16
	if key in ("off", "none"):
		return None
	raise ValueError(
		f"Unknown amp_dtype {amp_dtype!r}. Use auto, bf16, fp16, or off."
	)


def autocast_context(device, amp_dtype: str):
	"""Return a torch.amp.autocast context (or nullcontext on CPU / off)."""
	dtype = resolve_amp_dtype(amp_dtype, device)
	if dtype is None:
		return contextlib.nullcontext()
	return torch.amp.autocast("cuda", dtype=dtype)


def amp_dtype_label(amp_dtype: str, device) -> str:
	"""Human-readable label for logging which AMP mode is active."""
	resolved = resolve_amp_dtype(amp_dtype, device)
	if resolved is None:
		return "off"
	return "bf16" if resolved is torch.bfloat16 else "fp16"
