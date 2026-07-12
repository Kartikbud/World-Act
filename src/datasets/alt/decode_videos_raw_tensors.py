import os
from pathlib import Path

import torch
from torchcodec.decoders import VideoDecoder
import time

start = time.time()

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = PROJECT_DIR / "data" / "pusht_noise" / "pusht_noise"

for split in ("train", "val"):
	video_dir = DATA_ROOT / split / "obses"
	out_dir = DATA_ROOT / split / "vid_tensors"
	os.makedirs(out_dir, exist_ok=True)

	for ep in sorted(video_dir.iterdir()):
		out_path = out_dir / f"{ep.stem}.pt"
		if out_path.exists():
			continue

		decoder = VideoDecoder(ep)
		tensor = torch.stack([decoder[i] for i in range(len(decoder))])
		torch.save(tensor, out_path)
		print(f"{split}: {ep.name}")

end = time.time()

print(f"time: {end - start}")