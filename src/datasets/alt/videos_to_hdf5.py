from pathlib import Path

import h5py
import torch
from torchcodec.decoders import VideoDecoder
from tqdm.auto import tqdm

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = PROJECT_DIR / "data" / "pusht_noise" / "pusht_noise"

for split in ("train", "val"):
	video_dir = DATA_ROOT / split / "obses"
	out_path = DATA_ROOT / split / "obs_tensors.h5"
	videos = sorted(p for p in video_dir.iterdir() if p.is_file())

	with h5py.File(out_path, "w") as f:
		for video_path in tqdm(videos, desc=split):
			decoder = VideoDecoder(video_path)
			frames = torch.stack([decoder[i] for i in range(len(decoder))]).numpy()
			_, c, h, w = frames.shape

			grp = f.create_group(video_path.stem)
			dset = grp.create_dataset(
				"frames",
				data=frames,
				chunks=(1, c, h, w),
				shuffle=True,
				compression="gzip",
				compression_opts=4,
			)
			dset.attrs["num_frames"] = frames.shape[0]

	print(f"wrote {out_path}")
