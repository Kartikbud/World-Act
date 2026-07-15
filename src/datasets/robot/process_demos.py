"""
Merge per-demo raw HDF5s and convert with robomimic.

Keeps demo_XXX.hdf5 in the raw folder untouched; writes processed demo.hdf5
to data/robot/processed/ (or --out_dir).
"""

import argparse
import subprocess
import sys
from pathlib import Path

import h5py

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RAW_DIR = ROOT / "data" / "robot" / "raw_demo"
DEFAULT_OUT_DIR = ROOT / "data" / "robot" / "processed"


def merge_demos(dataset_dir: Path, out_dir: Path, out_name: str = "demo.hdf5") -> Path:
    files = sorted(dataset_dir.glob("demo_*.hdf5"))
    files = [p for p in files if p.name != out_name]
    if not files:
        raise FileNotFoundError(f"No demo_*.hdf5 files found in {dataset_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / out_name
    with h5py.File(out, "w") as dst:
        dgrp = dst.create_group("data")
        for i, path in enumerate(files):
            with h5py.File(path, "r") as src:
                if i == 0:
                    for k, v in src["data"].attrs.items():
                        dgrp.attrs[k] = v
                src.copy("data/demo_0", dgrp, name=f"demo_{i}")

    print(f"Merged {len(files)} demos → {out}")
    return out


def convert_robosuite(dataset_path: Path) -> None:
    script = (
        ROOT / "robomimic" / "robomimic" / "scripts" / "conversion" / "convert_robosuite.py"
    )
    if not script.exists():
        raise FileNotFoundError(f"convert_robosuite.py not found at {script}")

    cmd = [sys.executable, str(script), "--dataset", str(dataset_path)]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Merge raw demos and convert with robomimic into processed/"
    )
    parser.add_argument(
        "dataset_dir",
        nargs="?",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help=f"Directory with demo_*.hdf5 (default: {DEFAULT_RAW_DIR})",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Where to write merged demo.hdf5 (default: {DEFAULT_OUT_DIR})",
    )
    args = parser.parse_args()
    dataset_dir = args.dataset_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()

    merged = merge_demos(dataset_dir, out_dir)
    convert_robosuite(merged)
    print(f"Done. Raw demos preserved in {dataset_dir}; processed file: {merged}")


if __name__ == "__main__":
    main()
