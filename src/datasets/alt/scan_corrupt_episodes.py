"""
Scan Push-T video episodes for torchcodec decode failures.

Episode-level scanning is used instead of calling __getitem__ on every training
sample (~2M+ indices for train). Each episode only needs the frame indices that
PushTDataset actually reads during training.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from torch.utils.data import DataLoader
from torchcodec.decoders import VideoDecoder
from tqdm.auto import tqdm

SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from datasets.pusht_dataset import PushTDataset


@dataclass
class EpisodeFailure:
    split: str
    episode_idx: int
    episode_path: str
    stage: str
    frame_idx: int | None
    error: str


def split_dir(data_dir: Path, val: bool) -> Path:
    dir_var = "val" if val else "train"
    return data_dir / "pusht_noise" / "pusht_noise" / dir_var


def episode_number(path: Path) -> int:
    match = re.search(r"(\d+)", path.stem)
    if match is None:
        raise ValueError(f"Could not parse episode number from {path.name}")
    return int(match.group(1))


def episode_paths(data_dir: Path, val: bool) -> list[Path]:
    # Episode tensors and seq_lengths.pkl are in numeric episode order, not lexicographic.
    return sorted(split_dir(data_dir, val).joinpath("obses").iterdir(), key=episode_number)


def load_seq_lengths(data_dir: Path, val: bool) -> list[int] | None:
    seq_path = split_dir(data_dir, val) / "seq_lengths.pkl"
    if not seq_path.exists():
        return None
    with open(seq_path, "rb") as f:
        lengths = pickle.load(f)
    return [int(x) for x in lengths]


def frame_indices_used_by_dataset(ep_len: int, frame_skip: int, window: int) -> list[int]:
    indices: set[int] = set()
    start = (window - 1) * frame_skip
    end = ep_len - 1 - frame_skip
    for i in range(start, end):
        for j in reversed(range(window)):
            indices.add(i - (j * frame_skip))
        indices.add(i + frame_skip)
    return sorted(indices)


def first_sample_index_per_episode(dataset: PushTDataset) -> dict[int, int]:
    first_idx: dict[int, int] = {}
    for sample_idx, (ep_idx, _ep_path, _frames) in enumerate(dataset.frame_samples):
        if ep_idx not in first_idx:
            first_idx[ep_idx] = sample_idx
    return first_idx


def scan_episode_frames(
    split: str,
    episode_idx: int,
    episode_path: Path,
    frame_skip: int,
    window: int,
    ep_len: int | None = None,
) -> list[EpisodeFailure]:
    failures: list[EpisodeFailure] = []

    try:
        decoder = VideoDecoder(episode_path)
    except Exception as exc:
        failures.append(
            EpisodeFailure(
                split=split,
                episode_idx=episode_idx,
                episode_path=str(episode_path),
                stage="open_decoder",
                frame_idx=None,
                error=repr(exc),
            )
        )
        return failures

    try:
        decoded_len = len(decoder)
    except Exception as exc:
        failures.append(
            EpisodeFailure(
                split=split,
                episode_idx=episode_idx,
                episode_path=str(episode_path),
                stage="decoder_len",
                frame_idx=None,
                error=repr(exc),
            )
        )
        return failures

    if ep_len is not None and decoded_len != ep_len:
        failures.append(
            EpisodeFailure(
                split=split,
                episode_idx=episode_idx,
                episode_path=str(episode_path),
                stage="length_mismatch",
                frame_idx=None,
                error=f"decoder_len={decoded_len}, seq_lengths={ep_len}",
            )
        )
        return failures

    frame_indices = frame_indices_used_by_dataset(decoded_len, frame_skip, window)
    for frame_idx in frame_indices:
        try:
            _ = decoder[frame_idx]
        except Exception as exc:
            failures.append(
                EpisodeFailure(
                    split=split,
                    episode_idx=episode_idx,
                    episode_path=str(episode_path),
                    stage="decode_frame",
                    frame_idx=frame_idx,
                    error=repr(exc),
                )
            )

    return failures


def scan_split_episodes(
    data_dir: Path,
    val: bool,
    frame_skip: int,
    window: int,
) -> list[EpisodeFailure]:
    split = "val" if val else "train"
    paths = episode_paths(data_dir, val)
    seq_lengths = load_seq_lengths(data_dir, val)

    if seq_lengths is not None and len(seq_lengths) != len(paths):
        raise ValueError(
            f"{split}: {len(paths)} videos but {len(seq_lengths)} seq_lengths entries"
        )

    failures: list[EpisodeFailure] = []
    for episode_idx, episode_path in enumerate(
        tqdm(paths, desc=f"scanning {split} episodes", unit="ep")
    ):
        ep_len = seq_lengths[episode_idx] if seq_lengths is not None else None
        failures.extend(
            scan_episode_frames(
                split=split,
                episode_idx=episode_idx,
                episode_path=episode_path,
                frame_skip=frame_skip,
                window=window,
                ep_len=ep_len,
            )
        )

    return failures


def scan_split_getitem_probe(
    data_dir: Path,
    val: bool,
    frame_skip: int,
    window: int,
) -> list[EpisodeFailure]:
    split = "val" if val else "train"
    dataset = PushTDataset(
        data_dir=data_dir,
        frame_skip=frame_skip,
        window=window,
        val=val,
    )
    first_indices = first_sample_index_per_episode(dataset)

    failures: list[EpisodeFailure] = []
    for episode_idx, sample_idx in tqdm(
        sorted(first_indices.items()),
        desc=f"getitem probe {split}",
        unit="ep",
    ):
        ep_path = dataset.frame_samples[sample_idx][1]
        try:
            _ = dataset[sample_idx]
        except Exception as exc:
            failures.append(
                EpisodeFailure(
                    split=split,
                    episode_idx=episode_idx,
                    episode_path=str(ep_path),
                    stage="dataset_getitem",
                    frame_idx=None,
                    error=repr(exc),
                )
            )

    return failures


def scan_split_dataloader(
    data_dir: Path,
    val: bool,
    frame_skip: int,
    window: int,
    batch_size: int,
    num_workers: int,
    persistent_workers: bool,
    pin_memory: bool,
    max_batches: int | None,
) -> list[EpisodeFailure]:
    split = "val" if val else "train"
    dataset = PushTDataset(
        data_dir=data_dir,
        frame_skip=frame_skip,
        window=window,
        val=val,
    )

    failures: list[EpisodeFailure] = []
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=persistent_workers and num_workers > 0,
        pin_memory=pin_memory,
    )

    total_batches = len(dataloader)
    if max_batches is not None:
        total_batches = min(total_batches, max_batches)

    for batch_idx, _batch in enumerate(
        tqdm(dataloader, total=total_batches, desc=f"dataloader {split}", unit="batch")
    ):
        if max_batches is not None and batch_idx >= max_batches:
            break

    return failures


def scan_split_dataloader_samples(
    data_dir: Path,
    val: bool,
    frame_skip: int,
    window: int,
    num_workers: int,
    persistent_workers: bool,
    pin_memory: bool,
    max_samples: int | None,
) -> list[EpisodeFailure]:
    """Iterate with batch_size=1 so failures map back to a single episode."""
    split = "val" if val else "train"
    dataset = PushTDataset(
        data_dir=data_dir,
        frame_skip=frame_skip,
        window=window,
        val=val,
    )

    failures: list[EpisodeFailure] = []
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=persistent_workers and num_workers > 0,
        pin_memory=pin_memory,
    )

    total_samples = len(dataloader)
    if max_samples is not None:
        total_samples = min(total_samples, max_samples)

    for sample_idx, _batch in enumerate(
        tqdm(dataloader, total=total_samples, desc=f"dataloader {split}", unit="sample")
    ):
        if max_samples is not None and sample_idx >= max_samples:
            break

    return failures


def dedupe_failures(failures: list[EpisodeFailure]) -> list[EpisodeFailure]:
    seen: set[tuple] = set()
    unique: list[EpisodeFailure] = []
    for failure in failures:
        key = (
            failure.split,
            failure.episode_idx,
            failure.stage,
            failure.frame_idx,
            failure.error,
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(failure)
    return unique


def summarize_failures(failures: list[EpisodeFailure]) -> dict:
    by_split: dict[str, set[int]] = defaultdict(set)
    decode_by_split: dict[str, set[int]] = defaultdict(set)
    by_stage: dict[str, int] = defaultdict(int)
    for failure in failures:
        by_split[failure.split].add(failure.episode_idx)
        by_stage[failure.stage] += 1
        if failure.stage in {"open_decoder", "decoder_len", "decode_frame"}:
            decode_by_split[failure.split].add(failure.episode_idx)

    return {
        "total_failure_records": len(failures),
        "corrupt_episode_counts": {split: len(ids) for split, ids in by_split.items()},
        "decode_failure_episode_counts": {
            split: len(ids) for split, ids in decode_by_split.items()
        },
        "failure_stages": dict(by_stage),
        "corrupt_episode_indices": {
            split: sorted(ids) for split, ids in by_split.items()
        },
        "decode_failure_episode_indices": {
            split: sorted(ids) for split, ids in decode_by_split.items()
        },
    }


def write_report(output_path: Path, failures: list[EpisodeFailure], summary: dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": summary,
        "failures": [asdict(f) for f in failures],
    }
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)


def parse_args() -> argparse.Namespace:
    project_dir = SRC_DIR.parent
    parser = argparse.ArgumentParser(
        description="Find Push-T episodes that fail to decode with torchcodec."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=project_dir / "data",
        help="Path to the data directory.",
    )
    parser.add_argument("--frame-skip", type=int, default=5)
    parser.add_argument("--window", type=int, default=3)
    parser.add_argument(
        "--output",
        type=Path,
        default=project_dir / "data" / "corrupt_episodes_report.json",
        help="Where to write the JSON report.",
    )
    parser.add_argument(
        "--probe-getitem",
        action="store_true",
        help="Also run one PushTDataset.__getitem__ call per episode.",
    )
    parser.add_argument(
        "--dataloader",
        action="store_true",
        help="Also smoke-test DataLoaders (slow for train; use --max-batches).",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Limit dataloader batches per split (recommended for train).",
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Scan train split only.",
    )
    parser.add_argument(
        "--val-only",
        action="store_true",
        help="Scan val split only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    splits = [False, True]
    if args.train_only:
        splits = [False]
    if args.val_only:
        splits = [True]

    all_failures: list[EpisodeFailure] = []

    print(
        "Scanning episodes directly (not every dataset sample). "
        "This checks all frame indices used by PushTDataset."
    )

    for val in splits:
        all_failures.extend(
            scan_split_episodes(
                data_dir=args.data_dir,
                val=val,
                frame_skip=args.frame_skip,
                window=args.window,
            )
        )

    if args.probe_getitem:
        for val in splits:
            all_failures.extend(
                scan_split_getitem_probe(
                    data_dir=args.data_dir,
                    val=val,
                    frame_skip=args.frame_skip,
                    window=args.window,
                )
            )

    if args.dataloader:
        for val in splits:
            split = "val" if val else "train"
            try:
                if args.batch_size == 1:
                    scan_split_dataloader_samples(
                        data_dir=args.data_dir,
                        val=val,
                        frame_skip=args.frame_skip,
                        window=args.window,
                        num_workers=args.num_workers,
                        persistent_workers=args.persistent_workers,
                        pin_memory=args.pin_memory,
                        max_samples=args.max_batches,
                    )
                else:
                    scan_split_dataloader(
                        data_dir=args.data_dir,
                        val=val,
                        frame_skip=args.frame_skip,
                        window=args.window,
                        batch_size=args.batch_size,
                        num_workers=args.num_workers,
                        persistent_workers=args.persistent_workers,
                        pin_memory=args.pin_memory,
                        max_batches=args.max_batches,
                    )
            except Exception as exc:
                all_failures.append(
                    EpisodeFailure(
                        split=split,
                        episode_idx=-1,
                        episode_path="",
                        stage="dataloader",
                        frame_idx=None,
                        error=repr(exc),
                    )
                )

    all_failures = dedupe_failures(all_failures)
    summary = summarize_failures(all_failures)
    write_report(args.output, all_failures, summary)

    print(json.dumps(summary, indent=2))
    print(f"Wrote report to {args.output}")

    decode_failures = summary.get("decode_failure_episode_counts", {})
    if decode_failures:
        print(
            "\nDecode failures found. Filter these episode indices in PushTDataset "
            "or re-download/re-encode the listed files."
        )
    elif summary["corrupt_episode_counts"]:
        print(
            "\nOnly metadata mismatches were found (e.g. length_mismatch). "
            "Check episode ordering against rel_actions.pth / seq_lengths.pkl."
        )
    else:
        print("\nNo decode failures found at the episode level.")
        if not args.dataloader:
            print(
                "If training still fails, rerun with --dataloader --num-workers 8 "
                "to check for multiprocessing-specific issues."
            )


if __name__ == "__main__":
    main()
