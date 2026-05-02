#!/usr/bin/env python3
"""Extract PulsePPG embeddings for supervised ring PPG windows.

This script uses preprocessed ring CSV files and writes one embedding per
supervised window_id. It defaults to the green optical channel, matching
wearable HR PPG most closely.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import signal as scipy_signal


REPO_ROOT = Path(__file__).resolve().parents[4]
PULSEPPG_ROOT = REPO_ROOT / "foundational_models" / "pulseppg"
if str(PULSEPPG_ROOT) not in sys.path:
    sys.path.insert(0, str(PULSEPPG_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create one PulsePPG embedding per supervised ring PPG window."
    )
    parser.add_argument(
        "--preprocessed-root",
        type=Path,
        default=REPO_ROOT / "preprocessed_data",
        help="Directory containing P*/<session>_supervised_windows.csv.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PULSEPPG_ROOT / "pulseppg" / "experiments" / "out" / "pulseppg" / "checkpoint_best.pkl",
        help="PulsePPG checkpoint_best.pkl path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "output" / "pulseppg_ring_green_embeddings",
        help="Directory where embeddings and manifests are written.",
    )
    parser.add_argument("--batch-size", type=int, default=128, help="Inference batch size.")
    parser.add_argument(
        "--target-fs",
        type=float,
        default=50.0,
        help="Sampling rate after resampling. PulsePPG commonly uses 50 Hz.",
    )
    parser.add_argument(
        "--target-samples",
        type=int,
        default=None,
        help="Fixed sample count after resampling. Defaults to duration * target-fs.",
    )
    parser.add_argument(
        "--ring-channel",
        choices=["green", "red", "ir"],
        default="green",
        help="Ring optical channel to embed.",
    )
    parser.add_argument(
        "--normalization",
        choices=["participant", "window", "none"],
        default="participant",
        help="Normalization scope. Participant uses stats from the whole ring session.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Inference device.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional debug limit.")
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of parallel shards.")
    parser.add_argument("--shard-index", type=int, default=0, help="Zero-based shard index.")
    parser.add_argument(
        "--output-prefix",
        default="pulseppg_ring_green_embeddings",
        help="Output filename prefix. Useful when running multiple shards.",
    )
    parser.add_argument(
        "--write-wide-csv",
        action="store_true",
        help="Also write a 512-column CSV with one row per window.",
    )
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def stable_shard(value: object, num_shards: int) -> int:
    digest = hashlib.sha1(str(value).encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % num_shards


def zscore(x: np.ndarray) -> np.ndarray:
    return ((x - np.mean(x)) / (np.std(x) + 1e-8)).astype(np.float32)


def apply_zscore(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    return ((x - mean) / (std + 1e-8)).astype(np.float32)


def infer_source_fs(sample_count: int, duration_sec: float) -> float:
    if sample_count <= 0 or duration_sec <= 0:
        raise ValueError("Cannot infer ring PPG sampling rate from sample count/duration.")
    return float(sample_count / duration_sec)


def compute_normalization_stats(ring_df: pd.DataFrame, ring_channel: str) -> tuple[float, float]:
    values = ring_df[ring_channel].dropna().to_numpy(dtype=np.float32)
    if values.size == 0:
        raise ValueError(f"No samples available for ring channel {ring_channel!r}.")
    return float(np.mean(values)), float(np.std(values))


def preprocess_window(
    ring_df: pd.DataFrame,
    t_start: float,
    t_end: float,
    ring_channel: str,
    target_fs: float,
    target_samples: int | None,
    normalization: str,
    normalization_stats: tuple[float, float] | None,
) -> tuple[np.ndarray, dict[str, float | int | str]]:
    mask = (ring_df["t_sec"].to_numpy() >= t_start) & (ring_df["t_sec"].to_numpy() < t_end)
    segment = ring_df.loc[mask, ["t_sec", ring_channel]].dropna()
    if segment.empty:
        raise ValueError("No ring PPG samples in requested window.")

    duration_sec = float(t_end - t_start)
    source_fs = infer_source_fs(len(segment), duration_sec)
    values = segment[ring_channel].to_numpy(dtype=np.float32)

    n_samples = target_samples or int(round(duration_sec * target_fs))
    if n_samples <= 0:
        raise ValueError(f"Invalid output sample count {n_samples}.")

    values = scipy_signal.resample(values, n_samples)
    if normalization == "participant":
        if normalization_stats is None:
            raise ValueError("Participant normalization requested without stats.")
        values = apply_zscore(values, *normalization_stats)
    elif normalization == "window":
        values = zscore(values)
    elif normalization == "none":
        values = values.astype(np.float32)
    else:
        raise ValueError(f"Unsupported normalization: {normalization}")

    info = {
        "source_samples": int(len(segment)),
        "source_fs_est": source_fs,
        "resampled_samples": int(n_samples),
        "ring_channel": ring_channel,
        "normalization": normalization,
    }
    return values[None, :].astype(np.float32), info


def iter_sessions(preprocessed_root: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for windows_path in sorted(preprocessed_root.glob("P*/*_supervised_windows.csv")):
        ring_candidates = sorted(windows_path.parent.glob("*_7.csv"))
        if ring_candidates:
            pairs.append((windows_path, ring_candidates[0]))
    return pairs


def read_windows_without_labels(windows_path: Path) -> pd.DataFrame:
    required = {"window_id", "t_start_sec", "t_end_sec"}
    optional = {"participant_id", "session_key", "video_uid", "window_size_sec", "has_ring"}
    header = pd.read_csv(windows_path, nrows=0).columns
    missing = sorted(required - set(header))
    if missing:
        raise ValueError(f"{windows_path} is missing required columns: {missing}")
    usecols = [col for col in header if col in required or col in optional]
    return pd.read_csv(windows_path, usecols=usecols)


def load_pulseppg_net(checkpoint: Path, device: torch.device) -> torch.nn.Module:
    from pulseppg.nets.ResNet1D.ResNet1D_Net import Net

    net = Net(
        in_channels=1,
        base_filters=128,
        kernel_size=11,
        stride=2,
        groups=1,
        n_block=12,
        finalpool="max",
    ).to(device)
    checkpoint_obj = torch.load(checkpoint, map_location=device, weights_only=False)
    net.load_state_dict(checkpoint_obj["net"])
    net.eval()
    return net


def main() -> None:
    args = parse_args()
    if args.num_shards < 1:
        raise ValueError("--num-shards must be at least 1.")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard-index < num-shards.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device(args.device)
    net = load_pulseppg_net(args.checkpoint, device)

    all_window_ids: list[str] = []
    all_embeddings: list[np.ndarray] = []
    manifest_rows: list[dict[str, object]] = []
    batch: list[np.ndarray] = []
    batch_meta: list[dict[str, object]] = []

    def flush_batch() -> None:
        if not batch:
            return
        x = torch.from_numpy(np.stack(batch)).to(device=device, dtype=torch.float32)
        with torch.inference_mode():
            features = net(x)
        features_np = features.detach().cpu().numpy().astype(np.float32)
        for meta, embedding in zip(batch_meta, features_np, strict=True):
            all_window_ids.append(str(meta["window_id"]))
            all_embeddings.append(embedding)
            manifest_rows.append(meta)
        batch.clear()
        batch_meta.clear()

    for windows_path, ring_path in iter_sessions(args.preprocessed_root):
        windows = read_windows_without_labels(windows_path)
        if "has_ring" in windows.columns:
            windows = windows[windows["has_ring"] == True]  # noqa: E712
        if args.num_shards > 1:
            shard_mask = windows["window_id"].map(
                lambda value: stable_shard(value, args.num_shards) == args.shard_index
            )
            windows = windows[shard_mask]

        ring_header = pd.read_csv(ring_path, nrows=0).columns
        required_ring_cols = {"t_sec", args.ring_channel}
        missing_ring_cols = sorted(required_ring_cols - set(ring_header))
        if missing_ring_cols:
            raise ValueError(f"{ring_path} is missing required columns: {missing_ring_cols}")
        ring = pd.read_csv(ring_path, usecols=["t_sec", args.ring_channel])
        if ring.empty:
            for _, row in windows.iterrows():
                manifest_rows.append(
                    {
                        "window_id": row["window_id"],
                        "participant_id": row.get("participant_id"),
                        "session_key": row.get("session_key"),
                        "t_start_sec": row["t_start_sec"],
                        "t_end_sec": row["t_end_sec"],
                        "status": "skipped",
                        "reason": "Ring parser returned no samples.",
                        "ring_path": str(ring_path),
                    }
                )
            continue
        normalization_stats = (
            compute_normalization_stats(ring, args.ring_channel)
            if args.normalization == "participant"
            else None
        )

        for _, row in windows.iterrows():
            if args.limit is not None and len(all_window_ids) + len(batch) >= args.limit:
                break
            try:
                sample, info = preprocess_window(
                    ring,
                    float(row["t_start_sec"]),
                    float(row["t_end_sec"]),
                    args.ring_channel,
                    args.target_fs,
                    args.target_samples,
                    args.normalization,
                    normalization_stats,
                )
            except ValueError as exc:
                manifest_rows.append(
                    {
                        "window_id": row["window_id"],
                        "participant_id": row.get("participant_id"),
                        "session_key": row.get("session_key"),
                        "t_start_sec": row["t_start_sec"],
                        "t_end_sec": row["t_end_sec"],
                        "status": "skipped",
                        "reason": str(exc),
                        "ring_path": str(ring_path),
                    }
                )
                continue

            batch.append(sample)
            batch_meta.append(
                {
                    "window_id": row["window_id"],
                    "participant_id": row.get("participant_id"),
                    "session_key": row.get("session_key"),
                    "video_uid": row.get("video_uid"),
                    "t_start_sec": row["t_start_sec"],
                    "t_end_sec": row["t_end_sec"],
                    "window_size_sec": row.get("window_size_sec"),
                    "status": "embedded",
                    "reason": "",
                    "ring_path": str(ring_path),
                    **info,
                }
            )
            if len(batch) >= args.batch_size:
                flush_batch()

        if args.limit is not None and len(all_window_ids) + len(batch) >= args.limit:
            break

    flush_batch()

    embedding_dim = all_embeddings[0].shape[0] if all_embeddings else 512
    embeddings = (
        np.stack(all_embeddings).astype(np.float32)
        if all_embeddings
        else np.empty((0, embedding_dim), dtype=np.float32)
    )
    window_ids = np.asarray(all_window_ids, dtype=str)

    npz_path = args.output_dir / f"{args.output_prefix}.npz"
    manifest_path = args.output_dir / f"{args.output_prefix}_manifest.csv"
    metadata_path = args.output_dir / f"{args.output_prefix}_metadata.json"

    np.savez_compressed(npz_path, window_id=window_ids, embeddings=embeddings)
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    metadata_path.write_text(
        json.dumps(
            {
                "embedding_file": str(npz_path),
                "manifest_file": str(manifest_path),
                "checkpoint": str(args.checkpoint),
                "embedding_shape": list(embeddings.shape),
                "embedding_dtype": str(embeddings.dtype),
                "target_fs": args.target_fs,
                "target_samples": args.target_samples,
                "ring_channel": args.ring_channel,
                "normalization": args.normalization,
                "num_shards": args.num_shards,
                "shard_index": args.shard_index,
                "device": str(device),
            },
            indent=2,
        )
        + "\n"
    )

    if args.write_wide_csv:
        wide = pd.DataFrame(embeddings, columns=[f"pulseppg_ring_{i:04d}" for i in range(embeddings.shape[1])])
        wide.insert(0, "window_id", window_ids)
        wide.to_csv(args.output_dir / f"{args.output_prefix}_wide.csv", index=False)

    print(f"Wrote {embeddings.shape[0]} embeddings with shape {embeddings.shape} to {npz_path}")
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
