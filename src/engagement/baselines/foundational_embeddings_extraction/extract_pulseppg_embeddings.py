#!/usr/bin/env python3
"""Extract PulsePPG embeddings for supervised PPG windows.

The main output is a compressed NPZ file containing:
  - window_id: string array of shape [N]
  - embeddings: float32 array of shape [N, 512]

Rows are aligned, so embeddings[i] belongs to window_id[i].
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
        description="Create one PulsePPG embedding per supervised PPG window."
    )
    parser.add_argument(
        "--preprocessed-root",
        type=Path,
        default=REPO_ROOT / "preprocessed_data",
        help="Directory containing P*/<session>_supervised_windows.csv and *_PPG.csv.",
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
        default=REPO_ROOT / "output" / "pulseppg_embeddings",
        help="Directory where embeddings and manifests are written.",
    )
    parser.add_argument("--batch-size", type=int, default=128, help="Inference batch size.")
    parser.add_argument(
        "--target-fs",
        type=float,
        default=50.0,
        help="Sampling rate after resampling. PulsePPG evaluation data commonly uses 50 Hz.",
    )
    parser.add_argument(
        "--target-samples",
        type=int,
        default=None,
        help="Fixed sample count after resampling. Defaults to duration * target-fs.",
    )
    parser.add_argument(
        "--ppg-column",
        default="ppg1",
        help="PPG column to embed, or 'mean' to average ppg1..ppg4 before embedding.",
    )
    parser.add_argument(
        "--normalization",
        choices=["participant", "window", "none"],
        default="participant",
        help=(
            "PPG normalization scope. 'participant' z-scores with stats from the "
            "whole session file, matching PulsePPG's person-specific normalization most closely."
        ),
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
        default="pulseppg_embeddings",
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
        raise ValueError("Cannot infer PPG sampling rate from sample count/duration.")
    return float(sample_count / duration_sec)


def select_ppg_values(segment: pd.DataFrame, ppg_column: str) -> np.ndarray:
    ppg_cols = [col for col in ["ppg1", "ppg2", "ppg3", "ppg4"] if col in segment.columns]
    if ppg_column == "mean":
        if not ppg_cols:
            raise ValueError("No ppg1..ppg4 columns found.")
        return segment[ppg_cols].mean(axis=1).to_numpy(dtype=np.float32)
    if ppg_column not in segment.columns:
        raise ValueError(f"PPG column {ppg_column!r} not found.")
    return segment[ppg_column].to_numpy(dtype=np.float32)


def compute_normalization_stats(ppg_df: pd.DataFrame, ppg_column: str) -> tuple[float, float]:
    values = select_ppg_values(ppg_df.dropna(), ppg_column)
    return float(np.mean(values)), float(np.std(values))


def preprocess_window(
    ppg_df: pd.DataFrame,
    t_start: float,
    t_end: float,
    ppg_column: str,
    target_fs: float,
    target_samples: int | None,
    normalization: str,
    normalization_stats: tuple[float, float] | None,
) -> tuple[np.ndarray, dict[str, float | int | str]]:
    mask = (ppg_df["t_sec"].to_numpy() >= t_start) & (ppg_df["t_sec"].to_numpy() < t_end)
    cols = ["t_sec", *[col for col in ["ppg1", "ppg2", "ppg3", "ppg4"] if col in ppg_df.columns]]
    segment = ppg_df.loc[mask, cols].dropna()
    if segment.empty:
        raise ValueError("No PPG samples in requested window.")

    duration_sec = float(t_end - t_start)
    source_fs = infer_source_fs(len(segment), duration_sec)
    values = select_ppg_values(segment, ppg_column)

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
        "ppg_column": ppg_column,
        "normalization": normalization,
    }
    return values[None, :].astype(np.float32), info


def iter_sessions(preprocessed_root: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for windows_path in sorted(preprocessed_root.glob("P*/*_supervised_windows.csv")):
        prefix = windows_path.name.removesuffix("_supervised_windows.csv")
        ppg_path = windows_path.with_name(f"{prefix}_PPG.csv")
        if ppg_path.exists():
            pairs.append((windows_path, ppg_path))
    return pairs


def read_windows_without_labels(windows_path: Path) -> pd.DataFrame:
    required = {"window_id", "t_start_sec", "t_end_sec"}
    optional = {"participant_id", "session_key", "video_uid", "window_size_sec", "has_ppg"}
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

    for windows_path, ppg_path in iter_sessions(args.preprocessed_root):
        windows = read_windows_without_labels(windows_path)
        if "has_ppg" in windows.columns:
            windows = windows[windows["has_ppg"] == True]  # noqa: E712
        if args.num_shards > 1:
            shard_mask = windows["window_id"].map(
                lambda value: stable_shard(value, args.num_shards) == args.shard_index
            )
            windows = windows[shard_mask]

        ppg_header = pd.read_csv(ppg_path, nrows=0).columns
        ppg_usecols = [col for col in ["t_sec", "ppg1", "ppg2", "ppg3", "ppg4"] if col in ppg_header]
        ppg = pd.read_csv(ppg_path, usecols=ppg_usecols)
        normalization_stats = (
            compute_normalization_stats(ppg, args.ppg_column)
            if args.normalization == "participant"
            else None
        )

        for _, row in windows.iterrows():
            if args.limit is not None and len(all_window_ids) + len(batch) >= args.limit:
                break
            try:
                sample, info = preprocess_window(
                    ppg,
                    float(row["t_start_sec"]),
                    float(row["t_end_sec"]),
                    args.ppg_column,
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
                        "ppg_path": str(ppg_path),
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
                    "ppg_path": str(ppg_path),
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
    meta_path = args.output_dir / f"{args.output_prefix}_metadata.json"

    np.savez_compressed(npz_path, window_id=window_ids, embeddings=embeddings)
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    meta_path.write_text(
        json.dumps(
            {
                "embedding_file": str(npz_path),
                "manifest_file": str(manifest_path),
                "checkpoint": str(args.checkpoint),
                "embedding_shape": list(embeddings.shape),
                "embedding_dtype": str(embeddings.dtype),
                "target_fs": args.target_fs,
                "target_samples": args.target_samples,
                "ppg_column": args.ppg_column,
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
        wide = pd.DataFrame(embeddings, columns=[f"pulseppg_{i:04d}" for i in range(embeddings.shape[1])])
        wide.insert(0, "window_id", window_ids)
        wide.to_csv(args.output_dir / f"{args.output_prefix}_wide.csv", index=False)

    print(f"Wrote {embeddings.shape[0]} embeddings with shape {embeddings.shape} to {npz_path}")
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
