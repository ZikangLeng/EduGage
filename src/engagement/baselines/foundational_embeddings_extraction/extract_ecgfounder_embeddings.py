#!/usr/bin/env python3
"""Extract ECGFounder embeddings for supervised ECG windows.

The main output is a compressed NPZ file containing:
  - window_id: string array of shape [N]
  - embeddings: float32 array of shape [N, 1024]

Rows are aligned, so embeddings[i] belongs to window_id[i]. Labels are not read.
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
ECGFOUNDER_ROOT = REPO_ROOT / "foundational_models" / "ECGFounder"
if str(ECGFOUNDER_ROOT) not in sys.path:
    sys.path.insert(0, str(ECGFOUNDER_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create one ECGFounder embedding per supervised ECG window."
    )
    parser.add_argument(
        "--preprocessed-root",
        type=Path,
        default=REPO_ROOT / "preprocessed_data",
        help="Directory containing P*/<session>_supervised_windows.csv and *_Polar_ECG.csv.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ECGFOUNDER_ROOT / "checkpoint" / "1_lead_ECGFounder.pth",
        help="1-lead ECGFounder checkpoint path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "output" / "ecgfounder_embeddings",
        help="Directory where embeddings and manifests are written.",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Inference batch size.")
    parser.add_argument(
        "--target-fs",
        type=float,
        default=500.0,
        help="Sampling rate after resampling. ECGFounder expects 500 Hz ECG.",
    )
    parser.add_argument(
        "--target-samples",
        type=int,
        default=None,
        help="Fixed sample count after resampling. Defaults to duration * target-fs.",
    )
    parser.add_argument(
        "--ecg-column",
        default="ecg_val",
        help="Column in *_Polar_ECG.csv containing the ECG signal.",
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
        default="ecgfounder_embeddings",
        help="Output filename prefix. Useful when running multiple shards.",
    )
    parser.add_argument(
        "--write-wide-csv",
        action="store_true",
        help="Also write a 1024-column CSV with one row per window.",
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


def zscore_window(x: np.ndarray) -> np.ndarray:
    return ((x - np.mean(x)) / (np.std(x) + 1e-8)).astype(np.float32)


def infer_source_fs(sample_count: int, duration_sec: float) -> float:
    if sample_count <= 0 or duration_sec <= 0:
        raise ValueError("Cannot infer ECG sampling rate from sample count/duration.")
    return float(sample_count / duration_sec)


def preprocess_window(
    ecg_df: pd.DataFrame,
    t_start: float,
    t_end: float,
    ecg_column: str,
    target_fs: float,
    target_samples: int | None,
) -> tuple[np.ndarray, dict[str, float | int | str]]:
    mask = (ecg_df["t_sec"].to_numpy() >= t_start) & (ecg_df["t_sec"].to_numpy() < t_end)
    segment = ecg_df.loc[mask, ["t_sec", ecg_column]].dropna()
    if segment.empty:
        raise ValueError("No ECG samples in requested window.")

    duration_sec = float(t_end - t_start)
    source_fs = infer_source_fs(len(segment), duration_sec)
    values = segment[ecg_column].to_numpy(dtype=np.float32)

    n_samples = target_samples or int(round(duration_sec * target_fs))
    if n_samples <= 0:
        raise ValueError(f"Invalid output sample count {n_samples}.")

    values = scipy_signal.resample(values, n_samples)
    values = zscore_window(values)
    info = {
        "source_samples": int(len(segment)),
        "source_fs_est": source_fs,
        "resampled_samples": int(n_samples),
        "ecg_column": ecg_column,
        "normalization": "window_zscore",
    }
    return values[None, :].astype(np.float32), info


def iter_sessions(preprocessed_root: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for windows_path in sorted(preprocessed_root.glob("P*/*_supervised_windows.csv")):
        prefix = windows_path.name.removesuffix("_supervised_windows.csv")
        ecg_path = windows_path.with_name(f"{prefix}_Polar_ECG.csv")
        if ecg_path.exists():
            pairs.append((windows_path, ecg_path))
    return pairs


def read_windows_without_labels(windows_path: Path) -> pd.DataFrame:
    required = {"window_id", "t_start_sec", "t_end_sec"}
    optional = {"participant_id", "session_key", "video_uid", "window_size_sec", "has_ecg"}
    header = pd.read_csv(windows_path, nrows=0).columns
    missing = sorted(required - set(header))
    if missing:
        raise ValueError(f"{windows_path} is missing required columns: {missing}")
    usecols = [col for col in header if col in required or col in optional]
    return pd.read_csv(windows_path, usecols=usecols)


def load_ecgfounder_net(checkpoint: Path, device: torch.device) -> torch.nn.Module:
    from net1d import Net1D

    model = Net1D(
        in_channels=1,
        base_filters=64,
        ratio=1,
        filter_list=[64, 160, 160, 400, 400, 1024, 1024],
        m_blocks_list=[2, 2, 2, 3, 3, 4, 4],
        kernel_size=16,
        stride=2,
        groups_width=16,
        verbose=False,
        use_bn=False,
        use_do=False,
        n_classes=1,
        return_features=True,
    ).to(device)
    checkpoint_obj = torch.load(checkpoint, map_location=device, weights_only=False)
    state_dict = checkpoint_obj["state_dict"]
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith("dense.")}
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    if args.num_shards < 1:
        raise ValueError("--num-shards must be at least 1.")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard-index < num-shards.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device(args.device)
    model = load_ecgfounder_net(args.checkpoint, device)

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
            _, features = model(x)
        features_np = features.detach().cpu().numpy().astype(np.float32)
        for meta, embedding in zip(batch_meta, features_np, strict=True):
            all_window_ids.append(str(meta["window_id"]))
            all_embeddings.append(embedding)
            manifest_rows.append(meta)
        batch.clear()
        batch_meta.clear()

    processed = 0
    for windows_path, ecg_path in iter_sessions(args.preprocessed_root):
        windows = read_windows_without_labels(windows_path)
        ecg_header = pd.read_csv(ecg_path, nrows=0).columns
        if "t_sec" not in ecg_header or args.ecg_column not in ecg_header:
            raise ValueError(f"{ecg_path} must contain t_sec and {args.ecg_column}.")
        ecg_df = pd.read_csv(ecg_path, usecols=["t_sec", args.ecg_column])

        for row in windows.itertuples(index=False):
            window_id = str(row.window_id)
            if stable_shard(window_id, args.num_shards) != args.shard_index:
                continue
            if args.limit is not None and processed >= args.limit:
                break

            meta: dict[str, object] = {
                "window_id": window_id,
                "windows_path": str(windows_path),
                "ecg_path": str(ecg_path),
                "t_start_sec": float(row.t_start_sec),
                "t_end_sec": float(row.t_end_sec),
                "status": "embedded",
            }
            for col in ["participant_id", "session_key", "video_uid", "window_size_sec", "has_ecg"]:
                if hasattr(row, col):
                    meta[col] = getattr(row, col)

            try:
                window, info = preprocess_window(
                    ecg_df=ecg_df,
                    t_start=float(row.t_start_sec),
                    t_end=float(row.t_end_sec),
                    ecg_column=args.ecg_column,
                    target_fs=args.target_fs,
                    target_samples=args.target_samples,
                )
                meta.update(info)
                batch.append(window)
                batch_meta.append(meta)
                processed += 1
                if len(batch) >= args.batch_size:
                    flush_batch()
            except Exception as exc:  # keep a manifest trail without killing long shard runs
                meta["status"] = "error"
                meta["error"] = repr(exc)
                manifest_rows.append(meta)

        if args.limit is not None and processed >= args.limit:
            break

    flush_batch()
    if not all_embeddings:
        raise RuntimeError("No ECG embeddings were created. Check manifest for errors.")

    embeddings = np.stack(all_embeddings).astype(np.float32)
    window_ids = np.asarray(all_window_ids, dtype=str)

    npz_path = args.output_dir / f"{args.output_prefix}.npz"
    manifest_path = args.output_dir / f"{args.output_prefix}_manifest.csv"
    metadata_path = args.output_dir / f"{args.output_prefix}_metadata.json"

    np.savez_compressed(npz_path, window_id=window_ids, embeddings=embeddings)
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    metadata_path.write_text(
        json.dumps(
            {
                "model": "ECGFounder 1-lead",
                "checkpoint": str(args.checkpoint),
                "embedding_dim": int(embeddings.shape[1]),
                "target_fs": args.target_fs,
                "target_samples": args.target_samples,
                "normalization": "window_zscore",
                "num_embeddings": int(embeddings.shape[0]),
                "num_shards": args.num_shards,
                "shard_index": args.shard_index,
            },
            indent=2,
        )
    )

    if args.write_wide_csv:
        wide = pd.DataFrame(embeddings, columns=[f"emb_{i}" for i in range(embeddings.shape[1])])
        wide.insert(0, "window_id", window_ids)
        wide.to_csv(args.output_dir / f"{args.output_prefix}.csv", index=False)

    print(f"Wrote {len(window_ids)} embeddings with shape {embeddings.shape} to {npz_path}")
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
