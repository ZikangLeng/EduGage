#!/usr/bin/env python3
"""Extract NormWear embeddings for supervised IMU windows.

The main output is a compressed NPZ file containing:
  - window_id: string array of shape [N]
  - embeddings: float32 array of shape [N, D]

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
FOUNDATIONAL_ROOT = REPO_ROOT / "foundational_models"
NORMWEAR_ROOT = FOUNDATIONAL_ROOT / "NormWear"
if str(FOUNDATIONAL_ROOT) not in sys.path:
    sys.path.insert(0, str(FOUNDATIONAL_ROOT))


IMU_CHANNELS = [
    "acc_x",
    "acc_y",
    "acc_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create one NormWear IMU embedding per supervised window."
    )
    parser.add_argument(
        "--preprocessed-root",
        type=Path,
        default=REPO_ROOT / "preprocessed_data",
        help="Directory containing P*/<session>_supervised_windows.csv and IMU CSVs.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=NORMWEAR_ROOT / "checkpoints" / "normwear_pretrain_ckpt.pth",
        help="NormWear backbone checkpoint path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "output" / "normwear_museimu_embeddings",
        help="Directory where embeddings and manifests are written.",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Inference batch size.")
    parser.add_argument(
        "--sensor-source",
        choices=["muse", "esense"],
        default="muse",
        help="IMU source to embed. muse uses *_ACC.csv + *_GYRO.csv; esense uses *_eSense.csv.",
    )
    parser.add_argument(
        "--target-fs",
        type=float,
        default=65.0,
        help="Sampling rate after explicit resampling. NormWear's wrapper uses 65 Hz internally.",
    )
    parser.add_argument(
        "--target-samples",
        type=int,
        default=None,
        help="Fixed sample count after resampling. Defaults to duration * target-fs.",
    )
    parser.add_argument(
        "--normalization",
        choices=["participant", "window", "none"],
        default="participant",
        help="Normalization scope for each IMU axis. Participant uses the whole session.",
    )
    parser.add_argument(
        "--patch-pooling",
        choices=["mean", "cls"],
        default="mean",
        help="How to pool NormWear's patch dimension.",
    )
    parser.add_argument(
        "--channel-pooling",
        choices=["mean", "flatten"],
        default="mean",
        help="How to combine the six IMU channel embeddings.",
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
        default="normwear_museimu_embeddings",
        help="Output filename prefix. Useful when running multiple shards.",
    )
    parser.add_argument(
        "--write-wide-csv",
        action="store_true",
        help="Also write an embedding-column CSV with one row per window.",
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


def infer_source_fs(sample_count: int, duration_sec: float) -> float:
    if sample_count <= 0 or duration_sec <= 0:
        raise ValueError("Cannot infer IMU sampling rate from sample count/duration.")
    return float(sample_count / duration_sec)


def zscore(values: np.ndarray) -> np.ndarray:
    return ((values - np.mean(values, axis=1, keepdims=True)) / (np.std(values, axis=1, keepdims=True) + 1e-8)).astype(
        np.float32
    )


def apply_zscore(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((values - mean[:, None]) / (std[:, None] + 1e-8)).astype(np.float32)


def iter_sessions(preprocessed_root: Path, sensor_source: str) -> list[tuple[Path, Path, Path | None]]:
    triples: list[tuple[Path, Path, Path | None]] = []
    for windows_path in sorted(preprocessed_root.glob("P*/*_supervised_windows.csv")):
        stem = windows_path.name.replace("_supervised_windows.csv", "")
        if sensor_source == "muse":
            acc_path = windows_path.with_name(f"{stem}_ACC.csv")
            gyro_path = windows_path.with_name(f"{stem}_GYRO.csv")
            if acc_path.exists() and gyro_path.exists():
                triples.append((windows_path, acc_path, gyro_path))
        elif sensor_source == "esense":
            esense_path = windows_path.with_name(f"{stem}_eSense.csv")
            if esense_path.exists():
                triples.append((windows_path, esense_path, None))
        else:
            raise ValueError(f"Unsupported sensor source: {sensor_source}")
    return triples


def read_windows_without_labels(windows_path: Path) -> pd.DataFrame:
    required = {"window_id", "t_start_sec", "t_end_sec"}
    optional = {"participant_id", "session_key", "video_uid", "window_size_sec", "has_acc", "has_gyro", "has_esense"}
    header = pd.read_csv(windows_path, nrows=0).columns
    missing = sorted(required - set(header))
    if missing:
        raise ValueError(f"{windows_path} is missing required columns: {missing}")
    usecols = [col for col in header if col in required or col in optional]
    return pd.read_csv(windows_path, usecols=usecols)


def load_imu(acc_path: Path, gyro_path: Path | None, sensor_source: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if sensor_source == "muse":
        if gyro_path is None:
            raise ValueError("Muse IMU requires separate GYRO path.")
        acc = pd.read_csv(acc_path, usecols=["t_sec", "x", "y", "z"]).rename(
            columns={"x": "acc_x", "y": "acc_y", "z": "acc_z"}
        )
        gyro = pd.read_csv(gyro_path, usecols=["t_sec", "x", "y", "z"]).rename(
            columns={"x": "gyro_x", "y": "gyro_y", "z": "gyro_z"}
        )
    elif sensor_source == "esense":
        cols = [
            "t_sec",
            "Accel_X_g",
            "Accel_Y_g",
            "Accel_Z_g",
            "Gyro_X_deg_per_s",
            "Gyro_Y_deg_per_s",
            "Gyro_Z_deg_per_s",
        ]
        esense = pd.read_csv(acc_path, usecols=cols)
        acc = esense[["t_sec", "Accel_X_g", "Accel_Y_g", "Accel_Z_g"]].rename(
            columns={"Accel_X_g": "acc_x", "Accel_Y_g": "acc_y", "Accel_Z_g": "acc_z"}
        )
        gyro = esense[["t_sec", "Gyro_X_deg_per_s", "Gyro_Y_deg_per_s", "Gyro_Z_deg_per_s"]].rename(
            columns={
                "Gyro_X_deg_per_s": "gyro_x",
                "Gyro_Y_deg_per_s": "gyro_y",
                "Gyro_Z_deg_per_s": "gyro_z",
            }
        )
    else:
        raise ValueError(f"Unsupported sensor source: {sensor_source}")
    acc = acc.dropna().sort_values("t_sec")
    gyro = gyro.dropna().sort_values("t_sec")
    return acc, gyro


def compute_normalization_stats(acc: pd.DataFrame, gyro: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    values = np.vstack(
        [
            acc[["acc_x", "acc_y", "acc_z"]].to_numpy(dtype=np.float32),
            gyro[["gyro_x", "gyro_y", "gyro_z"]].to_numpy(dtype=np.float32),
        ]
    )
    # The stacked array is only for a quick empty check; stats are computed per named axis below.
    if values.size == 0:
        raise ValueError("No IMU samples available for participant/session normalization.")
    means = np.asarray(
        [
            acc["acc_x"].mean(),
            acc["acc_y"].mean(),
            acc["acc_z"].mean(),
            gyro["gyro_x"].mean(),
            gyro["gyro_y"].mean(),
            gyro["gyro_z"].mean(),
        ],
        dtype=np.float32,
    )
    stds = np.asarray(
        [
            acc["acc_x"].std(ddof=0),
            acc["acc_y"].std(ddof=0),
            acc["acc_z"].std(ddof=0),
            gyro["gyro_x"].std(ddof=0),
            gyro["gyro_y"].std(ddof=0),
            gyro["gyro_z"].std(ddof=0),
        ],
        dtype=np.float32,
    )
    return means, stds


def resample_axis(values: np.ndarray, n_samples: int) -> np.ndarray:
    if values.size < 2:
        raise ValueError("Need at least two IMU samples to resample a window.")
    return scipy_signal.resample(values.astype(np.float32), n_samples).astype(np.float32)


def preprocess_window(
    acc: pd.DataFrame,
    gyro: pd.DataFrame,
    t_start: float,
    t_end: float,
    target_fs: float,
    target_samples: int | None,
    normalization: str,
    normalization_stats: tuple[np.ndarray, np.ndarray] | None,
) -> tuple[np.ndarray, dict[str, float | int | str]]:
    acc_segment = acc[(acc["t_sec"].to_numpy() >= t_start) & (acc["t_sec"].to_numpy() < t_end)]
    gyro_segment = gyro[(gyro["t_sec"].to_numpy() >= t_start) & (gyro["t_sec"].to_numpy() < t_end)]
    if acc_segment.empty:
        raise ValueError("No ACC samples in requested window.")
    if gyro_segment.empty:
        raise ValueError("No GYRO samples in requested window.")

    duration_sec = float(t_end - t_start)
    n_samples = target_samples or int(round(duration_sec * target_fs))
    if n_samples <= 0:
        raise ValueError(f"Invalid output sample count {n_samples}.")

    channels = [
        resample_axis(acc_segment["acc_x"].to_numpy(), n_samples),
        resample_axis(acc_segment["acc_y"].to_numpy(), n_samples),
        resample_axis(acc_segment["acc_z"].to_numpy(), n_samples),
        resample_axis(gyro_segment["gyro_x"].to_numpy(), n_samples),
        resample_axis(gyro_segment["gyro_y"].to_numpy(), n_samples),
        resample_axis(gyro_segment["gyro_z"].to_numpy(), n_samples),
    ]
    values = np.stack(channels).astype(np.float32)

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
        "acc_source_samples": int(len(acc_segment)),
        "gyro_source_samples": int(len(gyro_segment)),
        "acc_source_fs_est": infer_source_fs(len(acc_segment), duration_sec),
        "gyro_source_fs_est": infer_source_fs(len(gyro_segment), duration_sec),
        "resampled_samples": int(n_samples),
        "target_fs": float(target_fs),
        "channels": ",".join(IMU_CHANNELS),
        "normalization": normalization,
    }
    return values, info


def load_normwear(checkpoint: Path, device: torch.device) -> torch.nn.Module:
    from NormWear.main_model import NormWearModel

    if not checkpoint.exists():
        raise FileNotFoundError(f"NormWear checkpoint not found: {checkpoint}")
    model = NormWearModel(weight_path="", optimized_cwt=True).to(device)
    checkpoint_obj = torch.load(checkpoint, map_location="cpu")
    state_dict = checkpoint_obj["model"] if isinstance(checkpoint_obj, dict) and "model" in checkpoint_obj else checkpoint_obj
    model.backbone.load_state_dict(state_dict)
    model.eval()
    return model


def pool_normwear_output(features: torch.Tensor, patch_pooling: str, channel_pooling: str) -> torch.Tensor:
    # features: [batch, channels, patches, 768]
    if patch_pooling == "cls":
        pooled = features[:, :, 0, :]
    elif patch_pooling == "mean":
        pooled = features.mean(dim=2)
    else:
        raise ValueError(f"Unsupported patch pooling: {patch_pooling}")

    if channel_pooling == "mean":
        return pooled.mean(dim=1)
    if channel_pooling == "flatten":
        return pooled.flatten(start_dim=1)
    raise ValueError(f"Unsupported channel pooling: {channel_pooling}")


def main() -> None:
    args = parse_args()
    if args.num_shards < 1:
        raise ValueError("--num-shards must be at least 1.")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard-index < num-shards.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device(args.device)
    model = load_normwear(args.checkpoint, device)

    all_window_ids: list[str] = []
    all_embeddings: list[np.ndarray] = []
    manifest_rows: list[dict[str, object]] = []
    batch: list[np.ndarray] = []
    batch_meta: list[dict[str, object]] = []

    def flush_batch() -> None:
        if not batch:
            return
        x = np.stack(batch).astype(np.float32)
        with torch.inference_mode():
            raw_features = model.get_embedding(x, sampling_rate=args.target_fs, device=device)
            features = pool_normwear_output(raw_features, args.patch_pooling, args.channel_pooling)
        features_np = features.detach().cpu().numpy().astype(np.float32)
        for meta, embedding in zip(batch_meta, features_np, strict=True):
            all_window_ids.append(str(meta["window_id"]))
            all_embeddings.append(embedding)
            manifest_rows.append(meta)
        batch.clear()
        batch_meta.clear()

    for windows_path, acc_path, gyro_path in iter_sessions(args.preprocessed_root, args.sensor_source):
        windows = read_windows_without_labels(windows_path)
        if args.sensor_source == "muse":
            if "has_acc" in windows.columns:
                windows = windows[windows["has_acc"] == True]  # noqa: E712
            if "has_gyro" in windows.columns:
                windows = windows[windows["has_gyro"] == True]  # noqa: E712
        elif args.sensor_source == "esense" and "has_esense" in windows.columns:
            windows = windows[windows["has_esense"] == True]  # noqa: E712
        if args.num_shards > 1:
            shard_mask = windows["window_id"].map(
                lambda value: stable_shard(value, args.num_shards) == args.shard_index
            )
            windows = windows[shard_mask]

        acc, gyro = load_imu(acc_path, gyro_path, args.sensor_source)
        normalization_stats = compute_normalization_stats(acc, gyro) if args.normalization == "participant" else None

        for _, row in windows.iterrows():
            if args.limit is not None and len(all_window_ids) + len(batch) >= args.limit:
                break
            try:
                sample, info = preprocess_window(
                    acc,
                    gyro,
                    float(row["t_start_sec"]),
                    float(row["t_end_sec"]),
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
                        "acc_path": str(acc_path),
                        "gyro_path": str(gyro_path) if gyro_path is not None else "",
                        "sensor_source": args.sensor_source,
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
                    "acc_path": str(acc_path),
                    "gyro_path": str(gyro_path) if gyro_path is not None else "",
                    "sensor_source": args.sensor_source,
                    **info,
                }
            )
            if len(batch) >= args.batch_size:
                flush_batch()

        if args.limit is not None and len(all_window_ids) + len(batch) >= args.limit:
            break

    flush_batch()

    embedding_dim = all_embeddings[0].shape[0] if all_embeddings else (4608 if args.channel_pooling == "flatten" else 768)
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
                "normalization": args.normalization,
                "patch_pooling": args.patch_pooling,
                "channel_pooling": args.channel_pooling,
                "sensor_source": args.sensor_source,
                "imu_channels": IMU_CHANNELS,
                "num_shards": args.num_shards,
                "shard_index": args.shard_index,
                "device": str(device),
            },
            indent=2,
        )
        + "\n"
    )

    if args.write_wide_csv:
        wide = pd.DataFrame(embeddings, columns=[f"normwear_imu_{i:04d}" for i in range(embeddings.shape[1])])
        wide.insert(0, "window_id", window_ids)
        wide.to_csv(args.output_dir / f"{args.output_prefix}_wide.csv", index=False)

    print(f"Wrote {embeddings.shape[0]} embeddings with shape {embeddings.shape} to {npz_path}")
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
