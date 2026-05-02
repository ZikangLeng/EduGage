#!/usr/bin/env python3
"""Extract MOMENT embeddings for miscellaneous supervised sensor windows.

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
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[4]


@dataclass(frozen=True)
class ModalityConfig:
    output_prefix: str
    output_dir: str
    window_flag: str
    channels: tuple[str, ...]
    units_note: str


MODALITIES: dict[str, ModalityConfig] = {
    "beameye": ModalityConfig(
        output_prefix="moment_beameye_embeddings",
        output_dir="moment_beameye_embeddings",
        window_flag="has_eye",
        channels=(
            "gaze_por_x",
            "gaze_por_y",
            "head_pos_x_m",
            "head_pos_y_m",
            "head_pos_z_m",
        ),
        units_note="BeamEye gaze point x/y plus head position xyz in native export units; confidence and head rotation matrix omitted.",
    ),
    "ringtemp_raw": ModalityConfig(
        output_prefix="moment_ringtemp_raw_embeddings",
        output_dir="moment_ringtemp_raw_embeddings",
        window_flag="has_ring",
        channels=("temp_0", "temp_1", "temp_2"),
        units_note="Raw signed int16 ring temperature channels; physical Celsius scale unresolved.",
    ),
    "msbandeda": ModalityConfig(
        output_prefix="moment_msbandeda_embeddings",
        output_dir="moment_msbandeda_embeddings",
        window_flag="has_eda",
        channels=("eda_log_conductance_uS",),
        units_note="Microsoft Band resistance_kOhms converted to log1p conductance_uS where conductance_uS=1000/resistance_kOhms.",
    ),
    "msbandhr": ModalityConfig(
        output_prefix="moment_msbandhr_embeddings",
        output_dir="moment_msbandhr_embeddings",
        window_flag="has_hr",
        channels=("HeartRate_bpm",),
        units_note="Microsoft Band HeartRate_bpm; defaults to Quality == Locked rows when available.",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create MOMENT embeddings for BeamEye, raw ring temperature, MS Band EDA, or MS Band HR."
    )
    parser.add_argument("--modality", choices=sorted(MODALITIES), required=True)
    parser.add_argument(
        "--preprocessed-root",
        type=Path,
        default=REPO_ROOT / "preprocessed_data",
        help="Directory containing P*/<session>_supervised_windows.csv and modality CSVs.",
    )
    parser.add_argument(
        "--model-id",
        default="AutonLab/MOMENT-1-large",
        help="Hugging Face MOMENT model id.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where embeddings and manifests are written. Defaults to output/<modality preset>.",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Output filename prefix. Defaults to the modality preset prefix.",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Inference batch size.")
    parser.add_argument(
        "--context-length",
        type=int,
        default=512,
        help="Fixed time dimension for MOMENT input.",
    )
    parser.add_argument(
        "--normalization",
        choices=["participant", "window", "none"],
        default="participant",
        help="Normalization scope for each channel.",
    )
    parser.add_argument(
        "--hr-quality",
        choices=["locked", "all"],
        default="locked",
        help="For msbandhr, whether to keep only Quality == Locked rows when available.",
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


def iter_sessions(preprocessed_root: Path, modality: str) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for windows_path in sorted(preprocessed_root.glob("P*/*_supervised_windows.csv")):
        stem = windows_path.name.replace("_supervised_windows.csv", "")
        if modality == "beameye":
            data_path = windows_path.with_name(f"{stem}_BeamEyeTracker.csv")
        elif modality == "ringtemp_raw":
            candidates = sorted(windows_path.parent.glob("*_7.csv"))
            if not candidates:
                continue
            data_path = candidates[0]
        elif modality == "msbandeda":
            data_path = windows_path.with_name(f"{stem}_msband_gsr.csv")
        elif modality == "msbandhr":
            data_path = windows_path.with_name(f"{stem}_msband_hr.csv")
        else:
            raise ValueError(f"Unsupported modality: {modality}")
        if data_path.exists():
            pairs.append((windows_path, data_path))
    return pairs


def read_windows_without_labels(windows_path: Path, flag: str) -> pd.DataFrame:
    required = {"window_id", "t_start_sec", "t_end_sec"}
    optional = {"participant_id", "session_key", "video_uid", "window_size_sec", flag}
    header = pd.read_csv(windows_path, nrows=0).columns
    missing = sorted(required - set(header))
    if missing:
        raise ValueError(f"{windows_path} is missing required columns: {missing}")
    usecols = [col for col in header if col in required or col in optional]
    return pd.read_csv(windows_path, usecols=usecols)


def load_modality_frame(path: Path, modality: str, hr_quality: str) -> pd.DataFrame:
    if modality == "beameye":
        cfg = MODALITIES[modality]
        usecols = ["t_sec", *cfg.channels]
        df = pd.read_csv(path, usecols=usecols)
    elif modality == "ringtemp_raw":
        df = pd.read_csv(path, usecols=["t_sec", "temp_0", "temp_1", "temp_2"])
    elif modality == "msbandeda":
        df = pd.read_csv(path, usecols=["t_sec", "Resistance_kOhms"])
        resistance = pd.to_numeric(df["Resistance_kOhms"], errors="coerce")
        conductance = np.where(resistance > 0, 1000.0 / resistance, np.nan)
        df["eda_log_conductance_uS"] = np.log1p(conductance)
        df = df.drop(columns=["Resistance_kOhms"])
    elif modality == "msbandhr":
        df = pd.read_csv(path, usecols=["t_sec", "HeartRate_bpm", "Quality"])
        if hr_quality == "locked" and "Quality" in df.columns:
            locked = df["Quality"].astype(str).str.lower().eq("locked")
            if locked.any():
                df = df[locked].copy()
        df = df.drop(columns=["Quality"])
    else:
        raise ValueError(f"Unsupported modality: {modality}")

    df["t_sec"] = pd.to_numeric(df["t_sec"], errors="coerce")
    df = df.dropna(subset=["t_sec"]).sort_values("t_sec", kind="mergesort").reset_index(drop=True)
    return df


def compute_normalization_stats(df: pd.DataFrame, channels: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    values = df[list(channels)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    if values.size == 0:
        raise ValueError("No samples available for participant/session normalization.")
    means = np.nanmean(values, axis=0).astype(np.float32)
    stds = np.nanstd(values, axis=0).astype(np.float32)
    return means, stds


def normalize(values: np.ndarray, normalization: str, stats: tuple[np.ndarray, np.ndarray] | None) -> np.ndarray:
    if normalization == "participant":
        if stats is None:
            raise ValueError("Participant normalization requested without stats.")
        mean, std = stats
        return ((values - mean[None, :]) / (std[None, :] + 1e-8)).astype(np.float32)
    if normalization == "window":
        mean = np.nanmean(values, axis=0, keepdims=True)
        std = np.nanstd(values, axis=0, keepdims=True)
        return ((values - mean) / (std + 1e-8)).astype(np.float32)
    if normalization == "none":
        return values.astype(np.float32)
    raise ValueError(f"Unsupported normalization: {normalization}")


def resample_segment(segment: pd.DataFrame, channels: tuple[str, ...], target_points: int) -> np.ndarray:
    if segment.empty:
        raise ValueError("No samples in requested window.")
    values_df = segment[list(channels)].apply(pd.to_numeric, errors="coerce")
    values_df = values_df.interpolate(limit_direction="both").ffill().bfill()
    values = values_df.to_numpy(dtype=np.float32)
    times = pd.to_numeric(segment["t_sec"], errors="coerce").to_numpy(dtype=np.float64)
    valid = np.isfinite(times)
    values = values[valid]
    times = times[valid]
    if len(times) == 0:
        raise ValueError("No finite timestamps in requested window.")
    if len(times) == 1:
        return np.repeat(values[:1], target_points, axis=0).astype(np.float32)

    order = np.argsort(times, kind="mergesort")
    times = times[order]
    values = values[order]
    unique_times, unique_indices = np.unique(times, return_index=True)
    times = unique_times
    values = values[unique_indices]
    if len(times) == 1:
        return np.repeat(values[:1], target_points, axis=0).astype(np.float32)

    target_t = np.linspace(float(times.min()), float(times.max()), target_points)
    resampled = np.stack(
        [np.interp(target_t, times, values[:, idx]) for idx in range(values.shape[1])],
        axis=1,
    )
    return resampled.astype(np.float32)


def preprocess_window(
    df: pd.DataFrame,
    t_start: float,
    t_end: float,
    channels: tuple[str, ...],
    context_length: int,
    normalization: str,
    normalization_stats: tuple[np.ndarray, np.ndarray] | None,
) -> tuple[np.ndarray, dict[str, float | int | str]]:
    segment = df[(df["t_sec"].to_numpy() >= t_start) & (df["t_sec"].to_numpy() < t_end)].copy()
    source_samples = int(len(segment))
    values = resample_segment(segment, channels, context_length)
    values = normalize(values, normalization, normalization_stats)
    duration = float(t_end - t_start)
    info = {
        "source_samples": source_samples,
        "source_fs_est": float(source_samples / duration) if duration > 0 else np.nan,
        "resampled_samples": int(context_length),
        "channels": ",".join(channels),
        "normalization": normalization,
    }
    # MOMENT expects [channels, context_length].
    return values.T.astype(np.float32), info


def load_moment(model_id: str, device: torch.device) -> torch.nn.Module:
    from momentfm import MOMENTPipeline

    model = MOMENTPipeline.from_pretrained(
        model_id,
        model_kwargs={"task_name": "embedding"},
    )
    model.init()
    model = model.to(device)
    model.eval()
    return model


def extract_embeddings(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    out = model(x_enc=x)
    embeddings = getattr(out, "embeddings", None)
    if embeddings is None:
        raise ValueError("MOMENT output did not contain an embeddings attribute.")
    return embeddings


def main() -> None:
    args = parse_args()
    cfg = MODALITIES[args.modality]
    if args.num_shards < 1:
        raise ValueError("--num-shards must be at least 1.")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard-index < num-shards.")
    if args.context_length <= 0:
        raise ValueError("--context-length must be positive.")

    output_dir = args.output_dir or (REPO_ROOT / "output" / cfg.output_dir)
    output_prefix = args.output_prefix or cfg.output_prefix
    output_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device(args.device)
    model = load_moment(args.model_id, device)

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
            features = extract_embeddings(model, x)
        features_np = features.detach().cpu().numpy().astype(np.float32)
        for meta, embedding in zip(batch_meta, features_np, strict=True):
            all_window_ids.append(str(meta["window_id"]))
            all_embeddings.append(embedding)
            manifest_rows.append(meta)
        batch.clear()
        batch_meta.clear()

    for windows_path, data_path in iter_sessions(args.preprocessed_root, args.modality):
        windows = read_windows_without_labels(windows_path, cfg.window_flag)
        if cfg.window_flag in windows.columns:
            windows = windows[windows[cfg.window_flag] == True]  # noqa: E712
        if args.num_shards > 1:
            shard_mask = windows["window_id"].map(
                lambda value: stable_shard(value, args.num_shards) == args.shard_index
            )
            windows = windows[shard_mask]

        data = load_modality_frame(data_path, args.modality, args.hr_quality)
        normalization_stats = (
            compute_normalization_stats(data, cfg.channels) if args.normalization == "participant" else None
        )

        for _, row in windows.iterrows():
            if args.limit is not None and len(all_window_ids) + len(batch) >= args.limit:
                break
            try:
                sample, info = preprocess_window(
                    data,
                    float(row["t_start_sec"]),
                    float(row["t_end_sec"]),
                    cfg.channels,
                    args.context_length,
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
                        "data_path": str(data_path),
                        "modality": args.modality,
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
                    "data_path": str(data_path),
                    "modality": args.modality,
                    **info,
                }
            )
            if len(batch) >= args.batch_size:
                flush_batch()

        if args.limit is not None and len(all_window_ids) + len(batch) >= args.limit:
            break

    flush_batch()

    embedding_dim = all_embeddings[0].shape[0] if all_embeddings else 1024
    embeddings = (
        np.stack(all_embeddings).astype(np.float32)
        if all_embeddings
        else np.empty((0, embedding_dim), dtype=np.float32)
    )
    window_ids = np.asarray(all_window_ids, dtype=str)

    npz_path = output_dir / f"{output_prefix}.npz"
    manifest_path = output_dir / f"{output_prefix}_manifest.csv"
    metadata_path = output_dir / f"{output_prefix}_metadata.json"

    np.savez_compressed(npz_path, window_id=window_ids, embeddings=embeddings)
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    metadata_path.write_text(
        json.dumps(
            {
                "embedding_file": str(npz_path),
                "manifest_file": str(manifest_path),
                "model_id": args.model_id,
                "embedding_shape": list(embeddings.shape),
                "embedding_dtype": str(embeddings.dtype),
                "modality": args.modality,
                "channels": list(cfg.channels),
                "units_note": cfg.units_note,
                "context_length": args.context_length,
                "normalization": args.normalization,
                "hr_quality": args.hr_quality if args.modality == "msbandhr" else None,
                "num_shards": args.num_shards,
                "shard_index": args.shard_index,
                "device": str(device),
            },
            indent=2,
        )
        + "\n"
    )

    if args.write_wide_csv:
        wide = pd.DataFrame(embeddings, columns=[f"{output_prefix}_{i:04d}" for i in range(embeddings.shape[1])])
        wide.insert(0, "window_id", window_ids)
        wide.to_csv(output_dir / f"{output_prefix}_wide.csv", index=False)

    print(f"Wrote {embeddings.shape[0]} embeddings with shape {embeddings.shape} to {npz_path}")
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
