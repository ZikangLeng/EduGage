#!/usr/bin/env python3
"""Extract NeuroLM embeddings for supervised EEG windows.

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
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import signal as scipy_signal


REPO_ROOT = Path(__file__).resolve().parents[4]
NEUROLM_ROOT = REPO_ROOT / "foundational_models" / "NeuroLM"
if str(NEUROLM_ROOT) not in sys.path:
    sys.path.insert(0, str(NEUROLM_ROOT))


CHANNEL_ALIASES = {
    "af7": "AF7",
    "af8": "AF8",
    "tp9": "TP9",
    "tp10": "TP10",
    "fp1": "FP1",
    "fp2": "FP2",
    "fz": "FZ",
    "cz": "CZ",
    "pz": "PZ",
}


def get_standard_1020() -> list[str]:
    from dataset import standard_1020

    return standard_1020


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create one NeuroLM embedding per supervised EEG window."
    )
    parser.add_argument(
        "--preprocessed-root",
        type=Path,
        default=REPO_ROOT / "preprocessed_data",
        help="Directory containing P*/<session>_supervised_windows.csv and *_EEG.csv.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=NEUROLM_ROOT / "checkpoints" / "NeuroLM-B.pt",
        help="NeuroLM checkpoint path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "output" / "neurolm_embeddings",
        help="Directory where embeddings and manifests are written.",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Inference batch size.")
    parser.add_argument(
        "--source-fs",
        type=float,
        default=256.0,
        help="Original EEG sampling rate. Use <=0 to estimate per window from timestamps.",
    )
    parser.add_argument(
        "--target-fs",
        type=float,
        default=200.0,
        help="Sampling rate after resampling. NeuroLM expects 200 Hz EEG.",
    )
    parser.add_argument(
        "--channels",
        nargs="+",
        default=None,
        help="EEG columns to use. Defaults to all signal columns in *_EEG.csv.",
    )
    parser.add_argument(
        "--filter-low-hz",
        type=float,
        default=0.1,
        help="High-pass cutoff before resampling. Set <=0 to disable.",
    )
    parser.add_argument(
        "--filter-high-hz",
        type=float,
        default=75.0,
        help="Low-pass cutoff before resampling. Clipped below source Nyquist when needed.",
    )
    parser.add_argument(
        "--notch-hz",
        type=float,
        default=60.0,
        help="Line-noise notch frequency before resampling. Set <=0 to disable.",
    )
    parser.add_argument(
        "--scale-divisor",
        type=float,
        default=100.0,
        help="Divide EEG microvolt values by this value, matching NeuroLM loaders.",
    )
    parser.add_argument(
        "--pooling",
        choices=["mean", "last"],
        default="mean",
        help="How to pool final hidden states over valid EEG tokens.",
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
        default="neurolm_embeddings",
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
        raise ValueError("Cannot infer EEG sampling rate from sample count/duration.")
    return float(sample_count / duration_sec)


def map_channel_name(column: str) -> str:
    standard_1020 = get_standard_1020()
    key = column.strip().lower()
    mapped = CHANNEL_ALIASES.get(key, column.strip().upper())
    if mapped not in standard_1020:
        raise ValueError(f"EEG channel {column!r} maps to {mapped!r}, not in NeuroLM channel list.")
    return mapped


def get_channel_indices(channel_names: list[str]) -> np.ndarray:
    standard_1020 = get_standard_1020()
    return np.asarray([standard_1020.index(name) for name in channel_names], dtype=np.int64)


def maybe_filter(values: np.ndarray, source_fs: float, low_hz: float, high_hz: float, notch_hz: float) -> np.ndarray:
    y = values.astype(np.float64, copy=False)
    nyquist = source_fs / 2.0

    if notch_hz > 0 and notch_hz < nyquist:
        b, a = scipy_signal.iirnotch(w0=notch_hz, Q=30.0, fs=source_fs)
        y = scipy_signal.filtfilt(b, a, y, axis=1)

    low = low_hz if low_hz > 0 else None
    high = min(high_hz, nyquist * 0.95) if high_hz > 0 else None
    if low is not None and high is not None and low < high:
        sos = scipy_signal.butter(4, [low, high], btype="bandpass", fs=source_fs, output="sos")
        y = scipy_signal.sosfiltfilt(sos, y, axis=1)
    elif low is not None and low < nyquist:
        sos = scipy_signal.butter(4, low, btype="highpass", fs=source_fs, output="sos")
        y = scipy_signal.sosfiltfilt(sos, y, axis=1)
    elif high is not None and high < nyquist:
        sos = scipy_signal.butter(4, high, btype="lowpass", fs=source_fs, output="sos")
        y = scipy_signal.sosfiltfilt(sos, y, axis=1)

    return y.astype(np.float32)


def preprocess_window(
    eeg_df: pd.DataFrame,
    t_start: float,
    t_end: float,
    channel_columns: list[str],
    channel_names: list[str],
    target_fs: float,
    source_fs_arg: float,
    low_hz: float,
    high_hz: float,
    notch_hz: float,
    scale_divisor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float | int | str]]:
    mask = (eeg_df["t_sec"].to_numpy() >= t_start) & (eeg_df["t_sec"].to_numpy() < t_end)
    segment = eeg_df.loc[mask, ["t_sec", *channel_columns]].dropna()
    if segment.empty:
        raise ValueError("No EEG samples in requested window.")

    duration_sec = float(t_end - t_start)
    source_fs_est = infer_source_fs(len(segment), duration_sec)
    source_fs = float(source_fs_arg) if source_fs_arg > 0 else source_fs_est
    target_samples = int(np.floor(duration_sec * target_fs))
    target_samples -= target_samples % int(target_fs)
    if target_samples <= 0:
        raise ValueError(f"Invalid output sample count {target_samples}.")

    values = segment[channel_columns].to_numpy(dtype=np.float32).T
    values = maybe_filter(values, source_fs, low_hz, high_hz, notch_hz)
    values = scipy_signal.resample(values, target_samples, axis=1).astype(np.float32)
    values = values / float(scale_divisor)

    samples_per_token = int(target_fs)
    n_seconds = values.shape[1] // samples_per_token
    values = values[:, : n_seconds * samples_per_token]
    if n_seconds <= 0:
        raise ValueError("Window is shorter than one NeuroLM token after resampling.")

    n_channels = values.shape[0]
    tokens = values.reshape(n_channels, n_seconds, samples_per_token).transpose(1, 0, 2)
    x_eeg = tokens.reshape(n_seconds * n_channels, samples_per_token).astype(np.float32)

    channel_indices = get_channel_indices(channel_names)
    input_chans = np.tile(channel_indices, n_seconds).astype(np.int64)
    input_time = np.repeat(np.arange(n_seconds, dtype=np.int64), n_channels)
    input_mask = np.ones(x_eeg.shape[0], dtype=bool)
    gpt_mask = build_gpt_mask(n_seconds, n_channels).astype(bool)

    info = {
        "source_samples": int(len(segment)),
        "source_fs_est": float(source_fs_est),
        "source_fs_used": float(source_fs),
        "resampled_samples": int(n_seconds * samples_per_token),
        "num_channels": int(n_channels),
        "num_eeg_tokens": int(x_eeg.shape[0]),
        "channels": ",".join(channel_names),
        "target_fs": float(target_fs),
        "filter_low_hz": float(low_hz),
        "filter_high_hz_requested": float(high_hz),
        "filter_high_hz_effective": float(min(high_hz, source_fs / 2.0 * 0.95) if high_hz > 0 else 0.0),
        "notch_hz": float(notch_hz),
        "scale_divisor": float(scale_divisor),
    }
    return x_eeg, input_chans, input_time, input_mask, gpt_mask, info


def build_gpt_mask(n_seconds: int, n_channels: int) -> np.ndarray:
    n_tokens = n_seconds * n_channels
    mask = np.tril(np.ones((n_tokens, n_tokens), dtype=bool))
    for i in range(n_seconds):
        start = i * n_channels
        end = (i + 1) * n_channels
        mask[start:end, start:end] = True
    return mask[None, :, :]


def pad_batch(items: list[dict[str, np.ndarray]]) -> dict[str, torch.Tensor]:
    standard_1020 = get_standard_1020()
    max_tokens = max(item["x_eeg"].shape[0] for item in items)
    batch_size = len(items)
    x_eeg = np.zeros((batch_size, max_tokens, 200), dtype=np.float32)
    input_chans = np.full((batch_size, max_tokens), standard_1020.index("pad"), dtype=np.int64)
    input_time = np.zeros((batch_size, max_tokens), dtype=np.int64)
    input_mask = np.zeros((batch_size, max_tokens), dtype=bool)
    gpt_mask = np.zeros((batch_size, 1, max_tokens, max_tokens), dtype=bool)

    for i, item in enumerate(items):
        n = item["x_eeg"].shape[0]
        x_eeg[i, :n] = item["x_eeg"]
        input_chans[i, :n] = item["input_chans"]
        input_time[i, :n] = item["input_time"]
        input_mask[i, :n] = item["input_mask"]
        gpt_mask[i, :, :n, :n] = item["gpt_mask"]

    return {
        "x_eeg": torch.from_numpy(x_eeg),
        "input_chans": torch.from_numpy(input_chans),
        "input_time": torch.from_numpy(input_time),
        "input_mask": torch.from_numpy(input_mask),
        "gpt_mask": torch.from_numpy(gpt_mask),
    }


def iter_sessions(preprocessed_root: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for windows_path in sorted(preprocessed_root.glob("P*/*_supervised_windows.csv")):
        prefix = windows_path.name.removesuffix("_supervised_windows.csv")
        eeg_path = windows_path.with_name(f"{prefix}_EEG.csv")
        if eeg_path.exists():
            pairs.append((windows_path, eeg_path))
    return pairs


def read_windows_without_labels(windows_path: Path) -> pd.DataFrame:
    required = {"window_id", "t_start_sec", "t_end_sec"}
    optional = {"participant_id", "session_key", "video_uid", "window_size_sec", "has_eeg"}
    header = pd.read_csv(windows_path, nrows=0).columns
    missing = sorted(required - set(header))
    if missing:
        raise ValueError(f"{windows_path} is missing required columns: {missing}")
    usecols = [col for col in header if col in required or col in optional]
    return pd.read_csv(windows_path, usecols=usecols)


def load_neurolm(checkpoint: Path, device: torch.device) -> torch.nn.Module:
    from model.model import GPTConfig
    from model.model_neurolm import NeuroLM

    checkpoint_obj = torch.load(checkpoint, map_location=device, weights_only=False)
    model_args = checkpoint_obj["model_args"]
    gpt_config = GPTConfig(**{k: model_args[k] for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]})
    model = NeuroLM(gpt_config, init_from="scratch")
    state_dict = checkpoint_obj["model"]
    unwanted_prefix = "_orig_mod."
    for key, value in list(state_dict.items()):
        if key.startswith(unwanted_prefix):
            state_dict[key[len(unwanted_prefix):]] = state_dict.pop(key)
    model.load_state_dict(OrderedDict(state_dict))
    model.eval()
    model.to(device)
    return model


def extract_features(
    model: torch.nn.Module,
    batch_tensors: dict[str, torch.Tensor],
    device: torch.device,
    pooling: str,
) -> np.ndarray:
    x_eeg = batch_tensors["x_eeg"].to(device=device, dtype=torch.float32)
    input_chans = batch_tensors["input_chans"].to(device=device)
    input_time = batch_tensors["input_time"].to(device=device)
    input_mask = batch_tensors["input_mask"].to(device=device)
    gpt_mask = batch_tensors["gpt_mask"].to(device=device)

    with torch.inference_mode():
        tokenizer_mask = input_mask.unsqueeze(1).repeat(1, x_eeg.size(1), 1).unsqueeze(1)
        eeg_tokens = model.tokenizer(
            x_eeg,
            input_chans,
            input_time,
            tokenizer_mask,
            return_all_tokens=True,
        )
        eeg_tokens = model.encode_transform_layer(eeg_tokens)
        eeg_tokens = eeg_tokens + model.pos_embed(input_chans)
        hidden = model.GPT2(
            x_eeg=eeg_tokens,
            x_text=None,
            eeg_time_idx=input_time,
            eeg_mask=gpt_mask,
            lm_head=False,
        )

        if pooling == "mean":
            mask = input_mask.unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        elif pooling == "last":
            lengths = input_mask.long().sum(dim=1).clamp_min(1) - 1
            pooled = hidden[torch.arange(hidden.size(0), device=device), lengths]
        else:
            raise ValueError(f"Unsupported pooling: {pooling}")

    return pooled.detach().cpu().numpy().astype(np.float32)


def main() -> None:
    args = parse_args()
    if args.num_shards < 1:
        raise ValueError("--num-shards must be at least 1.")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard-index < num-shards.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device(args.device)
    model = load_neurolm(args.checkpoint, device)

    all_window_ids: list[str] = []
    all_embeddings: list[np.ndarray] = []
    manifest_rows: list[dict[str, object]] = []
    batch_items: list[dict[str, np.ndarray]] = []
    batch_meta: list[dict[str, object]] = []

    def flush_batch() -> None:
        if not batch_items:
            return
        batch_tensors = pad_batch(batch_items)
        embeddings = extract_features(model, batch_tensors, device, args.pooling)
        for meta, embedding in zip(batch_meta, embeddings, strict=True):
            all_window_ids.append(str(meta["window_id"]))
            all_embeddings.append(embedding)
            manifest_rows.append(meta)
        batch_items.clear()
        batch_meta.clear()

    for windows_path, eeg_path in iter_sessions(args.preprocessed_root):
        windows = read_windows_without_labels(windows_path)
        if "has_eeg" in windows.columns:
            windows = windows[windows["has_eeg"] == True]  # noqa: E712
        if args.num_shards > 1:
            shard_mask = windows["window_id"].map(
                lambda value: stable_shard(value, args.num_shards) == args.shard_index
            )
            windows = windows[shard_mask]

        eeg_header = pd.read_csv(eeg_path, nrows=0).columns
        available_channels = [col for col in eeg_header if col not in {"timestamp", "t_sec"}]
        channel_columns = args.channels or available_channels
        missing_channels = sorted(set(channel_columns) - set(eeg_header))
        if missing_channels:
            raise ValueError(f"{eeg_path} is missing requested channels: {missing_channels}")
        channel_names = [map_channel_name(col) for col in channel_columns]
        eeg_df = pd.read_csv(eeg_path, usecols=["t_sec", *channel_columns])

        for _, row in windows.iterrows():
            if args.limit is not None and len(all_window_ids) + len(batch_items) >= args.limit:
                break

            meta: dict[str, object] = {
                "window_id": row["window_id"],
                "participant_id": row.get("participant_id"),
                "session_key": row.get("session_key"),
                "video_uid": row.get("video_uid"),
                "t_start_sec": row["t_start_sec"],
                "t_end_sec": row["t_end_sec"],
                "window_size_sec": row.get("window_size_sec"),
                "status": "embedded",
                "reason": "",
                "eeg_path": str(eeg_path),
            }
            try:
                x_eeg, input_chans, input_time, input_mask, gpt_mask, info = preprocess_window(
                    eeg_df=eeg_df,
                    t_start=float(row["t_start_sec"]),
                    t_end=float(row["t_end_sec"]),
                    channel_columns=channel_columns,
                    channel_names=channel_names,
                    target_fs=args.target_fs,
                    source_fs_arg=args.source_fs,
                    low_hz=args.filter_low_hz,
                    high_hz=args.filter_high_hz,
                    notch_hz=args.notch_hz,
                    scale_divisor=args.scale_divisor,
                )
            except ValueError as exc:
                meta["status"] = "skipped"
                meta["reason"] = str(exc)
                manifest_rows.append(meta)
                continue

            meta.update(info)
            batch_items.append(
                {
                    "x_eeg": x_eeg,
                    "input_chans": input_chans,
                    "input_time": input_time,
                    "input_mask": input_mask,
                    "gpt_mask": gpt_mask,
                }
            )
            batch_meta.append(meta)
            if len(batch_items) >= args.batch_size:
                flush_batch()

        if args.limit is not None and len(all_window_ids) + len(batch_items) >= args.limit:
            break

    flush_batch()

    embedding_dim = all_embeddings[0].shape[0] if all_embeddings else 768
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
                "source_fs": args.source_fs,
                "channels": args.channels,
                "pooling": args.pooling,
                "filter_low_hz": args.filter_low_hz,
                "filter_high_hz": args.filter_high_hz,
                "notch_hz": args.notch_hz,
                "scale_divisor": args.scale_divisor,
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
