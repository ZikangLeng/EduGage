"""Modality-specific data loaders and segmentation helpers (Phase D)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data_pipeline import detect_and_normalize_timestamps
from .ecg_parser import reconstruct_polar_ecg_timestamps
from .ppg_parser import load_ppg_dataframe
from .ring_parser import load_ring_dataframe
from .timestamp_repair import reconstruct_constant_timestamp_packets


@dataclass(frozen=True)
class ModalitySchema:
    timestamp_candidates: tuple[str, ...]
    required_columns: tuple[str, ...]
    optional_columns: tuple[str, ...] = ()


MODALITY_SCHEMAS: dict[str, ModalitySchema] = {
    "eeg": ModalitySchema(
        timestamp_candidates=("timestamp",),
        required_columns=("tp9", "af7", "af8", "tp10"),
    ),
    "acc": ModalitySchema(
        timestamp_candidates=("timestamp",),
        required_columns=("x", "y", "z"),
    ),
    "gyro": ModalitySchema(
        timestamp_candidates=("timestamp",),
        required_columns=("x", "y", "z"),
    ),
    "ppg": ModalitySchema(
        timestamp_candidates=("timestamp",),
        required_columns=("ppg1", "ppg2", "ppg3"),
        optional_columns=("ppg4",),
    ),
    "ring": ModalitySchema(
        timestamp_candidates=("timestamp",),
        required_columns=("ir", "red", "green", "acc_x", "acc_y", "acc_z"),
        optional_columns=("gyro_x", "gyro_y", "gyro_z", "temp_0", "temp_1", "temp_2"),
    ),
    "ecg": ModalitySchema(
        timestamp_candidates=("sys_time",),
        required_columns=("ecg_val",),
    ),
    "eda": ModalitySchema(
        timestamp_candidates=("SystemTime",),
        required_columns=("Resistance_kOhms",),
    ),
    "hr": ModalitySchema(
        timestamp_candidates=("SystemTime",),
        required_columns=("HeartRate_bpm",),
        optional_columns=("Quality",),
    ),
    "eye": ModalitySchema(
        timestamp_candidates=("sys_time",),
        required_columns=(
            "gaze_conf_int",
            "gaze_por_x",
            "gaze_por_y",
            "head_conf_int",
            "head_pos_x_m",
            "head_pos_y_m",
            "head_pos_z_m",
        ),
    ),
    "markers": ModalitySchema(
        timestamp_candidates=("timestamp",),
        required_columns=("marker_type",),
    ),
    "esense": ModalitySchema(
        timestamp_candidates=("t", "Timestamp", "timestamp"),
        required_columns=(
            "Accel_X_g",
            "Accel_Y_g",
            "Accel_Z_g",
            "Gyro_X_deg_per_s",
            "Gyro_Y_deg_per_s",
            "Gyro_Z_deg_per_s",
        ),
    ),
}


def _resolve_column(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    column_map = {c.lower(): c for c in columns}
    for candidate in candidates:
        if candidate in columns:
            return candidate
        matched = column_map.get(candidate.lower())
        if matched is not None:
            return matched
    return None


def _resolve_required_columns(columns: list[str], required: tuple[str, ...]) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for name in required:
        actual = _resolve_column(columns, (name,))
        if actual is None:
            raise ValueError(
                f"Missing required column '{name}'. Available columns: {columns}"
            )
        resolved[name] = actual
    return resolved


def validate_modality_schema(df: pd.DataFrame, modality: str) -> tuple[str, str]:
    if modality not in MODALITY_SCHEMAS:
        raise ValueError(f"Unsupported modality: {modality}")

    schema = MODALITY_SCHEMAS[modality]
    columns = list(df.columns)
    timestamp_col = _resolve_column(columns, schema.timestamp_candidates)
    if timestamp_col is None:
        raise ValueError(
            f"Missing timestamp column for modality '{modality}'. "
            f"Expected one of {schema.timestamp_candidates}, got: {columns}"
        )

    _resolve_required_columns(columns, schema.required_columns)
    return timestamp_col, modality


def load_modality_csv(path: Path, modality: str) -> pd.DataFrame:
    if modality == "ppg":
        # Muse PPG exports can include up to 7 channels; use only the first 4.
        df = load_ppg_dataframe(path, max_signal_cols=4, use_first_n_signal_cols=4)
    elif modality == "ring":
        df = load_ring_dataframe(path)
    else:
        df = pd.read_csv(path)
    timestamp_col, _ = validate_modality_schema(df, modality)

    normalized_df = df.copy()
    timestamp_series = pd.to_numeric(normalized_df[timestamp_col], errors="coerce")
    if modality == "ecg":
        normalized_df["_timestamp_packet_raw"] = timestamp_series
        timestamp_series = reconstruct_polar_ecg_timestamps(timestamp_series)

    # Generic packet repair for any modality with repeated packet-level timestamps.
    normalized_df["_timestamp_raw"] = reconstruct_constant_timestamp_packets(timestamp_series)
    ts_sec, unit = detect_and_normalize_timestamps(normalized_df["_timestamp_raw"])
    normalized_df["t_sec"] = pd.to_numeric(ts_sec, errors="coerce")
    normalized_df = normalized_df[normalized_df["t_sec"].notna()].copy()
    normalized_df = normalized_df.sort_values("t_sec", kind="mergesort").reset_index(drop=True)

    normalized_df.attrs["modality"] = modality
    normalized_df.attrs["source_path"] = str(path)
    normalized_df.attrs["timestamp_column"] = timestamp_col
    normalized_df.attrs["detected_timestamp_unit"] = unit
    return normalized_df


def load_session_modality(session_row: dict[str, Any], modality: str) -> pd.DataFrame:
    path = session_row.get(f"{modality}_path")
    if not path:
        raise FileNotFoundError(
            f"Session {session_row.get('session_key')} has no path for modality '{modality}'."
        )
    return load_modality_csv(Path(path), modality)


def numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {"t_sec", "_timestamp_raw", "packet_ts_ms"}
    numeric_cols = [
        col
        for col in df.columns
        if col not in excluded and pd.api.types.is_numeric_dtype(df[col])
    ]
    return numeric_cols


def slice_time_window(
    df: pd.DataFrame,
    t_start_sec: float,
    t_end_sec: float,
    boundary_mode: str = "half_open",
) -> pd.DataFrame:
    if boundary_mode != "half_open":
        raise ValueError(f"Unsupported boundary mode: {boundary_mode}")

    if pd.isna(t_start_sec) or pd.isna(t_end_sec):
        return df.iloc[0:0].copy()

    start = float(t_start_sec)
    end = float(t_end_sec)
    if end <= start:
        return df.iloc[0:0].copy()

    mask = (df["t_sec"] >= start) & (df["t_sec"] < end)
    return df.loc[mask].copy()


def resample_numeric_segment(
    segment_df: pd.DataFrame,
    feature_columns: list[str],
    target_points: int,
) -> np.ndarray:
    if target_points <= 0:
        raise ValueError("target_points must be positive.")
    if not feature_columns:
        return np.empty((0, 0), dtype=float)
    if segment_df.empty:
        return np.empty((0, len(feature_columns)), dtype=float)

    numeric = segment_df[feature_columns].apply(pd.to_numeric, errors="coerce").copy()
    numeric = numeric.interpolate(limit_direction="both").ffill().bfill().fillna(0.0)

    values = numeric.to_numpy(dtype=float)
    times = pd.to_numeric(segment_df["t_sec"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(times)
    values = values[valid]
    times = times[valid]
    if len(times) == 0:
        return np.empty((0, len(feature_columns)), dtype=float)

    if len(times) == 1:
        return np.repeat(values[:1], target_points, axis=0)

    order = np.argsort(times, kind="mergesort")
    times = times[order]
    values = values[order]

    unique_times, unique_indices = np.unique(times, return_index=True)
    values = values[unique_indices]
    times = unique_times
    if len(times) == 1:
        return np.repeat(values[:1], target_points, axis=0)

    target_t = np.linspace(float(times.min()), float(times.max()), target_points)
    channels = [
        np.interp(target_t, times, values[:, idx])
        for idx in range(values.shape[1])
    ]
    return np.stack(channels, axis=1)
