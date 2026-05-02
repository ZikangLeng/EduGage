from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .loaders import load_session_modality, resample_numeric_segment, slice_time_window

MODELED_MODALITIES = (
    "eeg",
    "ecg",
    "ppg",
    "eda",
    "eye",
    "imu_muse",
    "imu_esense",
    "hr",
    "ring_ppg",
    "ring_temp",
    "ring_imu",
)

MODELED_MODALITY_CHANNELS: dict[str, tuple[str, ...]] = {
    "eeg": ("tp9", "af7", "af8", "tp10"),
    "ecg": ("ecg_val",),
    "ppg": ("ppg1", "ppg2", "ppg3", "ppg4"),
    "eda": ("Resistance_kOhms",),
    "eye": (
        "gaze_conf_int",
        "gaze_por_x",
        "gaze_por_y",
        "head_conf_int",
        "head_pos_x_m",
        "head_pos_y_m",
        "head_pos_z_m",
    ),
    "imu_muse_acc": ("x", "y", "z"),
    "imu_muse_gyro": ("x", "y", "z"),
    "imu_esense": (
        "Accel_X_g",
        "Accel_Y_g",
        "Accel_Z_g",
        "Gyro_X_deg_per_s",
        "Gyro_Y_deg_per_s",
        "Gyro_Z_deg_per_s",
    ),
    "hr": ("HeartRate_bpm", "Quality"),
    "ring_ppg": ("ir", "red", "green"),
    "ring_temp": ("temp_0", "temp_1", "temp_2"),
    "ring_imu": ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"),
}


@dataclass(frozen=True)
class WindowTensorSample:
    window_id: str
    session_key: str
    participant_id: int
    event_id: str
    video_uid: str
    label_5class: int
    label_binary: int
    context_vector: np.ndarray
    modality_arrays: dict[str, np.ndarray]
    modality_mask: np.ndarray


def modality_input_dims() -> dict[str, int]:
    return {
        "eeg": len(MODELED_MODALITY_CHANNELS["eeg"]),
        "ecg": len(MODELED_MODALITY_CHANNELS["ecg"]),
        "ppg": len(MODELED_MODALITY_CHANNELS["ppg"]),
        "eda": len(MODELED_MODALITY_CHANNELS["eda"]),
        "eye": len(MODELED_MODALITY_CHANNELS["eye"]),
        "imu_muse": len(MODELED_MODALITY_CHANNELS["imu_muse_acc"])
        + len(MODELED_MODALITY_CHANNELS["imu_muse_gyro"]),
        "imu_esense": len(MODELED_MODALITY_CHANNELS["imu_esense"]),
        "hr": len(MODELED_MODALITY_CHANNELS["hr"]),
        "ring_ppg": len(MODELED_MODALITY_CHANNELS["ring_ppg"]),
        "ring_temp": len(MODELED_MODALITY_CHANNELS["ring_temp"]),
        "ring_imu": len(MODELED_MODALITY_CHANNELS["ring_imu"]),
    }


def _existing_columns(df: pd.DataFrame, expected: tuple[str, ...]) -> list[str]:
    column_map = {str(col).lower(): str(col) for col in df.columns}
    resolved: list[str] = []
    for name in expected:
        hit = column_map.get(name.lower())
        if hit is not None:
            resolved.append(hit)
    return resolved


def _extract_window_channels_native(
    df: pd.DataFrame,
    t_start_sec: float,
    t_end_sec: float,
    expected_columns: tuple[str, ...],
) -> np.ndarray | None:
    segment_df = slice_time_window(df, t_start_sec=t_start_sec, t_end_sec=t_end_sec)
    if segment_df.empty:
        return None

    feature_df = pd.DataFrame(index=segment_df.index)
    for expected in expected_columns:
        resolved = _existing_columns(segment_df, (expected,))
        if resolved:
            feature_df[expected] = pd.to_numeric(segment_df[resolved[0]], errors="coerce")
        else:
            feature_df[expected] = 0.0
    feature_df = feature_df.interpolate(limit_direction="both").ffill().bfill().fillna(0.0)
    matrix = feature_df[list(expected_columns)].to_numpy(dtype=float)
    if matrix.size == 0:
        return None
    return np.nan_to_num(matrix.T.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def _resample_window_channels(
    df: pd.DataFrame,
    t_start_sec: float,
    t_end_sec: float,
    expected_columns: tuple[str, ...],
    target_points: int,
) -> np.ndarray | None:
    segment_df = slice_time_window(df, t_start_sec=t_start_sec, t_end_sec=t_end_sec)
    if segment_df.empty:
        return None

    feature_df = pd.DataFrame({"t_sec": pd.to_numeric(segment_df["t_sec"], errors="coerce")})
    for expected in expected_columns:
        resolved = _existing_columns(segment_df, (expected,))
        if resolved:
            feature_df[expected] = pd.to_numeric(segment_df[resolved[0]], errors="coerce")
        else:
            feature_df[expected] = 0.0

    matrix = resample_numeric_segment(
        segment_df=feature_df,
        feature_columns=list(expected_columns),
        target_points=target_points,
    )
    if matrix.size == 0:
        return None
    return np.nan_to_num(matrix.T.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def _video_duration_map(window_index: pd.DataFrame) -> dict[tuple[str, str], float]:
    durations: dict[tuple[str, str], float] = {}
    if window_index.empty:
        return durations

    grouped = window_index.groupby(["session_key", "video_uid"], sort=False)
    for (session_key, video_uid), group in grouped:
        end_candidates = pd.to_numeric(group.get("t_end_video_sec"), errors="coerce")
        if end_candidates is None or end_candidates.dropna().empty:
            continue
        duration = float(end_candidates.max())
        if np.isfinite(duration) and duration > 0:
            durations[(str(session_key), str(video_uid))] = duration
    return durations


def build_relative_context_vector(
    window_row: dict[str, Any],
    *,
    video_durations: dict[tuple[str, str], float],
) -> np.ndarray:
    session_key = str(window_row.get("session_key", ""))
    video_uid = str(window_row.get("video_uid", ""))
    duration = float(video_durations.get((session_key, video_uid), np.nan))

    start_v = pd.to_numeric(window_row.get("t_start_video_sec"), errors="coerce")
    end_v = pd.to_numeric(window_row.get("t_end_video_sec"), errors="coerce")

    if not np.isfinite(duration) or duration <= 0 or pd.isna(start_v) or pd.isna(end_v):
        progress_center = 0.0
        span_ratio = 0.0
    else:
        center = 0.5 * (float(start_v) + float(end_v))
        progress_center = float(np.clip(center / duration, 0.0, 1.0))
        span_ratio = float(np.clip((float(end_v) - float(start_v)) / duration, 0.0, 1.0))

    angle = 2.0 * np.pi * progress_center
    return np.asarray(
        [
            progress_center,
            float(np.sin(angle)),
            float(np.cos(angle)),
            span_ratio,
        ],
        dtype=np.float32,
    )


def _load_stream_cached(
    session_row: dict[str, Any],
    base_modality: str,
    cache: dict[tuple[str, str], pd.DataFrame | None],
) -> pd.DataFrame | None:
    session_key = str(session_row.get("session_key", ""))
    cache_key = (session_key, base_modality)
    if cache_key in cache:
        return cache[cache_key]

    if not session_row.get(f"{base_modality}_path"):
        cache[cache_key] = None
        return None

    try:
        cache[cache_key] = load_session_modality(session_row, base_modality)
    except Exception:  # noqa: BLE001
        cache[cache_key] = None
    return cache[cache_key]


def _extract_single_stream_modality(
    session_row: dict[str, Any],
    *,
    base_modality: str,
    expected_columns: tuple[str, ...],
    t_start_sec: float,
    t_end_sec: float,
    use_native_frequency: bool,
    target_points: int,
    cache: dict[tuple[str, str], pd.DataFrame | None],
) -> np.ndarray | None:
    df = _load_stream_cached(session_row, base_modality, cache)
    if df is None:
        return None
    if use_native_frequency:
        return _extract_window_channels_native(
            df,
            t_start_sec=t_start_sec,
            t_end_sec=t_end_sec,
            expected_columns=expected_columns,
        )
    return _resample_window_channels(
        df,
        t_start_sec=t_start_sec,
        t_end_sec=t_end_sec,
        expected_columns=expected_columns,
        target_points=target_points,
    )


def extract_modeled_modality_array(
    session_row: dict[str, Any],
    *,
    modeled_modality: str,
    t_start_sec: float,
    t_end_sec: float,
    use_native_frequency: bool,
    target_points: int,
    cache: dict[tuple[str, str], pd.DataFrame | None],
) -> np.ndarray | None:
    if modeled_modality in {"eeg", "ecg", "ppg", "eda", "eye", "imu_esense", "hr"}:
        expected = MODELED_MODALITY_CHANNELS[modeled_modality]
        return _extract_single_stream_modality(
            session_row,
            base_modality=modeled_modality if modeled_modality != "imu_esense" else "esense",
            expected_columns=expected,
            t_start_sec=t_start_sec,
            t_end_sec=t_end_sec,
            use_native_frequency=use_native_frequency,
            target_points=target_points,
            cache=cache,
        )

    if modeled_modality in {"ring_ppg", "ring_temp", "ring_imu"}:
        expected = MODELED_MODALITY_CHANNELS[modeled_modality]
        return _extract_single_stream_modality(
            session_row,
            base_modality="ring",
            expected_columns=expected,
            t_start_sec=t_start_sec,
            t_end_sec=t_end_sec,
            use_native_frequency=use_native_frequency,
            target_points=target_points,
            cache=cache,
        )

    if modeled_modality == "imu_muse":
        acc = _extract_single_stream_modality(
            session_row,
            base_modality="acc",
            expected_columns=MODELED_MODALITY_CHANNELS["imu_muse_acc"],
            t_start_sec=t_start_sec,
            t_end_sec=t_end_sec,
            use_native_frequency=use_native_frequency,
            target_points=target_points,
            cache=cache,
        )
        gyro = _extract_single_stream_modality(
            session_row,
            base_modality="gyro",
            expected_columns=MODELED_MODALITY_CHANNELS["imu_muse_gyro"],
            t_start_sec=t_start_sec,
            t_end_sec=t_end_sec,
            use_native_frequency=use_native_frequency,
            target_points=target_points,
            cache=cache,
        )
        if acc is None and gyro is None:
            return None
        acc_dim = len(MODELED_MODALITY_CHANNELS["imu_muse_acc"])
        gyro_dim = len(MODELED_MODALITY_CHANNELS["imu_muse_gyro"])
        if use_native_frequency:
            max_len = max(
                acc.shape[1] if acc is not None else 0,
                gyro.shape[1] if gyro is not None else 0,
            )
            if max_len <= 0:
                return None
            if acc is None:
                acc = np.zeros((acc_dim, max_len), dtype=np.float32)
            elif acc.shape[1] < max_len:
                padded = np.zeros((acc_dim, max_len), dtype=np.float32)
                padded[:, : acc.shape[1]] = acc
                acc = padded
            if gyro is None:
                gyro = np.zeros((gyro_dim, max_len), dtype=np.float32)
            elif gyro.shape[1] < max_len:
                padded = np.zeros((gyro_dim, max_len), dtype=np.float32)
                padded[:, : gyro.shape[1]] = gyro
                gyro = padded
            return np.concatenate([acc, gyro], axis=0)
        if acc is None:
            acc = np.zeros((acc_dim, target_points), dtype=np.float32)
        if gyro is None:
            gyro = np.zeros((gyro_dim, target_points), dtype=np.float32)
        return np.concatenate([acc, gyro], axis=0)

    raise ValueError(f"Unsupported modeled modality: {modeled_modality}")


def build_window_tensor_samples(
    *,
    window_index: pd.DataFrame,
    session_manifest: pd.DataFrame,
    use_native_frequency: bool = True,
    target_points: int = 192,
) -> list[WindowTensorSample]:
    if window_index.empty or session_manifest.empty:
        return []

    windows = window_index.copy()
    windows["label_5class"] = pd.to_numeric(
        windows.get("label_5class", windows.get("label_3class")),
        errors="coerce",
    )
    windows["participant_id"] = pd.to_numeric(windows.get("participant_id"), errors="coerce")
    windows["t_start_sec"] = pd.to_numeric(windows.get("t_start_sec"), errors="coerce")
    windows["t_end_sec"] = pd.to_numeric(windows.get("t_end_sec"), errors="coerce")

    windows = windows[
        windows["label_5class"].isin([1, 2, 3, 4, 5])
        & windows["participant_id"].notna()
        & windows["t_start_sec"].notna()
        & windows["t_end_sec"].notna()
        & (windows["t_end_sec"] > windows["t_start_sec"])
    ].copy()
    if windows.empty:
        return []

    session_rows = {
        str(row["session_key"]): row for row in session_manifest.to_dict(orient="records")
    }
    video_durations = _video_duration_map(windows)
    stream_cache: dict[tuple[str, str], pd.DataFrame | None] = {}
    samples: list[WindowTensorSample] = []

    for row in windows.to_dict(orient="records"):
        session_key = str(row.get("session_key", ""))
        session_row = session_rows.get(session_key)
        if session_row is None:
            continue

        modality_arrays: dict[str, np.ndarray] = {}
        modality_mask = np.zeros((len(MODELED_MODALITIES),), dtype=np.float32)

        t_start_sec = float(row["t_start_sec"])
        t_end_sec = float(row["t_end_sec"])

        for idx, modeled_modality in enumerate(MODELED_MODALITIES):
            array = extract_modeled_modality_array(
                session_row,
                modeled_modality=modeled_modality,
                t_start_sec=t_start_sec,
                t_end_sec=t_end_sec,
                use_native_frequency=use_native_frequency,
                target_points=target_points,
                cache=stream_cache,
            )
            if array is None:
                continue
            modality_arrays[modeled_modality] = array
            modality_mask[idx] = 1.0

        if not modality_arrays:
            continue

        label_5class = int(row["label_5class"])
        samples.append(
            WindowTensorSample(
                window_id=str(row.get("window_id", "")),
                session_key=session_key,
                participant_id=int(row["participant_id"]),
                event_id=str(row.get("event_id", "")),
                video_uid=str(row.get("video_uid", "")),
                label_5class=label_5class,
                label_binary=1 if label_5class in {3, 4, 5} else 0,
                context_vector=build_relative_context_vector(
                    row,
                    video_durations=video_durations,
                ),
                modality_arrays=modality_arrays,
                modality_mask=modality_mask,
            )
        )

    return samples
