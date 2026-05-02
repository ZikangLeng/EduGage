"""Build 50 Hz raw tensors for architecture baselines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .train_eval import LABEL_TO_ID


TARGET_HZ = 50.0
TARGET_POINTS = 2200

CHANNELS: tuple[str, ...] = (
    "beam:gaze_por_x",
    "beam:gaze_por_y",
    "beam:head_pos_x_m",
    "beam:head_pos_y_m",
    "beam:head_pos_z_m",
    "bandeda:Resistance_kOhms",
    "bandhr:HeartRate_bpm",
    "museppg:ppg1",
    "ringppg:green",
    "ringtemp:temp_0",
    "ringtemp:temp_1",
    "ringtemp:temp_2",
    "esenseimu:Accel_X_g",
    "esenseimu:Accel_Y_g",
    "esenseimu:Accel_Z_g",
    "esenseimu:Gyro_X_deg_per_s",
    "esenseimu:Gyro_Y_deg_per_s",
    "esenseimu:Gyro_Z_deg_per_s",
    "ecg:ecg_val",
    "museimu:acc_x",
    "museimu:acc_y",
    "museimu:acc_z",
    "museimu:gyro_x",
    "museimu:gyro_y",
    "museimu:gyro_z",
    "museeeg:af7",
    "museeeg:af8",
    "metadata:video_progress_rounded",
)


@dataclass(frozen=True)
class ArchitectureWindowDataset:
    x: np.ndarray
    y: np.ndarray
    metadata: pd.DataFrame
    channels: tuple[str, ...] = CHANNELS
    target_hz: float = TARGET_HZ


def _read_csv(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _find_one(directory: Path, patterns: tuple[str, ...], required_columns: tuple[str, ...] = ()) -> Path | None:
    for pattern in patterns:
        for path in sorted(directory.glob(pattern)):
            if not required_columns:
                return path
            try:
                header = pd.read_csv(path, nrows=0)
            except Exception:  # noqa: BLE001
                continue
            lower = {str(c).lower() for c in header.columns}
            if all(col.lower() in lower for col in required_columns):
                return path
    return None


def _session_paths(windows_path: Path) -> dict[str, Path | None]:
    prefix = windows_path.name.replace("_supervised_windows.csv", "")
    participant_dir = windows_path.parent
    return {
        "beam": participant_dir / f"{prefix}_BeamEyeTracker.csv",
        "bandeda": participant_dir / f"{prefix}_msband_gsr.csv",
        "bandhr": participant_dir / f"{prefix}_msband_hr.csv",
        "museppg": participant_dir / f"{prefix}_PPG.csv",
        "ring": _find_one(
            participant_dir,
            ("*.csv",),
            required_columns=("green", "temp_0", "temp_1", "temp_2"),
        ),
        "esense": participant_dir / f"{prefix}_eSense.csv",
        "ecg": participant_dir / f"{prefix}_Polar_ECG.csv",
        "acc": participant_dir / f"{prefix}_ACC.csv",
        "gyro": participant_dir / f"{prefix}_GYRO.csv",
        "eeg": participant_dir / f"{prefix}_EEG.csv",
    }


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out.columns = [str(c) for c in out.columns]
    return out


def _resample_channel(
    df: pd.DataFrame,
    column: str,
    *,
    start_sec: float,
    end_sec: float,
    target_points: int,
) -> np.ndarray:
    if df.empty or "t_sec" not in df.columns or column not in df.columns:
        return np.zeros((target_points,), dtype=np.float32)

    t = pd.to_numeric(df["t_sec"], errors="coerce").to_numpy(dtype=float)
    v = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(t) & np.isfinite(v) & (t >= start_sec) & (t < end_sec)
    if not np.any(valid):
        return np.zeros((target_points,), dtype=np.float32)

    t = t[valid]
    v = v[valid]
    order = np.argsort(t, kind="mergesort")
    t = t[order]
    v = v[order]
    t, unique_idx = np.unique(t, return_index=True)
    v = v[unique_idx]

    if t.size == 1:
        return np.full((target_points,), float(v[0]), dtype=np.float32)

    grid = np.linspace(start_sec, end_sec, target_points, endpoint=False)
    values = np.interp(grid, t, v, left=float(v[0]), right=float(v[-1]))
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _metadata_progress(row: dict[str, Any], session_windows: pd.DataFrame) -> float:
    required_columns = {"video_uid", "t_end_video_sec"}
    if not required_columns.issubset(set(session_windows.columns)):
        return 0.0
    video_uid = str(row.get("video_uid", ""))
    group = session_windows[session_windows["video_uid"].astype(str) == video_uid]
    if group.empty:
        group = session_windows

    start = pd.to_numeric(row.get("t_start_video_sec"), errors="coerce")
    end = pd.to_numeric(row.get("t_end_video_sec"), errors="coerce")
    video_end = pd.to_numeric(group["t_end_video_sec"], errors="coerce").max()
    if pd.isna(start) or pd.isna(end) or pd.isna(video_end) or float(video_end) <= 0.0:
        return 0.0

    window_center_video_sec = (float(start) + float(end)) / 2.0
    return float(round(window_center_video_sec / float(video_end), 1))


def _window_tensor(
    row: dict[str, Any],
    *,
    session_windows: pd.DataFrame,
    streams: dict[str, pd.DataFrame],
    target_points: int,
) -> np.ndarray:
    start = float(pd.to_numeric(row.get("t_start_sec"), errors="coerce"))
    end = float(pd.to_numeric(row.get("t_end_sec"), errors="coerce"))
    if not np.isfinite(start) or not np.isfinite(end) or end <= start:
        raise ValueError("Window has invalid t_start_sec/t_end_sec bounds.")

    channel_values = [
        _resample_channel(streams["beam"], "gaze_por_x", start_sec=start, end_sec=end, target_points=target_points),
        _resample_channel(streams["beam"], "gaze_por_y", start_sec=start, end_sec=end, target_points=target_points),
        _resample_channel(streams["beam"], "head_pos_x_m", start_sec=start, end_sec=end, target_points=target_points),
        _resample_channel(streams["beam"], "head_pos_y_m", start_sec=start, end_sec=end, target_points=target_points),
        _resample_channel(streams["beam"], "head_pos_z_m", start_sec=start, end_sec=end, target_points=target_points),
        _resample_channel(streams["bandeda"], "Resistance_kOhms", start_sec=start, end_sec=end, target_points=target_points),
        _resample_channel(streams["bandhr"], "HeartRate_bpm", start_sec=start, end_sec=end, target_points=target_points),
        _resample_channel(streams["museppg"], "ppg1", start_sec=start, end_sec=end, target_points=target_points),
        _resample_channel(streams["ring"], "green", start_sec=start, end_sec=end, target_points=target_points),
        _resample_channel(streams["ring"], "temp_0", start_sec=start, end_sec=end, target_points=target_points),
        _resample_channel(streams["ring"], "temp_1", start_sec=start, end_sec=end, target_points=target_points),
        _resample_channel(streams["ring"], "temp_2", start_sec=start, end_sec=end, target_points=target_points),
        _resample_channel(streams["esense"], "Accel_X_g", start_sec=start, end_sec=end, target_points=target_points),
        _resample_channel(streams["esense"], "Accel_Y_g", start_sec=start, end_sec=end, target_points=target_points),
        _resample_channel(streams["esense"], "Accel_Z_g", start_sec=start, end_sec=end, target_points=target_points),
        _resample_channel(streams["esense"], "Gyro_X_deg_per_s", start_sec=start, end_sec=end, target_points=target_points),
        _resample_channel(streams["esense"], "Gyro_Y_deg_per_s", start_sec=start, end_sec=end, target_points=target_points),
        _resample_channel(streams["esense"], "Gyro_Z_deg_per_s", start_sec=start, end_sec=end, target_points=target_points),
        _resample_channel(streams["ecg"], "ecg_val", start_sec=start, end_sec=end, target_points=target_points),
        _resample_channel(streams["acc"], "x", start_sec=start, end_sec=end, target_points=target_points),
        _resample_channel(streams["acc"], "y", start_sec=start, end_sec=end, target_points=target_points),
        _resample_channel(streams["acc"], "z", start_sec=start, end_sec=end, target_points=target_points),
        _resample_channel(streams["gyro"], "x", start_sec=start, end_sec=end, target_points=target_points),
        _resample_channel(streams["gyro"], "y", start_sec=start, end_sec=end, target_points=target_points),
        _resample_channel(streams["gyro"], "z", start_sec=start, end_sec=end, target_points=target_points),
        _resample_channel(streams["eeg"], "af7", start_sec=start, end_sec=end, target_points=target_points),
        _resample_channel(streams["eeg"], "af8", start_sec=start, end_sec=end, target_points=target_points),
    ]
    progress = np.full(
        (target_points,),
        _metadata_progress(row, session_windows=session_windows),
        dtype=np.float32,
    )
    channel_values.append(progress)
    return np.stack(channel_values, axis=0).astype(np.float32)


def build_architecture_dataset_from_preprocessed(
    preprocessed_root: Path,
    *,
    target_points: int = TARGET_POINTS,
) -> ArchitectureWindowDataset:
    x_rows: list[np.ndarray] = []
    y_rows: list[int] = []
    meta_rows: list[dict[str, Any]] = []

    for windows_path in sorted(preprocessed_root.glob("P*/*_supervised_windows.csv")):
        session_windows = pd.read_csv(windows_path)
        if session_windows.empty:
            continue
        paths = _session_paths(windows_path)
        streams = {name: _normalize_columns(_read_csv(path)) for name, path in paths.items()}

        for row in session_windows.to_dict(orient="records"):
            label_value = pd.to_numeric(row.get("label_5class", row.get("label_3class")), errors="coerce")
            if pd.isna(label_value):
                continue
            label_name = str(int(label_value))
            if label_name not in LABEL_TO_ID:
                continue
            try:
                x_rows.append(
                    _window_tensor(
                        row,
                        session_windows=session_windows,
                        streams=streams,
                        target_points=target_points,
                    )
                )
            except ValueError:
                continue
            y_rows.append(LABEL_TO_ID[label_name])
            meta_rows.append(
                {
                    "window_id": str(row.get("window_id", "")),
                    "session_key": str(row.get("session_key", "")),
                    "participant_id": int(pd.to_numeric(row.get("participant_id"), errors="coerce")),
                    "event_id": str(row.get("event_id", "")),
                    "video_uid": str(row.get("video_uid", "")),
                    "label_5class": label_name,
                    "metadata_progress": float(_metadata_progress(row, session_windows=session_windows)),
                }
            )

    if not x_rows:
        raise ValueError(f"No architecture windows found under {preprocessed_root}.")

    return ArchitectureWindowDataset(
        x=np.stack(x_rows, axis=0),
        y=np.asarray(y_rows, dtype=np.int64),
        metadata=pd.DataFrame(meta_rows),
    )
