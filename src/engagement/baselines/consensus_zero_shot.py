
"""Phase Z1 ConSensus-style engagement baseline for 1..5 regression."""

from __future__ import annotations

import asyncio
import json
import logging
import math
from pathlib import Path
import re
import sys
import threading
from typing import Any
import warnings
from contextlib import contextmanager

import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq
from scipy.signal import butter, filtfilt, find_peaks, welch
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from ..config import RunConfig, validate_baseline_config, validate_core_config
from ..io_utils import read_table_with_fallback, write_table_with_fallback
from ..preprocessed_data import run_phase_preprocessed_data

LOGGER = logging.getLogger(__name__)
LABEL_ORDER = ("1", "2", "3", "4", "5")

STUDY_PROMPT_TEXT = (
    "How difficult was it to pay attention during the last minute of the lecture? (1-5)\n"
    "1: My attention is fused with the lecture; it is completely automatic and effortless.\n"
    "2\n"
    "3\n"
    "4\n"
    "5: I am forcing my attention. It feels like a heavy, conscious struggle to keep up with the lesson.\n"
    "X: External Distraction/Interruption - Use this if something outside the lecture forced your attention away entirely."
)

DEVICE_INFO = {
    "muse_headband": "Muse headband worn on the participant forehead near the middle of the forehead. Collects ACC, GYRO, EEG, and PPG.",
    "esense_left_earbud": "eSense earbud worn on the left ear. Collects acceleration and gyroscope data.",
    "polar_ecg_cheststrap": "Polar ECG chest strap near the participant chest. Collects one-lead ECG.",
    "msband_right_wrist": "MS Band worn on the right wrist. Collects EDA and heart-rate data.",
    "ring_right_finger": "Ring worn on a finger of the right hand. Records 3-channel PPG (IR/RED/GREEN), temperature, and 3-axis acceleration.",
    "beam_eye_tracker": "Eye tracking and head-motion device near the display for gaze/head signals.",
    "unknown_device": "Device metadata missing in export summary.",
}

MODALITY_DEFAULT_DEVICES = {
    "eeg": ["muse_headband"],
    "acc": ["muse_headband"],
    "gyro": ["muse_headband"],
    "ppg": ["muse_headband"],
    "ecg": ["polar_ecg_cheststrap"],
    "eda": ["msband_right_wrist"],
    "hr": ["msband_right_wrist"],
    "ring": ["ring_right_finger"],
    "esense": ["esense_left_earbud"],
    "eye": ["beam_eye_tracker"],
}

STREAM_UNIT_SPECS = {
    "muse_eeg": {
        "coarse_modality": "eeg",
        "sensor_modality": "eeg",
        "device_id": "muse_headband",
        "data_collection": "EEG cortical dynamics sampled from the Muse headband on the forehead.",
        "feature_extraction": "Window-level summary statistics over EEG channels and temporal drift.",
        "measurement_notes": "Raw columns: tp9, af7, af8, tp10. Physical unit is not explicitly declared in exported CSV; values should be treated as device-native EEG amplitude units.",
        "column_patterns": ("tp9", "af7", "af8", "tp10"),
    },
    "muse_acc": {
        "coarse_modality": "acc",
        "sensor_modality": "accelerometer",
        "device_id": "muse_headband",
        "data_collection": "Tri-axial accelerometer from the Muse headband capturing head movement intensity.",
        "feature_extraction": "Window-level distribution and motion variability statistics.",
        "measurement_notes": "Raw columns: x, y, z from Muse ACC. Unit is not explicitly declared in CSV; numeric range appears g-like and should be treated as device-native acceleration scale.",
        "column_patterns": ("x", "y", "z"),
    },
    "muse_gyro": {
        "coarse_modality": "gyro",
        "sensor_modality": "gyroscope",
        "device_id": "muse_headband",
        "data_collection": "Tri-axial gyroscope from the Muse headband capturing rotational head movement.",
        "feature_extraction": "Window-level rotational variability and trend statistics.",
        "measurement_notes": "Raw columns: x, y, z from Muse GYRO. Unit is not explicitly declared in CSV; treat as device-native angular-rate scale (commonly deg/s in Muse exports).",
        "column_patterns": ("x", "y", "z"),
    },
    "muse_ppg": {
        "coarse_modality": "ppg",
        "sensor_modality": "photoplethysmography",
        "device_id": "muse_headband",
        "data_collection": "Optical PPG channels from the Muse headband near the forehead.",
        "feature_extraction": "Window-level waveform distribution and temporal-change statistics.",
        "measurement_notes": "Raw columns: ppg1, ppg2, ppg3 (and optional ppg4). Values are optical-channel device readings (device units / ADC-like counts), not calibrated physiological units.",
        "column_patterns": ("ppg1", "ppg2", "ppg3", "ppg4"),
    },
    "polar_ecg": {
        "coarse_modality": "ecg",
        "sensor_modality": "electrocardiography",
        "device_id": "polar_ecg_cheststrap",
        "data_collection": "One-lead ECG from the Polar chest strap near the participant chest.",
        "feature_extraction": "Window-level ECG amplitude distribution and change statistics.",
        "measurement_notes": "Raw column: ecg_val from Polar ECG CSV. Pipeline reconstructs timestamps but does not apply amplitude calibration; ecg_val remains device-native integer amplitude.",
        "column_patterns": ("ecg_val",),
    },
    "msband_eda": {
        "coarse_modality": "eda",
        "sensor_modality": "electrodermal_activity",
        "device_id": "msband_right_wrist",
        "data_collection": "Electrodermal activity from the MS Band on the right wrist.",
        "feature_extraction": "Window-level conductance level, spread, and slope statistics.",
        "measurement_notes": "Raw column: Resistance_kOhms. Unit is kilo-ohms (kOhms).",
        "column_patterns": ("Resistance_kOhms",),
    },
    "beam_eye": {
        "coarse_modality": "eye",
        "sensor_modality": "eye_head_tracking",
        "device_id": "beam_eye_tracker",
        "data_collection": "Eye and head tracking stream from the Beam eye tracker near the display.",
        "feature_extraction": "Window-level gaze/head variability and drift summaries.",
        "measurement_notes": "Raw columns include gaze_por_x/y (screen or pixel coordinates), head_pos_x_m/head_pos_y_m/head_pos_z_m (meters), and confidence integers.",
        "column_patterns": (),
    },
    "esense_acc": {
        "coarse_modality": "esense",
        "sensor_modality": "accelerometer",
        "device_id": "esense_left_earbud",
        "data_collection": "Tri-axial accelerometer from the eSense earbud on the left ear.",
        "feature_extraction": "Window-level motion distribution and temporal-change summaries.",
        "measurement_notes": "Raw columns: Accel_X_g, Accel_Y_g, Accel_Z_g. Unit is g.",
        "column_patterns": ("Accel_",),
    },
    "esense_gyro": {
        "coarse_modality": "esense",
        "sensor_modality": "gyroscope",
        "device_id": "esense_left_earbud",
        "data_collection": "Tri-axial gyroscope from the eSense earbud on the left ear.",
        "feature_extraction": "Window-level rotational variability and temporal-change summaries.",
        "measurement_notes": "Raw columns: Gyro_X_deg_per_s, Gyro_Y_deg_per_s, Gyro_Z_deg_per_s. Unit is degrees per second (deg/s).",
        "column_patterns": ("Gyro_",),
    },
    "ring_ppg": {
        "coarse_modality": "ring",
        "sensor_modality": "photoplethysmography",
        "device_id": "ring_right_finger",
        "data_collection": "Optical PPG channels from the ring on the right hand finger.",
        "feature_extraction": "Window-level summary statistics over ring optical channels.",
        "measurement_notes": "Raw columns: ir, red, green. Parser preserves raw counts and does not apply vendor calibration.",
        "column_patterns": ("ir", "red", "green"),
    },
    "ring_acc": {
        "coarse_modality": "ring",
        "sensor_modality": "accelerometer",
        "device_id": "ring_right_finger",
        "data_collection": "Tri-axial acceleration stream from the ring on the right hand finger.",
        "feature_extraction": "Window-level motion distribution and temporal-change summaries.",
        "measurement_notes": "Raw columns: acc_x, acc_y, acc_z. Parser preserves raw device counts and does not apply vendor calibration.",
        "column_patterns": ("acc_x", "acc_y", "acc_z"),
    },
}

STREAM_UNIT_BY_MODALITY_DEVICE = {
    ("eeg", "muse_headband"): ("muse_eeg",),
    ("acc", "muse_headband"): ("muse_acc",),
    ("gyro", "muse_headband"): ("muse_gyro",),
    ("ppg", "muse_headband"): ("muse_ppg",),
    ("ecg", "polar_ecg_cheststrap"): ("polar_ecg",),
    ("eda", "msband_right_wrist"): ("msband_eda",),
    ("eye", "beam_eye_tracker"): ("beam_eye",),
    ("esense", "esense_left_earbud"): ("esense_acc", "esense_gyro"),
    ("ring", "ring_right_finger"): ("ring_ppg", "ring_acc"),
}

GENERIC_FEATURE_GUIDE = (
    "Window summary features: num_samples, num_channels, value_mean, value_std, "
    "value_p10, value_p90, value_iqr, abs_slope."
)

DISABLED_BASELINE_MODALITIES = {"hr"}


class _StageProgressBar:
    def __init__(
        self,
        stage: str,
        total: int,
        *,
        logger: logging.Logger | None = None,
        width: int = 24,
    ) -> None:
        self.stage = str(stage)
        self.total = max(0, int(total))
        self.logger = logger or LOGGER
        self.width = max(8, int(width))
        self.current = 0
        self._last_logged = -1
        self._log_every = max(1, int(math.ceil(self.total / 50))) if self.total > 0 else 1
        self._log(force=True)

    def _bar(self) -> str:
        if self.total <= 0:
            return "#" * self.width
        ratio = min(1.0, max(0.0, float(self.current) / float(self.total)))
        filled = int(round(ratio * self.width))
        return "#" * filled + "-" * (self.width - filled)

    def _log(self, *, force: bool = False, context: str | None = None) -> None:
        if not force:
            if self.current == self._last_logged:
                return
            if self.current != self.total and (self.current - self._last_logged) < self._log_every:
                return
        if self.total > 0:
            pct = 100.0 * float(self.current) / float(self.total)
            message = (
                f"[Progress][{self.stage}] [{self._bar()}] "
                f"{self.current}/{self.total} ({pct:5.1f}%)"
            )
        else:
            message = f"[Progress][{self.stage}] [{self._bar()}] {self.current}"
        if context:
            message += f" {context}"
        self.logger.info(message)
        self._last_logged = self.current

    def update(self, step: int = 1, *, context: str | None = None, force: bool = False) -> None:
        self.current = min(self.total, self.current + int(step)) if self.total > 0 else self.current + int(step)
        self._log(force=force, context=context)

    def close(self, *, context: str | None = None) -> None:
        if context is None and self.current == self._last_logged:
            return
        self._log(force=True, context=context)


def _as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None:
        return False
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}


def _coerce_label_1_to_5(value: Any) -> int | None:
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        return None
    rounded = int(round(float(number)))
    if abs(float(number) - rounded) > 1e-6:
        return None
    return rounded if 1 <= rounded <= 5 else None


def _coerce_answer_1_to_5(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    match = re.search(r"([1-5])", text)
    if match:
        return int(match.group(1))
    return None


INVALID_PATH_CHARS_RE = re.compile(r'[<>:"/\\|?*\x00-\x1F]+')


def _safe_path_component(value: Any, fallback: str = "unknown_window") -> str:
    text = str(value).strip() if value is not None else ""
    if not text:
        return fallback
    text = INVALID_PATH_CHARS_RE.sub("_", text)
    text = re.sub(r"\s+", "_", text)
    text = text.strip(" ._")
    if not text:
        return fallback
    return text[:120]


def _normalize_device_id(value: Any) -> str:
    text = str(value).strip().lower() if value is not None else ""
    if not text:
        return ""
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    if not text:
        return ""
    aliases = {
        "muse": "muse_headband",
        "muse_headband": "muse_headband",
        "esense": "esense_left_earbud",
        "polar": "polar_ecg_cheststrap",
        "polar_chest_strap": "polar_ecg_cheststrap",
        "polar_ecg": "polar_ecg_cheststrap",
        "msband": "msband_right_wrist",
        "ring": "ring_right_finger",
        "beam_eye_tracker": "beam_eye_tracker",
        "eye_tracker": "beam_eye_tracker",
    }
    return aliases.get(text, text)


def _infer_device_id(modality: str, source_path: str, row: dict[str, Any]) -> str:
    for key in ("device", "device_name", "source_device", "device_id"):
        if key in row and str(row.get(key, "")).strip():
            normalized = _normalize_device_id(row.get(key))
            if normalized:
                return normalized

    src = str(source_path).lower()
    if "esense" in src:
        return "esense_left_earbud"
    if "polar" in src or "ecg" in src:
        if modality == "ecg":
            return "polar_ecg_cheststrap"
    if "msband" in src or "_gsr" in src or "_hr" in src:
        if modality in {"eda", "hr"}:
            return "msband_right_wrist"
    if "ring" in src:
        return "ring_right_finger"
    if "beameyetracker" in src or "eyetracker" in src:
        return "beam_eye_tracker"

    defaults = MODALITY_DEFAULT_DEVICES.get(modality)
    if defaults:
        return defaults[0]
    return "unknown_device"


def _sanitize_feature_token(value: Any) -> str:
    text = str(value).strip().lower() if value is not None else ""
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text or "unknown"


def _stream_feature_keys(record: dict[str, Any], stream_name: str, max_items: int = 64) -> list[str]:
    features = record.get("features", {})
    if not isinstance(features, dict):
        return []

    keys: list[str] = []
    prefix_a = f"{stream_name}__"
    prefix_b = f"{stream_name}_"
    for key in sorted(features.keys()):
        key_s = str(key)
        if key_s.startswith(prefix_a) or key_s.startswith(prefix_b):
            keys.append(key_s)
        elif f"__{stream_name}__" in key_s:
            keys.append(key_s)

    return keys[:max_items]


def _feature_name_from_key(key: str, stream_name: str) -> str:
    key_s = str(key)
    prefix = f"{stream_name}__"
    if key_s.startswith(prefix):
        remainder = key_s[len(prefix):]
        if "__" in remainder:
            _, feature_name = remainder.split("__", 1)
            return feature_name or key_s
    return key_s


def _friendly_feature_names(feature_keys: list[str], stream_name: str) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for key in feature_keys:
        name = _feature_name_from_key(key, stream_name)
        if name in seen:
            continue
        seen.add(name)
        names.append(name)
    return names


NON_SENSOR_COLUMNS = {"t_sec", "timestamp", "sys_time", "SystemTime", "t", "Timestamp"}


def _numeric_sensor_columns(segment_df: pd.DataFrame) -> list[str]:
    return [
        col
        for col in segment_df.columns
        if str(col) not in NON_SENSOR_COLUMNS
        and pd.api.types.is_numeric_dtype(segment_df[col])
    ]


def _match_columns(segment_df: pd.DataFrame, column_patterns: tuple[str, ...]) -> list[str]:
    if not column_patterns:
        return _numeric_sensor_columns(segment_df)

    column_map = {str(col).lower(): str(col) for col in segment_df.columns}
    matched: list[str] = []
    for pattern in column_patterns:
        pattern_l = str(pattern).lower()
        for col_l, original in column_map.items():
            if col_l == pattern_l or col_l.startswith(pattern_l):
                if original not in matched and original not in NON_SENSOR_COLUMNS:
                    matched.append(original)
    return matched


def _default_stream_units(modality: str, device_id: str) -> tuple[str, ...]:
    canonical_device = _normalize_device_id(device_id) or "unknown_device"
    mapped = STREAM_UNIT_BY_MODALITY_DEVICE.get((modality, canonical_device))
    if mapped:
        return mapped

    default_device = MODALITY_DEFAULT_DEVICES.get(modality, ["unknown_device"])[0]
    mapped = STREAM_UNIT_BY_MODALITY_DEVICE.get((modality, default_device))
    if mapped:
        return mapped

    fallback = f"{_sanitize_feature_token(canonical_device)}_{_sanitize_feature_token(modality)}"
    return (fallback,)


def _split_stream_units(
    modality: str,
    device_id: str,
    segment_df: pd.DataFrame,
) -> list[dict[str, Any]]:
    if modality in DISABLED_BASELINE_MODALITIES:
        return []

    stream_names = _default_stream_units(modality, device_id)
    units: list[dict[str, Any]] = []
    for stream_name in stream_names:
        spec = STREAM_UNIT_SPECS.get(stream_name, {})
        matched_cols = _match_columns(segment_df, tuple(spec.get("column_patterns", ())))
        if not matched_cols:
            continue
        units.append(
            {
                "stream_name": stream_name,
                "stream_df": segment_df.loc[:, ["t_sec", *matched_cols]].copy(),
                "columns": matched_cols,
                "spec": spec,
            }
        )

    if units:
        return units

    fallback_stream = stream_names[0]
    all_numeric = _numeric_sensor_columns(segment_df)
    if not all_numeric:
        return []
    return [
        {
            "stream_name": fallback_stream,
            "stream_df": segment_df.loc[:, ["t_sec", *all_numeric]].copy(),
            "columns": all_numeric,
            "spec": STREAM_UNIT_SPECS.get(fallback_stream, {}),
        }
    ]


def _feature_definition_for_stream(stream_name: str) -> str:
    if stream_name in {"muse_acc", "ring_acc", "esense_acc", "muse_gyro", "esense_gyro"}:
        return (
            "Paper-style IMU features: per-axis mean, std, peak_frequency, and absolute_integral; "
            "plus magnitude mean, std, and absolute_integral."
        )
    if stream_name in {"polar_ecg", "muse_ppg", "ring_ppg"}:
        return (
            "Paper-style cardiac features: HR mean/std; HRV rmssd, pnn50, tinn, sdnn; "
            "frequency powers for ulf(0.01-0.04), lf(0.04-0.15), hf(0.15-0.4), uhf(0.4-1.0), "
            "plus total_power, lf_hf_ratio, relative powers, and normalized lf/hf."
        )
    if stream_name == "msband_eda":
        return (
            "Paper-style EDA features: mean, std, min, max, slope, dynamic_range; "
            "tonic SCL mean/std/time correlation; phasic SCR mean/std, event count, sum magnitudes, total duration, and auc."
        )
    if stream_name == "muse_eeg":
        return (
            "Paper-style EEG-inspired features: per-band mean, std, variance, dynamic_range, num_peaks, "
            "num_zero_crossings, difference_variance, absolute_power for delta/theta/alpha/beta/spindle/kcomplex/sawtooth; "
            "plus power ratios delta/theta, theta/alpha, alpha/beta, and (delta+theta)/(alpha+beta)."
        )
    if stream_name == "beam_eye":
        return (
            "Eye-tracking approximation of the paper's EOG family: time-domain stats and slow/rapid movement power ratios "
            "computed on gaze/head motion signals, plus confidence summaries."
        )
    return GENERIC_FEATURE_GUIDE


def _estimate_sampling_rate(t_values: np.ndarray) -> float:
    if t_values.size < 2:
        return 1.0
    diffs = np.diff(t_values)
    diffs = diffs[np.isfinite(diffs) & (diffs > 1e-6)]
    if diffs.size == 0:
        return 1.0
    diffs_ms = diffs[diffs >= 1e-3]
    if diffs_ms.size:
        diffs = diffs_ms
    median = float(np.median(diffs))
    if median <= 1e-6:
        return 1.0
    return float(1.0 / median)


def _safe_value(value: Any) -> float:
    scalar = float(value)
    return scalar if np.isfinite(scalar) else 0.0


def _prefixed_feature_map(prefix: str, feature_values: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, value in feature_values.items():
        out[f"{prefix}_{name}"] = _safe_value(value)
    return out


@contextmanager
def _warning_suppression() -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def _mean_normalized_signal(matrix: np.ndarray) -> np.ndarray:
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    normalized = matrix.astype(float)
    for idx in range(normalized.shape[1]):
        column = normalized[:, idx]
        column = column - np.mean(column)
        std = np.std(column)
        if std > 1e-8:
            column = column / std
        normalized[:, idx] = column
    return np.mean(normalized, axis=1)


def _mean_signal(matrix: np.ndarray) -> np.ndarray:
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    return np.mean(matrix.astype(float), axis=1)


def _imu_feature_values(matrix: np.ndarray, sr: float) -> dict[str, float]:
    axis_labels = ["x", "y", "z"][: matrix.shape[1]]
    features: dict[str, float] = {}
    mean = np.mean(matrix, axis=0)
    std = np.std(matrix, axis=0)
    freqs = rfftfreq(matrix.shape[0], d=1 / sr) if matrix.shape[0] > 1 else np.asarray([0.0])
    fft_vals = np.abs(rfft(matrix - mean, axis=0)) if matrix.shape[0] > 1 else np.zeros((1, matrix.shape[1]))
    peak_freq = freqs[np.argmax(fft_vals, axis=0)] if matrix.shape[0] > 1 else np.zeros(matrix.shape[1])

    for idx, axis in enumerate(axis_labels):
        features[f"{axis}_mean"] = mean[idx]
        features[f"{axis}_std"] = std[idx]
        features[f"{axis}_peak_frequency"] = peak_freq[idx]
        features[f"{axis}_absolute_integral"] = np.sum(np.abs(matrix[:, idx])) / sr

    mag = np.linalg.norm(matrix[:, : len(axis_labels)], axis=1)
    features["mag_mean"] = np.mean(mag)
    features["mag_std"] = np.std(mag)
    features["mag_absolute_integral"] = np.sum(np.abs(mag)) / sr
    return features


def _cardiac_feature_values(signal_1d: np.ndarray, sr: float, *, mode: str) -> dict[str, float]:
    features = {
        "hr_mean": 0.0,
        "hr_std": 0.0,
        "hrv_rmssd": 0.0,
        "hrv_pnn50": 0.0,
        "hrv_tinn": 0.0,
        "hrv_std": 0.0,
        "hrv_ulf": 0.0,
        "hrv_lf": 0.0,
        "hrv_hf": 0.0,
        "hrv_uhf": 0.0,
        "hrv_total_power": 0.0,
        "hrv_lf_hf_ratio": 0.0,
        "hrv_rel_ulf": 0.0,
        "hrv_rel_lf": 0.0,
        "hrv_rel_hf": 0.0,
        "hrv_rel_uhf": 0.0,
        "hrv_normalized_lf": 0.0,
        "hrv_normalized_hf": 0.0,
    }
    if signal_1d.size < max(8, int(sr * 3)):
        return features

    try:
        import neurokit2 as nk

        with _warning_suppression():
            if mode == "ecg":
                cleaned = nk.ecg_clean(signal_1d, sampling_rate=sr)
                peaks, info = nk.ecg_peaks(cleaned, sampling_rate=sr)
                peak_locs = np.asarray(info.get("ECG_R_Peaks", []), dtype=int)
                if peak_locs.size < 3:
                    return features
                hr = nk.ecg_rate(peaks=peak_locs, sampling_rate=sr)
                time_source: Any = peaks
                peak_key = "HRV_SDNN"
            else:
                cleaned = nk.ppg_clean(signal_1d, sampling_rate=sr)
                _, info = nk.ppg_peaks(cleaned, sampling_rate=sr)
                peak_locs = np.asarray(info.get("PPG_Peaks", []), dtype=int)
                if peak_locs.size < 3:
                    return features
                hr = nk.ppg_rate(peaks=peak_locs, sampling_rate=sr)
                time_source = info
                peak_key = "HRV_SDNN"

            hr = np.asarray(hr, dtype=float)
            hr = hr[np.isfinite(hr)]
            if hr.size == 0:
                return features

            features["hr_mean"] = np.mean(hr)
            features["hr_std"] = np.std(hr)

            hrv_time = nk.hrv_time(time_source, sampling_rate=sr, show=False)
            hrv_freq = nk.hrv_frequency(
                time_source,
                sampling_rate=sr,
                ulf=(0, 0.01),
                vlf=(0.01, 0.04),
                lf=(0.04, 0.15),
                hf=(0.15, 0.4),
                vhf=(0.4, 1.0),
                show=False,
            )

        features["hrv_rmssd"] = hrv_time["HRV_RMSSD"].values[0]
        features["hrv_pnn50"] = hrv_time["HRV_pNN50"].values[0]
        features["hrv_tinn"] = hrv_time["HRV_TINN"].values[0]
        features["hrv_std"] = hrv_time[peak_key].values[0]

        # Match the paper wording while using NeuroKit's available bands.
        features["hrv_ulf"] = hrv_freq["HRV_VLF"].values[0]
        features["hrv_lf"] = hrv_freq["HRV_LF"].values[0]
        features["hrv_hf"] = hrv_freq["HRV_HF"].values[0]
        features["hrv_uhf"] = hrv_freq["HRV_VHF"].values[0]
        features["hrv_total_power"] = hrv_freq["HRV_TP"].values[0]
        features["hrv_lf_hf_ratio"] = hrv_freq["HRV_LFHF"].values[0]

        total_power = max(features["hrv_total_power"], 1e-8)
        features["hrv_rel_ulf"] = features["hrv_ulf"] / total_power
        features["hrv_rel_lf"] = features["hrv_lf"] / total_power
        features["hrv_rel_hf"] = features["hrv_hf"] / total_power
        features["hrv_rel_uhf"] = features["hrv_uhf"] / total_power
        features["hrv_normalized_lf"] = hrv_freq["HRV_LFn"].values[0]
        features["hrv_normalized_hf"] = hrv_freq["HRV_HFn"].values[0]
    except Exception:  # noqa: BLE001
        return features

    return features


def _eda_feature_values(signal_1d: np.ndarray, sr: float) -> dict[str, float]:
    features = {
        "mean": 0.0,
        "std": 0.0,
        "min": 0.0,
        "max": 0.0,
        "slope": 0.0,
        "dynamic_range": 0.0,
        "scl_mean": 0.0,
        "scl_std": 0.0,
        "scl_time_corr": 0.0,
        "scr_mean": 0.0,
        "scr_std": 0.0,
        "scr_num_segments": 0.0,
        "scr_sum_magnitudes": 0.0,
        "scr_total_duration": 0.0,
        "scr_auc": 0.0,
    }
    if signal_1d.size < 4:
        return features

    try:
        import neurokit2 as nk

        with _warning_suppression():
            cleaned = nk.eda_clean(signal_1d, sampling_rate=sr)
        features["mean"] = np.mean(cleaned)
        features["std"] = np.std(cleaned)
        features["min"] = np.min(cleaned)
        features["max"] = np.max(cleaned)
        features["slope"] = (cleaned[-1] - cleaned[0]) / max(len(cleaned), 1)
        features["dynamic_range"] = np.max(cleaned) - np.min(cleaned)

        with _warning_suppression():
            decomposed = nk.eda_phasic(cleaned, sampling_rate=sr)
        scl = np.asarray(decomposed["EDA_Tonic"], dtype=float)
        scr = np.asarray(decomposed["EDA_Phasic"], dtype=float)
        features["scl_mean"] = np.mean(scl)
        features["scl_std"] = np.std(scl)
        features["scr_mean"] = np.mean(scr)
        features["scr_std"] = np.std(scr)
        time_vector = np.linspace(0, len(scl) / sr, len(scl))
        corr = np.corrcoef(scl, time_vector)[0, 1] if len(scl) > 1 else 0.0
        features["scl_time_corr"] = 0.0 if not np.isfinite(corr) else corr

        try:
            with _warning_suppression():
                _, scr_info = nk.eda_peaks(scr, sampling_rate=sr)
            peaks, props = find_peaks(scr, height=0.01, distance=max(1, int(1.0 * sr)))
            features["scr_num_segments"] = len(peaks)
            features["scr_sum_magnitudes"] = np.sum(props.get("peak_heights", np.array([])))
            features["scr_total_duration"] = np.nansum(scr_info.get("SCR_RiseTime", np.array([])))
            features["scr_auc"] = np.trapezoid(np.maximum(scr, 0), dx=1 / sr)
        except Exception:  # noqa: BLE001
            pass
    except Exception:  # noqa: BLE001
        return features

    return features


def _temp_feature_values(signal_1d: np.ndarray) -> dict[str, float]:
    if signal_1d.size == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "slope": 0.0,
            "dynamic_range": 0.0,
        }
    return {
        "mean": np.mean(signal_1d),
        "std": np.std(signal_1d),
        "min": np.min(signal_1d),
        "max": np.max(signal_1d),
        "slope": (signal_1d[-1] - signal_1d[0]) / max(len(signal_1d), 1),
        "dynamic_range": np.max(signal_1d) - np.min(signal_1d),
    }


def _hr_feature_values(signal_1d: np.ndarray) -> dict[str, float]:
    if signal_1d.size == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "slope": 0.0,
            "dynamic_range": 0.0,
            "rmssd": 0.0,
        }
    diffs = np.diff(signal_1d)
    rmssd = np.sqrt(np.mean(np.square(diffs))) if diffs.size else 0.0
    return {
        "mean": np.mean(signal_1d),
        "std": np.std(signal_1d),
        "min": np.min(signal_1d),
        "max": np.max(signal_1d),
        "slope": (signal_1d[-1] - signal_1d[0]) / max(len(signal_1d), 1),
        "dynamic_range": np.max(signal_1d) - np.min(signal_1d),
        "rmssd": rmssd,
    }


def _bandpass(data: np.ndarray, sr: float, low: float, high: float) -> np.ndarray:
    nyquist = 0.5 * sr
    if low <= 0 or high >= nyquist or low >= high:
        return data
    b, a = butter(4, [low / nyquist, high / nyquist], btype="band")
    padlen = min(len(data) - 1, max(0, 3 * max(len(a), len(b))))
    if len(data) <= padlen + 1:
        return data
    return filtfilt(b, a, data, padlen=padlen)


def _eeg_feature_values(signal_1d: np.ndarray, sr: float) -> dict[str, float]:
    bands = {
        "delta": (0.5, 4.0),
        "theta": (4.0, 8.0),
        "alpha": (8.0, 12.0),
        "beta": (12.0, 30.0),
        "spindle": (12.0, 14.0),
        "kcomplex": (0.5, 1.5),
        "sawtooth": (2.0, 6.0),
    }
    features: dict[str, float] = {}
    if signal_1d.size < max(16, int(sr * 1.0)):
        for band_name in bands:
            for suffix in (
                "mean",
                "std",
                "variance",
                "dynamic_range",
                "num_peaks",
                "num_zero_crossings",
                "difference_variance",
                "absolute_power",
            ):
                features[f"{band_name}_{suffix}"] = 0.0
        features["delta/theta_ratio"] = 0.0
        features["theta/alpha_ratio"] = 0.0
        features["alpha/beta_ratio"] = 0.0
        features["(delta+theta)/(alpha+beta)_ratio"] = 0.0
        return features

    freqs, psd = welch(signal_1d, fs=sr, nperseg=min(len(signal_1d), max(8, int(sr * 2))))
    band_powers: dict[str, float] = {}
    for band_name, (low, high) in bands.items():
        filtered = _bandpass(signal_1d, sr, low, high)
        centered = filtered - np.mean(filtered)
        features[f"{band_name}_mean"] = np.mean(filtered)
        features[f"{band_name}_std"] = np.std(filtered)
        features[f"{band_name}_variance"] = np.var(filtered)
        features[f"{band_name}_dynamic_range"] = np.max(filtered) - np.min(filtered)
        peaks = find_peaks(centered, height=3 * np.std(filtered))[0] if filtered.size else np.array([])
        features[f"{band_name}_num_peaks"] = len(peaks)
        zero_crossings = np.where(np.diff(np.sign(centered)))[0] if filtered.size > 1 else np.array([])
        features[f"{band_name}_num_zero_crossings"] = len(zero_crossings)
        diffs = np.diff(filtered)
        features[f"{band_name}_difference_variance"] = np.var(diffs) if diffs.size else 0.0

        idx = np.logical_and(freqs >= low, freqs <= high)
        power = np.trapezoid(psd[idx], freqs[idx]) if np.any(idx) else 0.0
        features[f"{band_name}_absolute_power"] = power
        band_powers[band_name] = power

    delta = band_powers["delta"]
    theta = band_powers["theta"]
    alpha = band_powers["alpha"]
    beta = band_powers["beta"]
    features["delta/theta_ratio"] = delta / theta if theta > 0 else 0.0
    features["theta/alpha_ratio"] = theta / alpha if alpha > 0 else 0.0
    features["alpha/beta_ratio"] = alpha / beta if beta > 0 else 0.0
    features["(delta+theta)/(alpha+beta)_ratio"] = (
        (delta + theta) / (alpha + beta) if (alpha + beta) > 0 else 0.0
    )
    return features


def _eog_like_feature_values(signal_1d: np.ndarray, sr: float, name: str) -> dict[str, float]:
    if signal_1d.size < 4:
        return {
            f"{name}_mean": 0.0,
            f"{name}_std": 0.0,
            f"{name}_variance": 0.0,
            f"{name}_dynamic_range": 0.0,
            f"{name}_num_zero_crossings": 0.0,
            f"{name}_difference_variance": 0.0,
            f"{name}_num_large_movements": 0.0,
            f"{name}_slow_movement_power_ratio": 0.0,
            f"{name}_rapid_movement_power_ratio": 0.0,
        }

    centered = signal_1d - np.mean(signal_1d)
    zero_crossings = np.where(np.diff(np.sign(centered)))[0]
    diffs = np.diff(signal_1d)
    threshold = np.percentile(np.abs(diffs), 95) if diffs.size else 0.0
    large_count = np.sum(np.abs(diffs) >= threshold) if threshold > 0 else 0
    freqs, psd = welch(signal_1d, fs=sr, nperseg=min(len(signal_1d), max(8, int(sr * 2))))
    total_idx = np.logical_and(freqs >= 0.5, freqs <= 30.0)
    slow_idx = np.logical_and(freqs >= 0.5, freqs <= 2.0)
    rapid_idx = np.logical_and(freqs >= 2.0, freqs <= 5.0)
    total_power = np.trapezoid(psd[total_idx], freqs[total_idx]) if np.any(total_idx) else 0.0
    slow_power = np.trapezoid(psd[slow_idx], freqs[slow_idx]) if np.any(slow_idx) else 0.0
    rapid_power = np.trapezoid(psd[rapid_idx], freqs[rapid_idx]) if np.any(rapid_idx) else 0.0
    return {
        f"{name}_mean": np.mean(signal_1d),
        f"{name}_std": np.std(signal_1d),
        f"{name}_variance": np.var(signal_1d),
        f"{name}_dynamic_range": np.max(signal_1d) - np.min(signal_1d),
        f"{name}_num_zero_crossings": len(zero_crossings),
        f"{name}_difference_variance": np.var(diffs) if diffs.size else 0.0,
        f"{name}_num_large_movements": float(large_count),
        f"{name}_slow_movement_power_ratio": (slow_power / total_power) if total_power > 0 else 0.0,
        f"{name}_rapid_movement_power_ratio": (rapid_power / total_power) if total_power > 0 else 0.0,
    }


def _eye_feature_values(matrix: np.ndarray, columns: list[str], sr: float) -> dict[str, float]:
    col_map = {str(col).lower(): matrix[:, idx] for idx, col in enumerate(columns)}
    features: dict[str, float] = {}

    if "gaze_conf_int" in col_map:
        features["gaze_conf_mean"] = np.mean(col_map["gaze_conf_int"])
        features["gaze_conf_std"] = np.std(col_map["gaze_conf_int"])
    if "head_conf_int" in col_map:
        features["head_conf_mean"] = np.mean(col_map["head_conf_int"])
        features["head_conf_std"] = np.std(col_map["head_conf_int"])

    gaze_components = [col_map[key] for key in ("gaze_por_x", "gaze_por_y") if key in col_map]
    if gaze_components:
        gaze_matrix = np.column_stack(gaze_components)
        gaze_speed = np.linalg.norm(np.diff(gaze_matrix, axis=0), axis=1) * sr if len(gaze_matrix) > 1 else np.zeros((0,))
        if gaze_matrix.shape[1] >= 1:
            features.update(_eog_like_feature_values(gaze_matrix[:, 0], sr, "gaze_x"))
        if gaze_matrix.shape[1] >= 2:
            features.update(_eog_like_feature_values(gaze_matrix[:, 1], sr, "gaze_y"))
        if gaze_speed.size:
            features.update(_eog_like_feature_values(gaze_speed, sr, "gaze_speed"))

    head_components = [col_map[key] for key in ("head_pos_x_m", "head_pos_y_m", "head_pos_z_m") if key in col_map]
    if head_components:
        head_matrix = np.column_stack(head_components)
        head_speed = np.linalg.norm(np.diff(head_matrix, axis=0), axis=1) * sr if len(head_matrix) > 1 else np.zeros((0,))
        for idx, axis_name in enumerate(("head_x", "head_y", "head_z")[: head_matrix.shape[1]]):
            features.update(_eog_like_feature_values(head_matrix[:, idx], sr, axis_name))
        if head_speed.size:
            features.update(_eog_like_feature_values(head_speed, sr, "head_speed"))

    return features


def _clip_1_to_5(value: float) -> float:
    if not np.isfinite(value):
        return 3.0
    return float(min(5.0, max(1.0, value)))


def _round_1_to_5(value: float) -> int:
    return int(min(5, max(1, int(round(value)))))


def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return 0.0
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    mean_true = float(np.mean(y_true))
    ss_tot = float(np.sum((y_true - mean_true) ** 2))
    if ss_tot <= 1e-12:
        return 0.0
    return 1.0 - (ss_res / ss_tot)


def _parse_availability(row: dict[str, Any]) -> dict[str, bool]:
    availability: dict[str, bool] = {}
    for key, value in row.items():
        key_s = str(key)
        if key_s.startswith("has_"):
            availability[key_s[4:]] = _as_bool(value)
    if availability:
        return availability

    raw = row.get("availability_mask")
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return {str(k): _as_bool(v) for k, v in parsed.items()}
        except json.JSONDecodeError:
            return {}
    return {}


def _resolve_windows(config: RunConfig, window_index: pd.DataFrame | None) -> pd.DataFrame:
    windows = window_index
    if windows is None:
        run_path = config.run_dir / "windows" / "window_index.parquet"
        windows = read_table_with_fallback(run_path)
        if windows.empty:
            windows = read_table_with_fallback(config.windows_dir / "window_index.parquet")
    if windows is None or windows.empty:
        return pd.DataFrame()

    out = windows.copy()
    if "label_5class" in out.columns:
        out["label_5class"] = out["label_5class"].apply(_coerce_label_1_to_5)
    else:
        out["label_5class"] = out.get("label_3class", pd.Series([None] * len(out))).apply(_coerce_label_1_to_5)

    out["participant_id"] = pd.to_numeric(out.get("participant_id"), errors="coerce")
    out["t_start_sec"] = pd.to_numeric(out.get("t_start_sec"), errors="coerce")
    out["t_end_sec"] = pd.to_numeric(out.get("t_end_sec"), errors="coerce")

    out = out[
        out["label_5class"].notna()
        & out["participant_id"].notna()
        & out["t_start_sec"].notna()
        & out["t_end_sec"].notna()
        & (out["t_end_sec"] > out["t_start_sec"])
    ].copy()
    if out.empty:
        return out

    out["label_5class"] = out["label_5class"].astype(int)
    out["participant_id"] = out["participant_id"].astype(int)
    return out


def _ensure_summary(config: RunConfig, preprocessed_root: Path) -> Path | None:
    summary_path = preprocessed_root / "export_summary.csv"
    if summary_path.exists():
        return summary_path
    try:
        _df, outputs = run_phase_preprocessed_data(config, output_root=preprocessed_root)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Could not create preprocessed export summary: %s", exc)
        return None
    created = Path(outputs["summary"])
    return created if created.exists() else None


def _build_stream_map(summary_df: pd.DataFrame) -> dict[tuple[str, str], list[dict[str, Any]]]:
    stream_map: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in summary_df.to_dict(orient="records"):
        session_key = str(row.get("session_key", "")).strip()
        modality = str(row.get("modality", "")).strip()
        output_path = str(row.get("output_path", "")).strip()
        source_path = str(row.get("source_path", output_path)).strip()
        if not session_key or not modality or not output_path:
            continue

        path = Path(output_path)
        if not path.exists():
            continue

        device_id = _infer_device_id(modality=modality, source_path=source_path, row=row)
        source_id = _safe_path_component(Path(source_path).stem if source_path else path.stem, fallback=path.stem)
        entry = {
            "path": path,
            "device_id": device_id,
            "source_id": source_id,
            "source_path": source_path,
        }
        bucket = stream_map.setdefault((session_key, modality), [])
        if not any(Path(existing["path"]).resolve() == path.resolve() for existing in bucket):
            bucket.append(entry)
    return stream_map


def _load_stream(path: Path, cache: dict[Path, pd.DataFrame]) -> pd.DataFrame:
    if path in cache:
        return cache[path]

    df = pd.read_csv(path)
    if "t_sec" not in df.columns:
        for candidate in ("timestamp", "sys_time", "SystemTime", "t", "Timestamp"):
            if candidate in df.columns:
                df["t_sec"] = pd.to_numeric(df[candidate], errors="coerce")
                break
    if "t_sec" not in df.columns:
        df["t_sec"] = np.nan

    df["t_sec"] = pd.to_numeric(df["t_sec"], errors="coerce")
    df = df[df["t_sec"].notna()].copy()
    df = df.sort_values("t_sec", kind="mergesort").reset_index(drop=True)
    cache[path] = df
    return df


def _summarize_segment(
    modality: str,
    segment_df: pd.DataFrame,
    *,
    feature_prefix: str | None = None,
    numeric_cols: list[str] | None = None,
) -> tuple[dict[str, float], float]:
    if numeric_cols is None:
        numeric_cols = _numeric_sensor_columns(segment_df)
    if not numeric_cols:
        return {}, 0.0

    numeric = segment_df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.interpolate(limit_direction="both").ffill().bfill().fillna(0.0)
    matrix = numeric.to_numpy(dtype=float)
    if matrix.size == 0:
        return {}, 0.0
    prefix = feature_prefix or modality
    sr = _estimate_sampling_rate(pd.to_numeric(segment_df["t_sec"], errors="coerce").to_numpy(dtype=float))
    signal_raw = _mean_signal(matrix)
    signal_normalized = _mean_normalized_signal(matrix)

    if modality in {"muse_acc", "ring_acc", "esense_acc", "muse_gyro", "esense_gyro"}:
        base_features = _imu_feature_values(matrix, sr)
    elif modality == "polar_ecg":
        base_features = _cardiac_feature_values(signal_raw, sr, mode="ecg")
    elif modality in {"muse_ppg", "ring_ppg"}:
        base_features = _cardiac_feature_values(signal_raw, sr, mode="ppg")
    elif modality == "msband_eda":
        base_features = _eda_feature_values(signal_raw, sr)
    elif modality == "muse_eeg":
        base_features = _eeg_feature_values(signal_normalized, sr)
    elif modality == "beam_eye":
        base_features = _eye_feature_values(matrix, list(numeric.columns), sr)
    else:
        flat = matrix.reshape(-1)
        diffs = np.diff(matrix, axis=0)
        base_features = {
            "num_samples": float(matrix.shape[0]),
            "num_channels": float(matrix.shape[1]),
            "value_mean": float(np.mean(flat)),
            "value_std": float(np.std(flat)),
            "value_p10": float(np.percentile(flat, 10)),
            "value_p90": float(np.percentile(flat, 90)),
            "value_iqr": float(np.percentile(flat, 75) - np.percentile(flat, 25)),
            "abs_slope": float(np.mean(np.abs(diffs))) if diffs.size else 0.0,
        }

    features = _prefixed_feature_map(prefix, base_features)
    values = np.asarray(list(base_features.values()), dtype=float)
    dynamic = np.abs(values[np.isfinite(values)])
    activity = float(np.log1p(np.mean(dynamic))) if dynamic.size else 0.0
    return features, activity


def _build_records(
    windows: pd.DataFrame,
    stream_map: dict[tuple[str, str], list[dict[str, Any]]],
    enabled_modalities: tuple[str, ...],
    *,
    allowed_devices: set[str] | None = None,
    progress_label: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, list[float]]]:
    cache: dict[Path, pd.DataFrame] = {}
    records: list[dict[str, Any]] = []
    activities: dict[str, list[float]] = {m: [] for m in enabled_modalities}
    rows = windows.to_dict(orient="records")
    progress = _StageProgressBar(
        progress_label or "Build records",
        total=len(rows),
        logger=LOGGER,
    )

    for row in rows:
        session_key = str(row.get("session_key", ""))
        availability = _parse_availability(row)
        t_start = float(row["t_start_sec"])
        t_end = float(row["t_end_sec"])

        feature_map: dict[str, float] = {}
        modality_activity: dict[str, float] = {}
        modality_devices: dict[str, list[str]] = {}
        stream_metadata: dict[str, dict[str, Any]] = {}
        stream_activity_values: dict[str, list[float]] = {}

        for modality in enabled_modalities:
            if availability and not availability.get(modality, False):
                continue

            stream_entries = stream_map.get((session_key, modality), [])
            if not stream_entries:
                continue

            modality_activities: list[float] = []
            modality_device_ids: list[str] = []

            for entry in stream_entries:
                device_id = _normalize_device_id(entry.get("device_id")) or "unknown_device"
                if allowed_devices is not None and device_id not in allowed_devices:
                    continue

                path = Path(entry["path"])
                try:
                    stream_df = _load_stream(path, cache)
                except Exception as exc:  # noqa: BLE001
                    LOGGER.warning("Could not load stream for %s/%s (%s): %s", session_key, modality, device_id, exc)
                    continue

                segment = stream_df[(stream_df["t_sec"] >= t_start) & (stream_df["t_sec"] < t_end)].copy()
                if segment.empty:
                    continue

                source_token = _sanitize_feature_token(entry.get("source_id", path.stem))
                for unit in _split_stream_units(modality, device_id, segment):
                    stream_name = str(unit["stream_name"])
                    feature_prefix = f"{stream_name}__{source_token}"
                    features, activity = _summarize_segment(
                        stream_name,
                        unit["stream_df"],
                        feature_prefix=feature_prefix,
                        numeric_cols=list(unit["columns"]),
                    )
                    if not features:
                        continue

                    feature_map.update(features)
                    modality_activities.append(float(activity))
                    modality_device_ids.append(device_id)
                    stream_activity_values.setdefault(stream_name, []).append(float(activity))

                    meta = stream_metadata.setdefault(
                        stream_name,
                        {
                            "device_id": device_id,
                            "source_ids": [],
                            "source_paths": [],
                            "coarse_modality": str(unit["spec"].get("coarse_modality", modality)),
                            "sensor_modality": str(unit["spec"].get("sensor_modality", modality)),
                            "raw_columns": [],
                        },
                    )
                    if source_token not in meta["source_ids"]:
                        meta["source_ids"].append(source_token)
                    source_path_s = str(entry.get("source_path", path))
                    if source_path_s not in meta["source_paths"]:
                        meta["source_paths"].append(source_path_s)
                    for column in unit["columns"]:
                        if column not in meta["raw_columns"]:
                            meta["raw_columns"].append(str(column))

            if modality_activities:
                modality_devices[modality] = sorted(set(modality_device_ids))

        for stream_name, values in stream_activity_values.items():
            modality_activity[stream_name] = float(np.mean(values))
            activities.setdefault(stream_name, []).extend(values)

        for stream_name, meta in stream_metadata.items():
            modality_devices[stream_name] = [str(meta.get("device_id", "unknown_device"))]

        records.append(
            {
                "window_id": str(row.get("window_id", "")),
                "session_key": session_key,
                "participant_id": int(row["participant_id"]),
                "event_id": str(row.get("event_id", "")),
                "video_uid": str(row.get("video_uid", "")),
                "t_start_video_sec": float(row.get("t_start_video_sec", 0.0)),
                "t_end_video_sec": float(row.get("t_end_video_sec", 0.0)),
                "label_true": int(row["label_5class"]),
                "features": feature_map,
                "modality_activity": modality_activity,
                "modality_devices": modality_devices,
                "stream_metadata": stream_metadata,
            }
        )
        progress.update(
            context=(
                f"window_id={str(row.get('window_id', ''))}"
                if row.get("window_id", "")
                else None
            )
        )

    progress.close()

    return records, activities


def _build_normalizers(activities: dict[str, list[float]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for modality, values in activities.items():
        if not values:
            out[modality] = {"median": 0.0, "scale": 1.0}
            continue
        arr = np.asarray(values, dtype=float)
        median = float(np.median(arr))
        iqr = float(np.percentile(arr, 75) - np.percentile(arr, 25))
        scale = iqr if iqr > 1e-6 else float(np.std(arr))
        if scale <= 1e-6:
            scale = 1.0
        out[modality] = {"median": median, "scale": scale}
    return out


def _stream_feature_subset(features: dict[str, Any], stream_name: str) -> dict[str, Any]:
    prefix = f"{stream_name}__"
    subset = {
        str(key): value
        for key, value in features.items()
        if str(key).startswith(prefix)
    }
    if not subset:
        return {}

    renamed: dict[str, Any] = {}
    short_name_counts: dict[str, int] = {}
    ordered_keys = sorted(subset.keys())
    for key in ordered_keys:
        short_name = _feature_name_from_key(key, stream_name)
        short_name_counts[short_name] = short_name_counts.get(short_name, 0) + 1

    for key in ordered_keys:
        short_name = _feature_name_from_key(key, stream_name)
        if short_name_counts.get(short_name, 0) <= 1:
            renamed[short_name] = subset[key]
            continue

        source_and_feature = str(key)[len(prefix):]
        source_id = source_and_feature.split("__", 1)[0]
        renamed[f"{source_id}__{short_name}"] = subset[key]

    return renamed


def _build_fold_example_pool(
    train_records: list[dict[str, Any]],
    examples_per_class: int,
) -> dict[int, list[dict[str, Any]]]:
    by_label: dict[int, list[dict[str, Any]]] = {label: [] for label in range(1, 6)}
    if examples_per_class <= 0:
        return by_label

    for label in range(1, 6):
        label_records = [
            record
            for record in train_records
            if int(record.get("label_true", -1)) == label
        ]
        by_label[label] = sorted(
            label_records,
            key=lambda record: (
                -len(record.get("modality_activity", {})),
                int(record.get("participant_id", -1)),
                str(record.get("window_id", "")),
            ),
        )

    return by_label


def _example_candidate_sort_key(
    record: dict[str, Any],
    candidate: dict[str, Any],
    target_streams: set[str],
) -> tuple[float, ...]:
    candidate_streams = set(candidate.get("modality_activity", {}).keys())
    shared_streams = sorted(target_streams & candidate_streams)
    missing_count = len(target_streams - candidate_streams)
    extra_count = len(candidate_streams - target_streams)
    activity_distance = 0.0
    record_activity = record.get("modality_activity", {})
    candidate_activity = candidate.get("modality_activity", {})
    for stream_name in shared_streams:
        activity_distance += abs(
            float(record_activity.get(stream_name, 0.0))
            - float(candidate_activity.get(stream_name, 0.0))
        )

    return (
        -float(len(shared_streams)),
        float(missing_count),
        float(activity_distance),
        float(extra_count),
        -float(len(candidate_streams)),
        float(int(candidate.get("participant_id", -1))),
        str(candidate.get("window_id", "")),
    )


def _select_examples_for_record(
    record: dict[str, Any],
    fold_example_pool: dict[int, list[dict[str, Any]]],
    examples_per_class: int,
) -> list[dict[str, Any]]:
    if examples_per_class <= 0:
        return []

    target_streams = set(record.get("modality_activity", {}).keys())
    selected: list[dict[str, Any]] = []
    for label in range(1, 6):
        candidates = list(fold_example_pool.get(label, []))
        candidates = sorted(
            candidates,
            key=lambda candidate: _example_candidate_sort_key(
                record,
                candidate,
                target_streams,
            ),
        )
        selected.extend(candidates[:examples_per_class])
    return selected


def _predict_record_heuristic(
    record: dict[str, Any], normalizers: dict[str, dict[str, float]]
) -> tuple[float, dict[str, float], str]:
    votes: dict[str, float] = {}
    for modality, activity in record["modality_activity"].items():
        norm = normalizers.get(modality, {"median": 0.0, "scale": 1.0})
        z = (float(activity) - float(norm["median"])) / max(1e-6, float(norm["scale"]))
        votes[modality] = _clip_1_to_5(3.0 + z)

    pred = float(np.mean(list(votes.values()))) if votes else 3.0
    top = [m for m, _ in sorted(votes.items(), key=lambda kv: abs(kv[1] - 3.0), reverse=True)[:3]]
    reason = "Heuristic zero-shot consensus over modality activity indices"
    if top:
        reason = f"{reason}; dominant modalities: {', '.join(top)}"
    return pred, votes, reason


def _empty_token_usage() -> dict[str, Any]:
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "by_agent": {},
    }


class _LLMConsensusPredictor:
    """Adapter over the full ConSensus chain: modality -> semantic -> statistical -> hybrid."""

    def __init__(self, config: RunConfig, phase_dir: Path):
        baseline_cfg = config.baseline.consensus_zero_shot
        repo_root = str(config.repo_root)
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)

        consensus_repo = Path(baseline_cfg.consensus_repo_path).resolve()
        if not consensus_repo.exists():
            raise FileNotFoundError(f"ConSensus repo path not found: {consensus_repo}")

        from consensus.core.agent_pool import AgentPool  # type: ignore
        from consensus.core.hybrid_fusion_agent import HybridFusionAgent  # type: ignore
        from consensus.core.modality_agent import ModalityAgent  # type: ignore
        from consensus.core.model import load_models  # type: ignore
        from consensus.core.semantic_fusion_agent import SemanticFusionAgent  # type: ignore
        from consensus.core.statistical_fusion_agent import StatisticalFusionAgent  # type: ignore

        model_count = max(1, int(baseline_cfg.max_concurrent_model_calls))
        models = [baseline_cfg.model for _ in range(model_count)]
        self.model_pool = load_models(
            models=models,
            temperature=float(baseline_cfg.temperature),
            num_ctx=int(baseline_cfg.num_ctx),
        )
        self.AgentPool = AgentPool
        self.ModalityAgent = ModalityAgent
        self.SemanticFusionAgent = SemanticFusionAgent
        self.StatisticalFusionAgent = StatisticalFusionAgent
        self.HybridFusionAgent = HybridFusionAgent
        self.phase_dir = phase_dir
        self.window_size_sec = float(config.window_size_sec)
        self._last_token_usage = _empty_token_usage()
        self.classes_info = list(LABEL_ORDER)
        self.task_info = (
            "**Task**:\n"
            "Predict the participant's self-report for this multimodal window on the 1-5 attention-difficulty scale.\n"
            "Return exactly one label from ['1', '2', '3', '4', '5']. Do not output X.\n\n"
            "**Scale**:\n"
            "1 = effortless, automatic attention; highest engagement.\n"
            "2 = fairly easy to stay attentive; relatively high engagement.\n"
            "3 = moderate or mixed difficulty.\n"
            "4 = hard to stay attentive; relatively low engagement.\n"
            "5 = very hard to stay attentive; lowest engagement and highest effort.\n\n"
            "**Notes**:\n"
            f"- Features summarize a {self.window_size_sec:g}-second window, even though the study prompt said 'last minute'.\n"
            "- Each modality agent sees one device-specific sensor stream (for example muse_acc, polar_ecg, ring_ppg, ring_acc, esense_acc, esense_gyro).\n"
            "- If examples are enabled, they come only from non-test training participants in the current fold.\n"
            "- This is a subjective self-report label, not a direct measure of body motion, eye motion, or physiological arousal.\n"
            "- More movement, scanning, or arousal does not automatically imply labels 4-5; it can also happen during active, engaged work.\n"
            "- Prefer the dataset's provided examples over generic commonsense assumptions whenever they conflict.\n"
            "- Treat label 3 as the default when evidence is mixed, weak, noisy, or contradictory across signals.\n"
            "- Reserve labels 4-5 for strong evidence of sustained difficulty, disengagement, or very low vigilance after considering artifacts and alternative explanations."
        )

    def _modality_info(self, modality: str, record: dict[str, Any]) -> dict[str, str]:
        stream_meta = record.get("stream_metadata", {}).get(modality, {})
        spec = STREAM_UNIT_SPECS.get(modality, {})
        coarse_modality = str(
            stream_meta.get("coarse_modality", spec.get("coarse_modality", modality))
        )
        sensor_modality = str(
            stream_meta.get("sensor_modality", spec.get("sensor_modality", coarse_modality))
        )
        device_id = _normalize_device_id(
            stream_meta.get("device_id", spec.get("device_id", "unknown_device"))
        ) or "unknown_device"

        data_collection = str(
            spec.get("data_collection", f"{modality} sensor stream from device {device_id}.")
        )
        feature_extraction = str(
            spec.get("feature_extraction", "Window-level summary feature statistics.")
        )
        measurement_notes = str(
            spec.get(
                "measurement_notes",
                "Measurement unit is not explicitly declared in raw exports; interpret as device-native units unless a unit-bearing column name is present.",
            )
        )
        feature_keys = _stream_feature_keys(record, modality)
        feature_names = _friendly_feature_names(feature_keys, modality)
        raw_columns = stream_meta.get("raw_columns", [])
        return {
            "Input stream name": modality,
            "Device id": device_id,
            "Device details": DEVICE_INFO.get(device_id, DEVICE_INFO["unknown_device"]),
            "Coarse exported modality": coarse_modality,
            "Sensor modality for this LLM input": sensor_modality,
            "Data collection": data_collection,
            "Feature extraction": feature_extraction,
            "Configured window length (seconds)": f"{self.window_size_sec:g}",
            "Raw measurement notes": measurement_notes,
            "Raw columns included in this input": ", ".join(raw_columns) if raw_columns else "(unknown)",
            "Feature definitions": _feature_definition_for_stream(modality),
            "Feature names for this input": (", ".join(feature_names) if feature_names else "(no feature names found)"),
        }

    async def _predict_async(
        self,
        record: dict[str, Any],
        examples: list[dict[str, Any]] | None = None,
    ) -> tuple[float, dict[str, float], str]:
        self._last_token_usage = _empty_token_usage()
        modalities_present = sorted(record["modality_activity"].keys())
        if not modalities_present:
            return 3.0, {}, "No available modalities in this window."

        sample_id = _safe_path_component(record.get("window_id", "unknown_window"), fallback="unknown_window")
        log_path = self.phase_dir / "llm_logs" / sample_id
        log_path.mkdir(parents=True, exist_ok=True)

        sample = {
            "label": "unknown",
            "features": record["features"],
        }
        example_records = list(examples or [])

        pool = self.AgentPool(str(log_path))
        for modality in modalities_present:
            modality_examples = [
                {
                    "label": str(example.get("label_true", "")),
                    "features": _stream_feature_subset(
                        example.get("features", {}),
                        modality,
                    ),
                }
                for example in example_records
                if _stream_feature_subset(example.get("features", {}), modality)
            ]
            pool.add_modality_agent(
                self.ModalityAgent(
                    name=modality,
                    model_pool=self.model_pool,
                    task_info=self.task_info,
                    classes_info=self.classes_info,
                    modality_info=self._modality_info(modality, record),
                    sample=sample,
                    examples=modality_examples,
                    log_path=str(log_path),
                )
            )

        pool.add_semantic_fusion_agent(
            self.SemanticFusionAgent(
                name="SemanticFusionAgent",
                model_pool=self.model_pool,
                task_info=self.task_info,
                classes_info=self.classes_info,
                log_path=str(log_path),
            )
        )
        pool.add_statistical_fusion_agent(
            self.StatisticalFusionAgent(
                name="StatisticalFusionAgent",
                model_pool=self.model_pool,
                task_info=self.task_info,
                classes_info=self.classes_info,
                log_path=str(log_path),
            )
        )
        pool.add_hybrid_fusion_agent(
            self.HybridFusionAgent(
                name="HybridFusionAgent",
                model_pool=self.model_pool,
                task_info=self.task_info,
                classes_info=self.classes_info,
                log_path=str(log_path),
            )
        )

        majority_answer = await pool.run_parallel_interpretation()
        if majority_answer is None:
            raise RuntimeError("ConSensus modality-agent interpretation failed.")
        pool.log_summary(f"[Majority Vote] answer: {majority_answer}")

        semantic_answer = await pool.run_semantic_fusion()
        statistical_answer = await pool.run_statistical_fusion()
        hybrid_answer = await pool.run_hybrid_fusion()

        final_answer = hybrid_answer or semantic_answer or statistical_answer or majority_answer

        pred_label = _coerce_answer_1_to_5(final_answer)
        if pred_label is None:
            pred_label = _coerce_answer_1_to_5(hybrid_answer)
        if pred_label is None:
            pred_label = _coerce_answer_1_to_5(semantic_answer)
        if pred_label is None:
            pred_label = _coerce_answer_1_to_5(statistical_answer)
        if pred_label is None:
            pred_label = _coerce_answer_1_to_5(majority_answer)
        if pred_label is None:
            pred_label = 3

        y_true = _coerce_label_1_to_5(record.get("label_true"))
        y_true_text = str(y_true) if y_true is not None else "unknown"
        pool.log_summary(
            "[Consensus Final] "
            f"window_id={record.get('window_id', 'unknown_window')} "
            f"y_true={y_true_text} "
            f"pred={pred_label} "
            f"majority={majority_answer} "
            f"semantic={semantic_answer} "
            f"statistical={statistical_answer} "
            f"hybrid={hybrid_answer}"
        )
        self._last_token_usage = pool.get_token_usage_summary()
        pool.log_summary(
            "[Token Usage] "
            f"window_id={record.get('window_id', 'unknown_window')} "
            f"input_tokens={self._last_token_usage['input_tokens']} "
            f"output_tokens={self._last_token_usage['output_tokens']} "
            f"total_tokens={self._last_token_usage['total_tokens']}"
        )
        usage_by_agent = dict(self._last_token_usage.get("by_agent", {}))
        modality_agent_names = {str(name) for name in modalities_present}
        modality_totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        fusion_totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        for agent_name, usage in sorted(
            usage_by_agent.items(),
            key=lambda kv: int(kv[1].get("total_tokens", 0)),
            reverse=True,
        ):
            in_tok = int(usage.get("input_tokens", 0))
            out_tok = int(usage.get("output_tokens", 0))
            total_tok = int(usage.get("total_tokens", 0))
            pool.log_summary(
                "[Token Usage Agent] "
                f"agent={agent_name} input_tokens={in_tok} output_tokens={out_tok} total_tokens={total_tok}"
            )
            target = modality_totals if agent_name in modality_agent_names else fusion_totals
            target["input_tokens"] += in_tok
            target["output_tokens"] += out_tok
            target["total_tokens"] += total_tok
        pool.log_summary(
            "[Token Usage Stage] "
            f"modality_input={modality_totals['input_tokens']} "
            f"modality_output={modality_totals['output_tokens']} "
            f"modality_total={modality_totals['total_tokens']} "
            f"fusion_input={fusion_totals['input_tokens']} "
            f"fusion_output={fusion_totals['output_tokens']} "
            f"fusion_total={fusion_totals['total_tokens']}"
        )

        responses = pool.get_last_responses()
        votes: dict[str, float] = {}
        reason_chunks: list[str] = []
        for modality, response in responses.items():
            vote = _coerce_answer_1_to_5(response.get("ANSWER"))
            if vote is not None:
                votes[modality] = float(vote)
            text = str(response.get("REASON", "")).strip()
            if text:
                reason_chunks.append(f"{modality}: {text[:140]}")

        reason = " | ".join(reason_chunks[:3]).strip()
        reason_suffix = (
            f"semantic={semantic_answer}; statistical={statistical_answer}; hybrid={hybrid_answer}; majority={majority_answer}"
        )
        reason = f"{reason} [{reason_suffix}]" if reason else reason_suffix
        return float(pred_label), votes, reason

    def predict(
        self,
        record: dict[str, Any],
        examples: list[dict[str, Any]] | None = None,
    ) -> tuple[float, dict[str, float], str]:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self._predict_async(record, examples=examples))

        result_holder: dict[str, tuple[float, dict[str, float], str]] = {}
        error_holder: dict[str, Exception] = {}

        def _thread_runner() -> None:
            try:
                result_holder["value"] = asyncio.run(
                    self._predict_async(record, examples=examples)
                )
            except Exception as exc:  # noqa: BLE001
                error_holder["error"] = exc

        thread = threading.Thread(target=_thread_runner, daemon=True)
        thread.start()
        thread.join()

        if "error" in error_holder:
            raise error_holder["error"]
        return result_holder["value"]

    def get_last_token_usage(self) -> dict[str, Any]:
        usage = self._last_token_usage or _empty_token_usage()
        return {
            "input_tokens": int(usage.get("input_tokens", 0)),
            "output_tokens": int(usage.get("output_tokens", 0)),
            "total_tokens": int(usage.get("total_tokens", 0)),
            "by_agent": dict(usage.get("by_agent", {})),
        }


def _compute_metrics(df: pd.DataFrame) -> tuple[dict[str, float], np.ndarray]:
    y_true = df["y_true"].to_numpy(dtype=float)
    y_pred_cont = df["y_pred_continuous"].to_numpy(dtype=float)
    y_true_int = df["y_true"].to_numpy(dtype=int)
    y_pred_int = df["y_pred"].to_numpy(dtype=int)

    mae = float(np.mean(np.abs(y_true - y_pred_cont))) if len(df) else 0.0
    rmse = float(np.sqrt(np.mean((y_true - y_pred_cont) ** 2))) if len(df) else 0.0
    r2 = _safe_r2(y_true, y_pred_cont)
    medae = float(np.median(np.abs(y_true - y_pred_cont))) if len(df) else 0.0
    rounded_acc = float(accuracy_score(y_true_int, y_pred_int)) if len(df) else 0.0
    rounded_macro_f1 = float(
        f1_score(y_true_int, y_pred_int, average="macro", labels=np.arange(1, 6), zero_division=0)
    ) if len(df) else 0.0
    rounded_weighted_f1 = float(
        f1_score(y_true_int, y_pred_int, average="weighted", labels=np.arange(1, 6), zero_division=0)
    ) if len(df) else 0.0
    rounded_precision_macro = float(
        precision_score(y_true_int, y_pred_int, average="macro", labels=np.arange(1, 6), zero_division=0)
    ) if len(df) else 0.0
    rounded_recall_macro = float(
        recall_score(y_true_int, y_pred_int, average="macro", labels=np.arange(1, 6), zero_division=0)
    ) if len(df) else 0.0
    if len(df):
        per_class_recalls: list[float] = []
        for label in range(1, 6):
            mask = y_true_int == label
            denom = int(np.sum(mask))
            if denom <= 0:
                per_class_recalls.append(0.0)
            else:
                per_class_recalls.append(float(np.sum(y_pred_int[mask] == label) / denom))
        rounded_balanced_accuracy = float(np.mean(per_class_recalls))
    else:
        rounded_balanced_accuracy = 0.0
    rounded_qwk = float(cohen_kappa_score(y_true_int, y_pred_int, weights="quadratic")) if len(df) else 0.0
    if not np.isfinite(rounded_qwk):
        rounded_qwk = 0.0
    exact_match_count = int(np.sum(y_true_int == y_pred_int)) if len(df) else 0

    y_true_bin = (y_true_int >= 4).astype(int) if len(df) else np.asarray([], dtype=int)
    y_pred_bin = (y_pred_int >= 4).astype(int) if len(df) else np.asarray([], dtype=int)
    binary_acc = float(accuracy_score(y_true_bin, y_pred_bin)) if len(df) else 0.0
    binary_macro_f1 = float(f1_score(y_true_bin, y_pred_bin, average="macro", zero_division=0)) if len(df) else 0.0
    binary_f1_high = float(f1_score(y_true_bin, y_pred_bin, pos_label=1, zero_division=0)) if len(df) else 0.0
    binary_precision_macro = float(
        precision_score(y_true_bin, y_pred_bin, average="macro", zero_division=0)
    ) if len(df) else 0.0
    binary_recall_macro = float(
        recall_score(y_true_bin, y_pred_bin, average="macro", zero_division=0)
    ) if len(df) else 0.0

    cm = confusion_matrix(y_true_int, y_pred_int, labels=np.arange(1, 6)) if len(df) else np.zeros((5, 5), dtype=int)
    y_true_counts = np.bincount(y_true_int, minlength=6) if len(df) else np.zeros(6, dtype=int)
    y_pred_counts = np.bincount(y_pred_int, minlength=6) if len(df) else np.zeros(6, dtype=int)
    y_true_low_count = int(np.sum(y_true_bin == 0)) if len(df) else 0
    y_true_high_count = int(np.sum(y_true_bin == 1)) if len(df) else 0
    y_pred_low_count = int(np.sum(y_pred_bin == 0)) if len(df) else 0
    y_pred_high_count = int(np.sum(y_pred_bin == 1)) if len(df) else 0

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "medae": medae,
        "rounded_accuracy": rounded_acc,
        "rounded_macro_f1": rounded_macro_f1,
        "rounded_weighted_f1": rounded_weighted_f1,
        "rounded_precision_macro": rounded_precision_macro,
        "rounded_recall_macro": rounded_recall_macro,
        "rounded_balanced_accuracy": rounded_balanced_accuracy,
        "rounded_qwk": rounded_qwk,
        "exact_match_count": exact_match_count,
        "binary_accuracy": binary_acc,
        "binary_macro_f1": binary_macro_f1,
        "binary_f1_high": binary_f1_high,
        "binary_precision_macro": binary_precision_macro,
        "binary_recall_macro": binary_recall_macro,
        "y_true_low_count": y_true_low_count,
        "y_true_high_count": y_true_high_count,
        "y_pred_low_count": y_pred_low_count,
        "y_pred_high_count": y_pred_high_count,
        "y_true_count_1": int(y_true_counts[1]),
        "y_true_count_2": int(y_true_counts[2]),
        "y_true_count_3": int(y_true_counts[3]),
        "y_true_count_4": int(y_true_counts[4]),
        "y_true_count_5": int(y_true_counts[5]),
        "y_pred_count_1": int(y_pred_counts[1]),
        "y_pred_count_2": int(y_pred_counts[2]),
        "y_pred_count_3": int(y_pred_counts[3]),
        "y_pred_count_4": int(y_pred_counts[4]),
        "y_pred_count_5": int(y_pred_counts[5]),
        "num_samples": int(len(df)),
    }, cm


def _live_count_dict(metrics: dict[str, float], prefix: str) -> dict[str, int]:
    return {
        "1": int(metrics.get(f"{prefix}_1", 0)),
        "2": int(metrics.get(f"{prefix}_2", 0)),
        "3": int(metrics.get(f"{prefix}_3", 0)),
        "4": int(metrics.get(f"{prefix}_4", 0)),
        "5": int(metrics.get(f"{prefix}_5", 0)),
    }


def _log_live_metrics(scope: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    live_df = pd.DataFrame(rows)
    metrics, _ = _compute_metrics(live_df)
    LOGGER.info(
        "[Live Metrics][%s] n=%d acc=%.4f macro_f1=%.4f weighted_f1=%.4f precision_macro=%.4f recall_macro=%.4f bal_acc=%.4f qwk=%.4f mae=%.4f rmse=%.4f medae=%.4f r2=%.4f bin_acc=%.4f bin_macro_f1=%.4f",
        scope,
        int(metrics.get("num_samples", 0)),
        float(metrics.get("rounded_accuracy", 0.0)),
        float(metrics.get("rounded_macro_f1", 0.0)),
        float(metrics.get("rounded_weighted_f1", 0.0)),
        float(metrics.get("rounded_precision_macro", 0.0)),
        float(metrics.get("rounded_recall_macro", 0.0)),
        float(metrics.get("rounded_balanced_accuracy", 0.0)),
        float(metrics.get("rounded_qwk", 0.0)),
        float(metrics.get("mae", 0.0)),
        float(metrics.get("rmse", 0.0)),
        float(metrics.get("medae", 0.0)),
        float(metrics.get("r2", 0.0)),
        float(metrics.get("binary_accuracy", 0.0)),
        float(metrics.get("binary_macro_f1", 0.0)),
    )
    LOGGER.info(
        "[Live Counters][%s] y_true=%s y_pred=%s low_high_true={low:%d,high:%d} low_high_pred={low:%d,high:%d} exact=%d",
        scope,
        json.dumps(_live_count_dict(metrics, "y_true_count"), sort_keys=True),
        json.dumps(_live_count_dict(metrics, "y_pred_count"), sort_keys=True),
        int(metrics.get("y_true_low_count", 0)),
        int(metrics.get("y_true_high_count", 0)),
        int(metrics.get("y_pred_low_count", 0)),
        int(metrics.get("y_pred_high_count", 0)),
        int(metrics.get("exact_match_count", 0)),
    )


def _export(
    phase_dir: Path,
    summary: dict[str, Any],
    split_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    fold_metric_rows: list[dict[str, Any]],
) -> tuple[dict[str, Path], dict[str, Any]]:
    outputs: dict[str, Path] = {}
    z1_summary_path = phase_dir / "z1_summary.json"
    with z1_summary_path.open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2, sort_keys=True)
    outputs["z1_summary"] = z1_summary_path

    if not prediction_rows:
        return outputs, {
            "baseline_config_hash": summary["baseline_config_hash"],
            "modalities_enabled": int(sum(bool(v) for v in summary["modalities"].values())),
            "num_prediction_rows": 0,
            "participants": [],
        }

    predictions_df = pd.DataFrame(prediction_rows)
    splits_df = pd.DataFrame(split_rows)
    fold_df = pd.DataFrame(fold_metric_rows)
    overall_metrics, cm = _compute_metrics(predictions_df)
    overall_df = pd.DataFrame([{"experiment": "consensus_zero_shot", **overall_metrics}])

    outputs["predictions"] = write_table_with_fallback(predictions_df, phase_dir / "predictions.parquet", logger=LOGGER)
    outputs["loso_splits"] = write_table_with_fallback(splits_df, phase_dir / "loso_splits.parquet", logger=LOGGER)
    outputs["metrics_per_fold"] = write_table_with_fallback(fold_df, phase_dir / "metrics_per_fold.parquet", logger=LOGGER)
    outputs["metrics_overall"] = write_table_with_fallback(overall_df, phase_dir / "metrics_overall.parquet", logger=LOGGER)

    confusion_dir = phase_dir / "confusion"
    confusion_dir.mkdir(parents=True, exist_ok=True)
    overall_cm = confusion_dir / "overall_confusion_matrix.csv"
    pd.DataFrame(cm, index=LABEL_ORDER, columns=LABEL_ORDER).to_csv(overall_cm, index=True)
    outputs["confusion_overall"] = overall_cm

    stage_summary = {
        "stage": "baseline_consensus_zs",
        "status": summary["status"],
        "backend": summary.get("backend", "heuristic_consensus"),
        "num_predictions": int(len(predictions_df)),
        "participants": sorted({int(v) for v in predictions_df["participant_id"].tolist()}),
        "label_order": list(LABEL_ORDER),
    }
    stage_summary_path = phase_dir / "summary.json"
    with stage_summary_path.open("w", encoding="utf-8") as fp:
        json.dump(stage_summary, fp, indent=2, sort_keys=True)
    outputs["summary"] = stage_summary_path

    return outputs, {
        "baseline_config_hash": summary["baseline_config_hash"],
        "modalities_enabled": int(sum(bool(v) for v in summary["modalities"].values())),
        "num_prediction_rows": int(len(predictions_df)),
        "participants": stage_summary["participants"],
        "mae": float(overall_metrics["mae"]),
        "rmse": float(overall_metrics["rmse"]),
        "r2": float(overall_metrics["r2"]),
        "rounded_accuracy": float(overall_metrics["rounded_accuracy"]),
        "rounded_macro_f1": float(overall_metrics["rounded_macro_f1"]),
        "rounded_weighted_f1": float(overall_metrics["rounded_weighted_f1"]),
        "rounded_precision_macro": float(overall_metrics["rounded_precision_macro"]),
        "rounded_recall_macro": float(overall_metrics["rounded_recall_macro"]),
        "rounded_balanced_accuracy": float(overall_metrics["rounded_balanced_accuracy"]),
        "rounded_qwk": float(overall_metrics["rounded_qwk"]),
        "binary_accuracy": float(overall_metrics["binary_accuracy"]),
        "binary_macro_f1": float(overall_metrics["binary_macro_f1"]),
    }


def run_phase_z1_baseline_consensus(
    config: RunConfig,
    *,
    window_index: pd.DataFrame | None = None,
    example_window_index: pd.DataFrame | None = None,
    preprocessed_root: Path | None = None,
) -> tuple[dict[str, Path], dict[str, Any]]:
    """Run the ConSensus engagement baseline for 1..5 engagement regression."""

    validate_core_config(config)
    validate_baseline_config(config)
    config.ensure_directories()

    baseline_cfg = config.baseline.consensus_zero_shot
    phase_dir = config.run_dir / "baselines" / "consensus_zs"
    phase_dir.mkdir(parents=True, exist_ok=True)

    effective_zero_shot = bool(int(baseline_cfg.examples_per_class) <= 0)

    summary: dict[str, Any] = {
        "stage": "baseline_consensus_zs",
        "status": "phase_z1_ready",
        "enabled": bool(baseline_cfg.enabled),
        "zero_shot": effective_zero_shot,
        "examples_per_class": int(baseline_cfg.examples_per_class),
        "model": baseline_cfg.model,
        "provider": baseline_cfg.provider,
        "temperature": float(baseline_cfg.temperature),
        "num_ctx": int(baseline_cfg.num_ctx),
        "max_concurrent_samples": int(baseline_cfg.max_concurrent_samples),
        "max_concurrent_model_calls": int(baseline_cfg.max_concurrent_model_calls),
        "modalities": dict(sorted(baseline_cfg.modalities.items())),
        "devices": dict(sorted(baseline_cfg.devices.items())),
        "log_raw_prompts": bool(baseline_cfg.log_raw_prompts),
        "consensus_repo_path": baseline_cfg.consensus_repo_path,
        "consensus_commit": baseline_cfg.consensus_commit,
        "baseline_config_hash": config.baseline_config_hash(),
        "runtime_zero_shot_verified": effective_zero_shot,
        "example_source_policy": (
            "none"
            if int(baseline_cfg.examples_per_class) <= 0
            else "train_participants_only"
        ),
    }

    windows = _resolve_windows(config, window_index)
    if windows.empty:
        summary["status"] = "phase_z1_ready_missing_windows"
        return _export(phase_dir, summary, [], [], [])
    LOGGER.info(
        "[Stage] Loaded %d evaluation windows (mode=%s, window=%.1fs, stride=%.1fs)",
        len(windows),
        str(config.window_generation_mode),
        float(config.window_size_sec),
        float(config.stride_sec),
    )

    example_windows = windows
    if example_window_index is not None:
        example_windows = _resolve_windows(config, example_window_index)
        if example_windows.empty:
            LOGGER.warning(
                "Example window index resolved to zero rows; falling back to evaluation windows for examples."
            )
            example_windows = windows

    root = (preprocessed_root or (config.repo_root / "preprocessed_data")).resolve()
    summary_csv = _ensure_summary(config, root)
    if summary_csv is None or not summary_csv.exists():
        summary["status"] = "phase_z1_ready_missing_preprocessed_summary"
        return _export(phase_dir, summary, [], [], [])

    stream_map = _build_stream_map(pd.read_csv(summary_csv))
    enabled_modalities = tuple(mod for mod, enabled in sorted(baseline_cfg.modalities.items()) if _as_bool(enabled))
    if not enabled_modalities:
        summary["status"] = "phase_z1_ready_no_modalities_enabled"
        return _export(phase_dir, summary, [], [], [])

    enabled_devices = {
        _normalize_device_id(device)
        for device, enabled in sorted(baseline_cfg.devices.items())
        if _as_bool(enabled)
    }
    enabled_devices = {device for device in enabled_devices if device}
    if not enabled_devices:
        summary["status"] = "phase_z1_ready_no_devices_enabled"
        return _export(phase_dir, summary, [], [], [])

    LOGGER.info("[Stage] Building evaluation records from modality slices")
    records, activities = _build_records(
        windows,
        stream_map,
        enabled_modalities,
        allowed_devices=enabled_devices,
        progress_label="Build evaluation records",
    )
    if not records:
        summary["status"] = "phase_z1_ready_no_window_records"
        return _export(phase_dir, summary, [], [], [])

    if example_windows is windows:
        LOGGER.info("[Stage] Reusing evaluation records as example source")
        example_records_source = records
    else:
        LOGGER.info("[Stage] Building example-source records")
        example_records_source, _ = _build_records(
            example_windows,
            stream_map,
            enabled_modalities,
            allowed_devices=enabled_devices,
            progress_label="Build example records",
        )
        if not example_records_source:
            example_records_source = records

    normalizers = _build_normalizers(activities)
    participants = sorted({int(r["participant_id"]) for r in records})
    example_participants = sorted({int(r["participant_id"]) for r in example_records_source})

    provider = str(baseline_cfg.provider).strip().lower()
    wants_llm = provider not in {"", "heuristic", "offline", "none"}
    llm_predictor: _LLMConsensusPredictor | None = None
    backend_name = "heuristic_consensus"
    if wants_llm:
        LOGGER.info(
            "[Stage] Initializing LLM backend provider=%s model=%s with pool_size=%d",
            baseline_cfg.provider,
            baseline_cfg.model,
            int(baseline_cfg.max_concurrent_model_calls),
        )
        try:
            llm_predictor = _LLMConsensusPredictor(config, phase_dir)
            backend_name = "llm_consensus_agent"
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Could not initialize LLM ConSensus backend, falling back to heuristic: %s", exc)
            llm_predictor = None

    split_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    fold_metric_rows: list[dict[str, Any]] = []
    prediction_progress = _StageProgressBar(
        "Predict windows",
        total=len(records),
        logger=LOGGER,
    )

    for fold_offset, test_pid in enumerate(participants, start=1):
        fold_id = int(fold_offset - 1)
        fold_records = [r for r in records if int(r["participant_id"]) == test_pid]
        train_records = [
            r for r in example_records_source if int(r["participant_id"]) != test_pid
        ]
        train_participants = sorted(pid for pid in example_participants if pid != test_pid)
        fold_example_pool = _build_fold_example_pool(
            train_records,
            int(baseline_cfg.examples_per_class),
        )
        total_fold_examples = int(sum(len(v) for v in fold_example_pool.values()))
        split_rows.append(
            {
                "experiment": "consensus_zero_shot",
                "fold_id": int(fold_id),
                "test_participant_id": int(test_pid),
                "train_participants": json.dumps(train_participants),
                "num_train_windows": int(len(train_records)),
                "num_test_windows": int(len(fold_records)),
                "zero_shot": bool(int(baseline_cfg.examples_per_class) <= 0),
                "examples_per_class": int(baseline_cfg.examples_per_class),
                "num_fold_examples": total_fold_examples,
                "example_source_policy": summary["example_source_policy"],
                "enabled_devices": json.dumps(sorted(enabled_devices)),
            }
        )
        if not fold_records:
            continue
        LOGGER.info(
            "[Stage] Fold %d/%d test_pid=%d train_windows=%d test_windows=%d examples_per_class=%d",
            fold_offset,
            len(participants),
            int(test_pid),
            len(train_records),
            len(fold_records),
            int(baseline_cfg.examples_per_class),
        )

        fold_pred_rows: list[dict[str, Any]] = []
        for record_offset, record in enumerate(fold_records, start=1):
            row_backend = backend_name
            token_usage = _empty_token_usage()
            record_examples = _select_examples_for_record(
                record,
                fold_example_pool,
                int(baseline_cfg.examples_per_class),
            )
            try:
                if llm_predictor is not None:
                    pred_cont, votes, reason = llm_predictor.predict(
                        record,
                        examples=record_examples,
                    )
                    token_usage = llm_predictor.get_last_token_usage()
                else:
                    pred_cont, votes, reason = _predict_record_heuristic(record, normalizers)
            except Exception as exc:  # noqa: BLE001
                pred_cont, votes, reason = _predict_record_heuristic(record, normalizers)
                row_backend = "heuristic_consensus_fallback"
                reason = f"{reason}; llm_error={type(exc).__name__}: {exc}"
                token_usage = _empty_token_usage()

            pred_cont = _clip_1_to_5(pred_cont)
            pred_int = _round_1_to_5(pred_cont)
            y_true = int(record["label_true"])
            LOGGER.info(
                "[Window Result] window_id=%s y_true=%d y_pred=%d y_pred_cont=%.3f backend=%s",
                str(record["window_id"]),
                y_true,
                pred_int,
                float(pred_cont),
                row_backend,
            )
            LOGGER.info(
                "[Window Tokens] window_id=%s input_tokens=%d output_tokens=%d total_tokens=%d",
                str(record["window_id"]),
                int(token_usage.get("input_tokens", 0)),
                int(token_usage.get("output_tokens", 0)),
                int(token_usage.get("total_tokens", 0)),
            )
            fold_pred_rows.append(
                {
                    "experiment": "consensus_zero_shot",
                    "backend": row_backend,
                    "fold_id": int(fold_id),
                    "test_participant_id": int(test_pid),
                    "window_id": str(record["window_id"]),
                    "session_key": str(record["session_key"]),
                    "participant_id": int(record["participant_id"]),
                    "event_id": str(record["event_id"]),
                    "video_uid": str(record["video_uid"]),
                    "y_true": int(y_true),
                    "y_pred_continuous": float(pred_cont),
                    "y_pred": int(pred_int),
                    "abs_error": float(abs(float(y_true) - pred_cont)),
                    "modalities": json.dumps(sorted(record["modality_activity"].keys())),
                    "modality_votes": json.dumps(votes, sort_keys=True),
                    "modality_devices": json.dumps(record.get("modality_devices", {}), sort_keys=True),
                    "num_examples": int(len(record_examples)),
                    "input_tokens": int(token_usage.get("input_tokens", 0)),
                    "output_tokens": int(token_usage.get("output_tokens", 0)),
                    "total_tokens": int(token_usage.get("total_tokens", 0)),
                    "token_usage_by_agent": json.dumps(token_usage.get("by_agent", {}), sort_keys=True),
                    "example_participants": json.dumps(
                        sorted({int(ex["participant_id"]) for ex in record_examples})
                    ),
                    "example_labels": json.dumps(
                        [int(ex["label_true"]) for ex in record_examples]
                    ),
                    "reason": reason,
                }
            )
            _log_live_metrics(
                f"fold={fold_id}|test_pid={test_pid}|fold_running",
                fold_pred_rows,
            )
            _log_live_metrics(
                f"global|processed_windows={len(prediction_rows) + len(fold_pred_rows)}",
                [*prediction_rows, *fold_pred_rows],
            )
            prediction_progress.update(
                context=(
                    f"fold={fold_offset}/{len(participants)} "
                    f"pid={int(test_pid)} "
                    f"window={record_offset}/{len(fold_records)}"
                )
            )

        prediction_rows.extend(fold_pred_rows)
        fold_df = pd.DataFrame(fold_pred_rows)
        fold_metrics, fold_cm = _compute_metrics(fold_df)
        fold_metric_rows.append(
            {
                "experiment": "consensus_zero_shot",
                "backend": backend_name,
                "fold_id": int(fold_id),
                "test_participant_id": int(test_pid),
                **fold_metrics,
            }
        )
        fold_cm_path = phase_dir / "confusion" / f"fold_{fold_id}__participant_{test_pid}.csv"
        fold_cm_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(fold_cm, index=LABEL_ORDER, columns=LABEL_ORDER).to_csv(fold_cm_path, index=True)
    prediction_progress.close(context="all folds complete")

    summary["status"] = "phase_z1_completed" if prediction_rows else "phase_z1_ready_no_predictions"
    summary["backend"] = backend_name
    summary["num_windows_considered"] = int(len(records))
    summary["window_prediction_coverage"] = float(len(prediction_rows) / len(records)) if records else 0.0
    if prediction_rows:
        errors = [float(row["abs_error"]) for row in prediction_rows]
        summary["prediction_abs_error_mean"] = float(np.mean(errors))
        summary["prediction_abs_error_median"] = float(np.median(errors))
        summary["input_tokens_total"] = int(sum(int(row.get("input_tokens", 0)) for row in prediction_rows))
        summary["output_tokens_total"] = int(sum(int(row.get("output_tokens", 0)) for row in prediction_rows))
        summary["total_tokens_total"] = int(sum(int(row.get("total_tokens", 0)) for row in prediction_rows))
        summary["tokens_per_window_mean"] = float(
            np.mean([float(row.get("total_tokens", 0)) for row in prediction_rows])
        )

    return _export(phase_dir, summary, split_rows, prediction_rows, fold_metric_rows)
