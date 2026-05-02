"""Helpers for Polar ECG packet-style timestamp reconstruction."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .timestamp_repair import reconstruct_constant_timestamp_packets

DEFAULT_ECG_SAMPLE_STEP_MS = 1000.0 / 130.0


def reconstruct_polar_ecg_timestamps(
    raw_timestamps: pd.Series,
    *,
    default_sample_step: float | None = None,
) -> pd.Series:
    step = DEFAULT_ECG_SAMPLE_STEP_MS if default_sample_step is None else float(default_sample_step)
    return reconstruct_constant_timestamp_packets(raw_timestamps, default_step=step)


def read_polar_ecg_timestamp_series(path: Path) -> pd.Series:
    df = pd.read_csv(path, usecols=["sys_time"])
    return reconstruct_polar_ecg_timestamps(df["sys_time"])
