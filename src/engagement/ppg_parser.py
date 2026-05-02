"""Robust parsing helpers for malformed PPG CSV exports.

Some device exports contain rows with many more values than header columns.
This module enforces a stable interpretation:
- column 1 (0-indexed 0) is timestamp
- remaining columns are candidate signal channels
- by default, keep active non-zero signal columns (up to max_signal_cols), mapped to ppg1..ppgN
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd


def _to_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def read_ppg_timestamp_series(path: Path) -> pd.Series:
    """Read timestamp as the first field from each data row."""
    timestamps: list[float] = []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as fp:
        reader = csv.reader(fp)
        # header
        try:
            _ = next(reader)
        except StopIteration:
            return pd.Series(dtype=float)

        for row in reader:
            if not row:
                continue
            timestamps.append(_to_float(row[0]))

    return pd.to_numeric(pd.Series(timestamps), errors="coerce")


def load_ppg_dataframe(
    path: Path,
    *,
    max_signal_cols: int = 8,
    min_abs_nonzero: float = 1e-12,
    use_first_n_signal_cols: int | None = None,
) -> pd.DataFrame:
    """Load malformed PPG CSV robustly.

    Returns DataFrame with columns:
    - timestamp
    - ppg1..ppgN (N <= max_signal_cols)

    If use_first_n_signal_cols is provided, channels are selected strictly by
    order from the raw stream (1..N) instead of active/non-zero detection.
    """
    timestamps: list[float] = []
    signal_rows: list[list[float]] = []

    with path.open("r", encoding="utf-8", errors="replace", newline="") as fp:
        reader = csv.reader(fp)
        try:
            _header = next(reader)
        except StopIteration:
            return pd.DataFrame(columns=["timestamp", "ppg1", "ppg2", "ppg3"])

        for row in reader:
            if not row:
                continue
            timestamps.append(_to_float(row[0]))
            signal_rows.append([_to_float(v) for v in row[1:]])

    n_rows = len(signal_rows)
    if n_rows == 0:
        return pd.DataFrame(columns=["timestamp", "ppg1", "ppg2", "ppg3"])

    n_sig = max((len(r) for r in signal_rows), default=0)
    sig = np.full((n_rows, n_sig), np.nan, dtype=float)
    for i, row in enumerate(signal_rows):
        if not row:
            continue
        length = min(len(row), n_sig)
        sig[i, :length] = np.asarray(row[:length], dtype=float)

    if use_first_n_signal_cols is not None and use_first_n_signal_cols > 0:
        first_n = int(use_first_n_signal_cols)
        selected_idx = list(range(min(n_sig, first_n, max_signal_cols)))
    else:
        finite = np.isfinite(sig)
        nonzero = np.abs(sig) > min_abs_nonzero
        active_mask = np.any(finite & nonzero, axis=0) if n_sig > 0 else np.array([], dtype=bool)
        active_idx = np.where(active_mask)[0].tolist()

        # Fallback: keep first channels if all channels appear zero.
        if not active_idx and n_sig > 0:
            active_idx = list(range(min(max_signal_cols, n_sig)))

        selected_idx = active_idx[:max_signal_cols]

    out: dict[str, np.ndarray] = {
        "timestamp": pd.to_numeric(pd.Series(timestamps), errors="coerce").to_numpy(dtype=float)
    }
    for rank, idx in enumerate(selected_idx, start=1):
        out[f"ppg{rank}"] = sig[:, idx]

    # Guarantee at least 3 channels for downstream schema expectations.
    for rank in range(1, 4):
        key = f"ppg{rank}"
        if key not in out:
            out[key] = np.zeros((n_rows,), dtype=float)

    df = pd.DataFrame(out)
    df.attrs["ppg_selected_signal_indices"] = selected_idx
    return df
