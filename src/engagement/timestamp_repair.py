"""Timestamp repair utilities for packetized exports with repeated timestamps."""

from __future__ import annotations

import numpy as np
import pandas as pd


def reconstruct_constant_timestamp_packets(
    raw_timestamps: pd.Series,
    *,
    default_step: float | None = None,
) -> pd.Series:
    """Expand repeated packet-level timestamps into per-row timestamps.

    For each run of equal timestamp with length N and next run timestamp T_next:
      step = (T_next - T_current) / N
      repaired[k] = T_current + k * step,  k=0..N-1

    This allows per-packet sampling rate to vary naturally.
    """
    numeric = pd.to_numeric(pd.Series(raw_timestamps), errors="coerce")
    values = numeric.to_numpy(dtype=float)
    n = len(values)
    if n == 0:
        return numeric

    runs: list[tuple[int, int, float]] = []
    i = 0
    while i < n:
        v = values[i]
        if not np.isfinite(v):
            i += 1
            continue

        j = i + 1
        while j < n and values[j] == v:
            j += 1
        runs.append((i, j, float(v)))
        i = j

    if not runs:
        return numeric

    step_candidates: list[float] = []
    for run_idx in range(len(runs) - 1):
        start, end, current_t = runs[run_idx]
        _next_start, _next_end, next_t = runs[run_idx + 1]
        count = end - start
        span = next_t - current_t
        if count > 0 and np.isfinite(span) and span > 0:
            step_candidates.append(float(span / count))

    if default_step is not None:
        fallback_step = float(default_step)
    elif step_candidates:
        fallback_step = float(np.median(step_candidates))
    else:
        fallback_step = 1.0

    repaired = np.full(n, np.nan, dtype=float)
    for run_idx, (start, end, current_t) in enumerate(runs):
        count = end - start
        if count <= 0:
            continue

        if run_idx < len(runs) - 1:
            _next_start, _next_end, next_t = runs[run_idx + 1]
            span = next_t - current_t
            if np.isfinite(span) and span > 0:
                step = float(span / count)
            else:
                step = fallback_step
        else:
            step = fallback_step

        repaired[start:end] = current_t + (np.arange(count, dtype=float) * step)

    return pd.Series(repaired, index=numeric.index, dtype=float)
