"""Export supervised, time-synchronized preprocessed modality CSVs."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import RunConfig
from .io_utils import read_table_with_fallback
from .loaders import load_modality_csv

LOGGER = logging.getLogger(__name__)

def _select_export_columns(filtered: pd.DataFrame, modality: str) -> pd.DataFrame:
    """Apply modality-specific export column selection."""
    if modality != "eeg":
        return filtered
    column_map = {str(col).lower(): col for col in filtered.columns}
    preferred = ("timestamp", "af7", "af8", "t_sec")
    keep = [column_map[name] for name in preferred if name in column_map]
    if not keep:
        return filtered
    return filtered.loc[:, keep].copy()


def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    cleaned = [(float(s), float(e)) for s, e in intervals if np.isfinite(s) and np.isfinite(e) and e > s]
    if not cleaned:
        return []

    cleaned.sort(key=lambda pair: pair[0])
    merged: list[list[float]] = [[cleaned[0][0], cleaned[0][1]]]
    for start, end in cleaned[1:]:
        last = merged[-1]
        if start <= last[1]:
            last[1] = max(last[1], end)
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged]


def _interval_mask(t_sec: pd.Series, intervals: list[tuple[float, float]]) -> np.ndarray:
    t = pd.to_numeric(t_sec, errors="coerce").to_numpy(dtype=float)
    if len(intervals) == 0:
        return np.zeros_like(t, dtype=bool)

    mask = np.zeros_like(t, dtype=bool)
    for start, end in intervals:
        mask |= (t >= start) & (t < end)
    mask &= np.isfinite(t)
    return mask


def _window_has_samples(session_windows: pd.DataFrame, t_sec: pd.Series) -> np.ndarray:
    if session_windows.empty:
        return np.zeros((0,), dtype=bool)

    starts = pd.to_numeric(session_windows["t_start_sec"], errors="coerce").to_numpy(dtype=float)
    ends = pd.to_numeric(session_windows["t_end_sec"], errors="coerce").to_numpy(dtype=float)
    valid_bounds = np.isfinite(starts) & np.isfinite(ends) & (ends > starts)

    ts = pd.to_numeric(t_sec, errors="coerce").to_numpy(dtype=float)
    ts = ts[np.isfinite(ts)]
    if ts.size == 0:
        return np.zeros((len(session_windows),), dtype=bool)

    ts = np.sort(ts, kind="mergesort")
    left = np.searchsorted(ts, starts, side="left")
    right = np.searchsorted(ts, ends, side="left")
    has = (right - left) > 0
    has &= valid_bounds
    return has


def _window_modalities(window_row: pd.Series) -> dict[str, bool]:
    has_cols = [c for c in window_row.index if str(c).startswith("has_")]
    if has_cols:
        return {str(c)[4:]: bool(window_row.get(c, False)) for c in has_cols}

    raw_mask = window_row.get("availability_mask")
    if isinstance(raw_mask, str) and raw_mask.strip():
        try:
            parsed = json.loads(raw_mask)
            if isinstance(parsed, dict):
                return {str(k): bool(v) for k, v in parsed.items()}
        except json.JSONDecodeError:
            return {}
    return {}


def _session_modalities_for_export(
    session_row: dict[str, Any],
    session_windows: pd.DataFrame,
    enabled_modalities: dict[str, bool],
) -> list[str]:
    candidate: set[str] = set()

    if not session_windows.empty:
        for row in session_windows.to_dict(orient="records"):
            mask = _window_modalities(pd.Series(row))
            for modality, available in mask.items():
                if available:
                    candidate.add(str(modality))

    if not candidate:
        for key, value in session_row.items():
            key_s = str(key)
            if key_s.startswith("has_") and bool(value):
                candidate.add(key_s[4:])

    out: list[str] = []
    for modality in sorted(candidate):
        if not enabled_modalities.get(modality, False):
            continue
        if not session_row.get(f"{modality}_path"):
            continue
        out.append(modality)
    return out


def _safe_output_filename(source_path: Path) -> str:
    if source_path.suffix.lower() == ".bin":
        return f"{source_path.stem}.csv"
    if source_path.suffix.lower() == ".csv":
        return source_path.name
    return f"{source_path.stem}.csv"


def build_preprocessed_data_folder(
    config: RunConfig,
    *,
    session_manifest: pd.DataFrame | None = None,
    window_index: pd.DataFrame | None = None,
    output_root: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Path]]:
    if session_manifest is None:
        session_manifest = read_table_with_fallback(config.manifests_dir / "session_manifest.parquet")
    if window_index is None:
        window_index = read_table_with_fallback(config.windows_dir / "window_index.parquet")

    if session_manifest.empty:
        raise FileNotFoundError("Session manifest is empty or missing. Run phase B before preprocessed export.")
    if window_index.empty:
        raise FileNotFoundError("Window index is empty or missing. Run phase C before preprocessed export.")

    out_root = (output_root or (config.repo_root / "preprocessed_data")).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    windows = window_index.copy()
    if "label_5class" in windows.columns:
        windows["label_5class"] = pd.to_numeric(windows.get("label_5class"), errors="coerce")
    else:
        windows["label_5class"] = pd.to_numeric(windows.get("label_3class"), errors="coerce")
    windows["t_start_sec"] = pd.to_numeric(windows.get("t_start_sec"), errors="coerce")
    windows["t_end_sec"] = pd.to_numeric(windows.get("t_end_sec"), errors="coerce")

    supervised = windows[
        windows["label_5class"].isin([1, 2, 3, 4, 5])
        & windows["t_start_sec"].notna()
        & windows["t_end_sec"].notna()
        & (windows["t_end_sec"] > windows["t_start_sec"])
    ].copy()

    summary_rows: list[dict[str, Any]] = []

    for session_row in session_manifest.to_dict(orient="records"):
        session_key = str(session_row.get("session_key", ""))
        participant_id = int(pd.to_numeric(session_row.get("participant_id"), errors="coerce"))
        session_id = int(pd.to_numeric(session_row.get("session_id"), errors="coerce"))

        session_windows = supervised[supervised["session_key"].astype(str) == session_key].copy()
        intervals = _merge_intervals(
            list(
                zip(
                    pd.to_numeric(session_windows["t_start_sec"], errors="coerce"),
                    pd.to_numeric(session_windows["t_end_sec"], errors="coerce"),
                )
            )
        )

        participant_dir = out_root / f"P{participant_id}"
        participant_dir.mkdir(parents=True, exist_ok=True)

        window_modalities = {
            str(c)[4:] for c in session_windows.columns if str(c).startswith("has_")
        }
        for key, value in session_row.items():
            key_s = str(key)
            if key_s.startswith("has_") and bool(value):
                window_modalities.add(key_s[4:])
        window_modalities = {
            m for m in window_modalities if config.enabled_modalities.get(m, False)
        }

        loaded_modalities: dict[str, pd.DataFrame] = {}
        for modality in sorted(window_modalities):
            source_path_value = session_row.get(f"{modality}_path")
            if not source_path_value:
                continue
            source_path = Path(str(source_path_value))
            if not source_path.exists():
                continue
            try:
                loaded_modalities[modality] = load_modality_csv(source_path, modality)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning(
                    "Could not pre-load %s (%s) for window missingness stats: %s",
                    session_key,
                    modality,
                    exc,
                )

        if not session_windows.empty:
            for modality in sorted(window_modalities):
                has_col = f"has_{modality}"
                modality_df = loaded_modalities.get(modality)
                if modality_df is None or "t_sec" not in modality_df.columns:
                    session_windows[has_col] = False
                else:
                    session_windows[has_col] = _window_has_samples(
                        session_windows,
                        modality_df["t_sec"],
                    )

            window_name = f"{participant_id}_{session_id}_supervised_windows.csv"
            session_windows.to_csv(participant_dir / window_name, index=False)

        modalities = _session_modalities_for_export(
            session_row,
            session_windows,
            enabled_modalities=config.enabled_modalities,
        )

        for modality in modalities:
            source_path_value = session_row.get(f"{modality}_path")
            if not source_path_value:
                continue
            source_path = Path(str(source_path_value))
            if not source_path.exists():
                continue

            df = loaded_modalities.get(modality)
            if df is None:
                try:
                    df = load_modality_csv(source_path, modality)
                except Exception as exc:  # noqa: BLE001
                    LOGGER.warning("Skipping %s (%s) due to load error: %s", session_key, modality, exc)
                    continue

            if "t_sec" not in df.columns:
                continue

            keep = _interval_mask(df["t_sec"], intervals)
            filtered = df.loc[keep].copy()
            if filtered.empty:
                continue

            drop_cols = [c for c in filtered.columns if str(c).startswith("_")]
            if drop_cols:
                filtered = filtered.drop(columns=drop_cols)

            filtered = _select_export_columns(filtered, modality)
            filtered = filtered.sort_values("t_sec", kind="mergesort").reset_index(drop=True)
            out_name = _safe_output_filename(source_path)
            out_path = participant_dir / out_name
            filtered.to_csv(out_path, index=False)

            summary_rows.append(
                {
                    "session_key": session_key,
                    "participant_id": participant_id,
                    "session_id": session_id,
                    "modality": modality,
                    "source_path": str(source_path),
                    "output_path": str(out_path),
                    "rows_written": int(len(filtered)),
                    "interval_count": int(len(intervals)),
                    "t_start_sec": float(pd.to_numeric(filtered["t_sec"], errors="coerce").min()),
                    "t_end_sec": float(pd.to_numeric(filtered["t_sec"], errors="coerce").max()),
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = out_root / "export_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    outputs = {
        "preprocessed_root": out_root,
        "summary": summary_path,
    }
    return summary_df, outputs


def run_phase_preprocessed_data(
    config: RunConfig,
    *,
    session_manifest: pd.DataFrame | None = None,
    window_index: pd.DataFrame | None = None,
    output_root: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Path]]:
    config.ensure_directories()
    return build_preprocessed_data_folder(
        config,
        session_manifest=session_manifest,
        window_index=window_index,
        output_root=output_root,
    )

