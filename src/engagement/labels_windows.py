from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import RunConfig
from .io_utils import read_table_with_fallback, write_table_with_fallback

LOGGER = logging.getLogger(__name__)


LABEL_VALUES_5CLASS = (1, 2, 3, 4, 5)

AVAILABILITY_MODALITIES = (
    "eeg",
    "acc",
    "gyro",
    "ppg",
    "ring",
    "ecg",
    "eda",
    "hr",
    "eye",
    "markers",
    "esense",
)

REQUIRED_NORMALIZED_EVENT_COLUMNS = {
    "session_key",
    "participant_id",
    "video_uid",
    "video_timestamp_sec",
    "system_time_sec_norm",
    "engagement_report",
}

REQUIRED_VIDEO_OFFSET_COLUMNS = {
    "session_key",
    "participant_id",
    "video_uid",
    "video_start_time_sec",
    "offset_mad_sec",
    "num_valid_offsets",
}


@dataclass(frozen=True)
class Interval:
    interval_id: str
    session_key: str
    participant_id: int
    video_uid: str
    event_id: str
    label_5class: int
    start_video_sec: float
    end_video_sec: float



def parse_report_value(raw_value: Any) -> int | str | None:
    if pd.isna(raw_value):
        return None

    text = str(raw_value).strip()
    if not text:
        return None

    if text.upper() == "X":
        return "X"

    try:
        number = float(text)
    except ValueError:
        return None

    if not np.isfinite(number):
        return None

    rounded = int(round(number))
    if rounded in {1, 2, 3, 4, 5} and abs(number - rounded) < 1e-6:
        return rounded
    return None



def _parse_bool_flag(value: Any) -> bool:
    if pd.isna(value):
        return False

    text = str(value).strip().lower()
    if not text:
        return False

    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n", "na", "n/a", "none", "null"}:
        return False

    number = pd.to_numeric(text, errors="coerce")
    if pd.notna(number):
        return bool(float(number) != 0.0)
    return False


def _is_video_excluded(video_uid: str, patterns: tuple[str, ...]) -> bool:
    uid = (video_uid or "").lower()
    return any(pattern.lower() in uid for pattern in patterns)



def _build_event_rows(
    session_row: dict[str, Any],
    engagement_df: pd.DataFrame,
    video_start_map: dict[str, float],
    config: RunConfig,
) -> list[dict[str, Any]]:
    session_key = str(session_row["session_key"])
    participant_id = int(session_row["participant_id"])

    rows: list[dict[str, Any]] = []
    event_counter = 0

    for row in engagement_df.itertuples(index=False):
        parsed = parse_report_value(getattr(row, "engagement_report"))
        if parsed is None:
            continue

        video_uid = str(getattr(row, "video_uid"))
        video_time = pd.to_numeric(getattr(row, "video_timestamp_sec"), errors="coerce")
        event_time_sec = pd.to_numeric(getattr(row, "system_time_sec_norm"), errors="coerce")
        is_x = parsed == "X"
        label_5class = int(parsed) if isinstance(parsed, int) and parsed in LABEL_VALUES_5CLASS else None

        event_id = f"{session_key}:event:{event_counter}"
        event_counter += 1

        video_start_time = video_start_map.get(video_uid, np.nan)
        is_excluded_video = _is_video_excluded(video_uid, config.exclude_video_uid_patterns)
        is_pulsed = _parse_bool_flag(getattr(row, "is_pulsed", False))
        is_valid_supervised = (
            (not is_x)
            and (label_5class is not None)
            and (not is_excluded_video)
            and (not is_pulsed)
        )

        time_basis = (
            "absolute_system"
            if pd.notna(event_time_sec) and pd.notna(video_start_time)
            else "video_relative"
        )

        rows.append(
            {
                "event_id": event_id,
                "session_key": session_key,
                "participant_id": participant_id,
                "video_uid": video_uid,
                "report_raw": str(parsed),
                "label_5class": label_5class,
                "label_3class": str(label_5class) if label_5class is not None else None,
                "is_valid_supervised": bool(is_valid_supervised),
                "is_excluded_video": bool(is_excluded_video),
                "is_pulsed": bool(is_pulsed),
                "event_time_video_sec": float(video_time) if pd.notna(video_time) else np.nan,
                "event_time_sec": float(event_time_sec) if pd.notna(event_time_sec) else np.nan,
                "video_start_time_sec": float(video_start_time)
                if pd.notna(video_start_time)
                else np.nan,
                "time_basis": time_basis,
                "event_source": "engagement_log",
            }
        )

    return rows



def _build_intervals(events_df: pd.DataFrame, config: RunConfig) -> list[Interval]:
    intervals: list[Interval] = []

    supervised = events_df[events_df["is_valid_supervised"]].copy()
    if supervised.empty:
        return intervals

    for (session_key, participant_id, video_uid), group in supervised.groupby(
        ["session_key", "participant_id", "video_uid"], sort=False
    ):
        ordered = group.sort_values("event_time_video_sec", kind="mergesort")
        previous_time: float | None = None
        interval_counter = 0

        for row in ordered.itertuples(index=False):
            current_t = float(row.event_time_video_sec)
            if config.window_generation_mode == "event_trailing":
                start_t = current_t - float(config.window_size_sec)
                end_t = current_t
            else:
                start_t = 0.0 if previous_time is None else previous_time
                end_t = current_t

            previous_time = current_t

            if not np.isfinite(start_t) or not np.isfinite(end_t):
                continue
            if end_t <= start_t:
                continue

            interval_id = f"{session_key}:{video_uid}:interval:{interval_counter}"
            interval_counter += 1
            intervals.append(
                Interval(
                    interval_id=interval_id,
                    session_key=str(session_key),
                    participant_id=int(participant_id),
                    video_uid=str(video_uid),
                    event_id=str(row.event_id),
                    label_5class=int(row.label_5class),
                    start_video_sec=float(start_t),
                    end_video_sec=float(end_t),
                )
            )

    return intervals



def _generate_interval_windows(
    interval: Interval,
    config: RunConfig,
) -> list[tuple[float, float, bool]]:
    length = interval.end_video_sec - interval.start_video_sec
    if length <= 0:
        return []

    windows: list[tuple[float, float, bool]] = []
    w = config.window_size_sec
    s = config.stride_sec

    if length >= w:
        t = interval.start_video_sec
        while t + w <= interval.end_video_sec + 1e-9:
            windows.append((float(t), float(t + w), False))
            t += s
        return windows

    if config.short_interval_policy == "pad":
        # Keep a truncated window placeholder for downstream padding logic.
        windows.append((interval.start_video_sec, interval.end_video_sec, True))

    return windows



def _availability_flags_from_manifest(session_row: dict[str, Any]) -> dict[str, bool]:
    return {
        f"has_{modality}": bool(session_row.get(f"has_{modality}", False))
        for modality in AVAILABILITY_MODALITIES
    }



def _validate_phase_b_engagement_artifacts(
    normalized_events_df: pd.DataFrame,
    video_offsets_df: pd.DataFrame,
) -> None:
    missing_events = REQUIRED_NORMALIZED_EVENT_COLUMNS - set(normalized_events_df.columns)
    if missing_events:
        raise ValueError(
            "engagement_events_normalized artifact missing columns: "
            f"{sorted(missing_events)}"
        )

    missing_offsets = REQUIRED_VIDEO_OFFSET_COLUMNS - set(video_offsets_df.columns)
    if missing_offsets:
        raise ValueError(
            "video_start_offsets artifact missing columns: "
            f"{sorted(missing_offsets)}"
        )

    event_sessions = set(normalized_events_df["session_key"].dropna().astype(str).tolist())
    offset_sessions = set(video_offsets_df["session_key"].dropna().astype(str).tolist())
    missing_offset_sessions = sorted(event_sessions - offset_sessions)
    if missing_offset_sessions:
        raise ValueError(
            "Missing offset mappings for sessions present in normalized engagement events: "
            f"{missing_offset_sessions}"
        )



def _session_event_inputs(
    *,
    session_row: dict[str, Any],
    normalized_events_df: pd.DataFrame,
    video_offsets_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float]]:
    session_key = str(session_row["session_key"])

    session_events_df = normalized_events_df[
        normalized_events_df["session_key"].astype(str) == session_key
    ].copy()
    session_offsets_df = video_offsets_df[
        video_offsets_df["session_key"].astype(str) == session_key
    ].copy()

    video_start_map: dict[str, float] = {}
    for row in session_offsets_df.to_dict(orient="records"):
        video_uid = str(row.get("video_uid", ""))
        video_start = pd.to_numeric(row.get("video_start_time_sec"), errors="coerce")
        if video_uid and pd.notna(video_start):
            video_start_map[video_uid] = float(video_start)

    return session_events_df, video_start_map



def build_report_events_and_windows(
    config: RunConfig,
    session_manifest: pd.DataFrame,
    normalized_events_df: pd.DataFrame,
    video_offsets_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    event_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []

    _validate_phase_b_engagement_artifacts(normalized_events_df, video_offsets_df)

    for session_row in session_manifest.to_dict(orient="records"):
        participant_id = int(session_row["participant_id"])
        session_key = str(session_row["session_key"])

        engagement_df, video_start_map = _session_event_inputs(
            session_row=session_row,
            normalized_events_df=normalized_events_df,
            video_offsets_df=video_offsets_df,
        )
        if engagement_df.empty:
            continue

        session_events = _build_event_rows(session_row, engagement_df, video_start_map, config)
        event_rows.extend(session_events)

        events_df = pd.DataFrame(session_events)
        if events_df.empty:
            continue

        intervals = _build_intervals(events_df, config)
        availability_flags = _availability_flags_from_manifest(session_row)

        if (
            config.absolute_time_mode
            and config.exclude_p2_in_absolute_mode
            and participant_id == 2
        ):
            LOGGER.info(
                "Skipping absolute-time windows for %s due to P2 policy.",
                session_key,
            )
            continue

        alignment_ok = bool(session_row.get("alignment_ok", False))

        for interval in intervals:
            windows = _generate_interval_windows(interval, config)
            for win_idx, (start_v, end_v, is_padded) in enumerate(windows):
                duration = end_v - start_v
                if duration <= 0:
                    continue

                video_start = video_start_map.get(interval.video_uid, np.nan)
                abs_start = np.nan
                abs_end = np.nan
                if pd.notna(video_start):
                    abs_start = float(video_start + start_v)
                    abs_end = float(video_start + end_v)

                if config.absolute_time_mode and (not alignment_ok or pd.isna(abs_start) or pd.isna(abs_end)):
                    # In primary mode, windows must be mappable to absolute time.
                    continue

                time_basis = (
                    "absolute_system_slice"
                    if pd.notna(abs_start) and pd.notna(abs_end)
                    else "video_relative_labels"
                )

                window_rows.append(
                    {
                        "window_id": f"{interval.interval_id}:window:{win_idx}",
                        "session_key": interval.session_key,
                        "participant_id": interval.participant_id,
                        "video_uid": interval.video_uid,
                        "event_id": interval.event_id,
                        "interval_id": interval.interval_id,
                        "label_5class": int(interval.label_5class),
                        "label_3class": str(interval.label_5class),
                        "t_start_video_sec": start_v,
                        "t_end_video_sec": end_v,
                        "t_start_sec": abs_start,
                        "t_end_sec": abs_end,
                        "window_size_sec": duration,
                        "stride_sec": config.stride_sec,
                        "window_generation_mode": config.window_generation_mode,
                        "time_basis": time_basis,
                        "is_padded_window": bool(is_padded),
                        "split_group": f"participant_{interval.participant_id}",
                        **availability_flags,
                    }
                )

    report_events_df = pd.DataFrame(event_rows)
    window_index_df = pd.DataFrame(window_rows)
    return report_events_df, window_index_df



def run_phase_c(
    config: RunConfig,
    session_manifest: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Path]]:
    config.ensure_directories()

    if session_manifest is None:
        manifest_path = config.manifests_dir / "session_manifest.parquet"
        session_manifest = read_table_with_fallback(manifest_path)
        if session_manifest.empty:
            raise FileNotFoundError(
                "Session manifest was not found or empty. Run phase B first. "
                f"Looked for: {manifest_path}"
            )

    normalized_events_path = config.manifests_dir / "engagement_events_normalized.parquet"
    video_offsets_path = config.manifests_dir / "video_start_offsets.parquet"

    has_events_artifact = normalized_events_path.exists() or normalized_events_path.with_suffix(".csv").exists()
    has_offsets_artifact = video_offsets_path.exists() or video_offsets_path.with_suffix(".csv").exists()

    if not has_events_artifact or not has_offsets_artifact:
        raise FileNotFoundError(
            "Phase C requires Phase B normalized engagement artifacts. Missing one or both of "
            "engagement_events_normalized and video_start_offsets under artifacts/manifests/."
        )

    normalized_events_df = read_table_with_fallback(normalized_events_path)
    video_offsets_df = read_table_with_fallback(video_offsets_path)

    report_events_df, window_index_df = build_report_events_and_windows(
        config,
        session_manifest,
        normalized_events_df=normalized_events_df,
        video_offsets_df=video_offsets_df,
    )

    report_events_path = write_table_with_fallback(
        report_events_df,
        config.manifests_dir / "report_events.parquet",
        logger=LOGGER,
    )
    window_index_path = write_table_with_fallback(
        window_index_df,
        config.windows_dir / "window_index.parquet",
        logger=LOGGER,
    )

    run_report_events_path = write_table_with_fallback(
        report_events_df,
        config.run_dir / "manifests" / "report_events.parquet",
        logger=LOGGER,
    )
    run_window_index_path = write_table_with_fallback(
        window_index_df,
        config.run_dir / "windows" / "window_index.parquet",
        logger=LOGGER,
    )

    return report_events_df, window_index_df, {
        "report_events": report_events_path,
        "window_index": window_index_path,
        "run_report_events": run_report_events_path,
        "run_window_index": run_window_index_path,
    }



