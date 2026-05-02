from __future__ import annotations

from dataclasses import dataclass
import logging
import math
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

from .config import RunConfig
from .io_utils import write_table_with_fallback
from .ecg_parser import read_polar_ecg_timestamp_series
from .ppg_parser import read_ppg_timestamp_series
from .ring_parser import read_ring_timestamp_series
from .timestamp_repair import reconstruct_constant_timestamp_packets

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class StreamSpec:
    file_suffix: str
    timestamp_column: str
    required: bool = False
    device_name: str = "unknown"


STREAM_SPECS: dict[str, StreamSpec] = {
    "eeg": StreamSpec("_EEG.csv", "timestamp", device_name="muse_headband"),
    "acc": StreamSpec("_ACC.csv", "timestamp", device_name="muse_headband"),
    "gyro": StreamSpec("_GYRO.csv", "timestamp", device_name="muse_headband"),
    "ppg": StreamSpec("_PPG.csv", "timestamp", device_name="muse_headband"),
    "ring": StreamSpec(".bin", "timestamp", device_name="ring"),
    "ecg": StreamSpec("_Polar_ECG.csv", "sys_time", device_name="polar_chest_strap"),
    "eda": StreamSpec("_msband_gsr.csv", "SystemTime", device_name="msband"),
    "hr": StreamSpec("_msband_hr.csv", "SystemTime", device_name="msband"),
    "eye": StreamSpec("_BeamEyeTracker.csv", "sys_time", device_name="beam_eye_tracker"),
    "markers": StreamSpec("_MARKERS.csv", "timestamp", device_name="muse_headband"),
    "engagement_log": StreamSpec("_engagement_log.csv", "system_time_sec", required=True, device_name="experiment_player"),
    "esense": StreamSpec("_eSense.csv", "timestamp", device_name="esense"),
}

SESSION_PATTERN = re.compile(r"^(?P<participant>\d+)_(?P<session>\d+)_engagement_log\.csv$")



def _write_table(df: pd.DataFrame, parquet_path: Path) -> Path:
    return write_table_with_fallback(df, parquet_path, logger=LOGGER)


def detect_and_normalize_timestamps(values: pd.Series) -> tuple[pd.Series, str]:
    numeric = pd.to_numeric(pd.Series(values), errors="coerce")
    valid = numeric.dropna()
    if valid.empty:
        return numeric, "unknown"

    median_abs = float(valid.abs().median())
    if median_abs > 1e11:
        return numeric / 1000.0, "milliseconds"
    if 1e9 <= median_abs <= 1e10:
        return numeric, "seconds"
    return numeric, "unknown"



def compute_timestamp_qc(ts_sec: pd.Series, detected_unit: str) -> dict[str, Any]:
    numeric = pd.to_numeric(pd.Series(ts_sec), errors="coerce")
    valid = numeric.dropna()
    if valid.empty:
        return {
            "detected_unit": detected_unit,
            "num_rows": int(len(numeric)),
            "num_valid": 0,
            "monotonicity_ratio": np.nan,
            "unique_ratio": np.nan,
            "plausible_date_ratio": np.nan,
            "median_dt_sec": np.nan,
            "start_sec": np.nan,
            "end_sec": np.nan,
            "timestamp_unit_ok": False,
        }

    diffs = valid.diff().dropna()
    monotonicity_ratio = float((diffs >= 0).mean()) if len(diffs) else 1.0
    unique_ratio = float(valid.nunique(dropna=True) / len(valid))

    lower = 1_546_300_800  # 2019-01-01 UTC
    upper = 2_050_000_000  # year ~2034
    plausible_date_ratio = float(((valid >= lower) & (valid <= upper)).mean())

    positive_diffs = diffs[diffs > 0]
    median_dt_sec = float(positive_diffs.median()) if not positive_diffs.empty else 0.0

    return {
        "detected_unit": detected_unit,
        "num_rows": int(len(numeric)),
        "num_valid": int(len(valid)),
        "monotonicity_ratio": monotonicity_ratio,
        "unique_ratio": unique_ratio,
        "plausible_date_ratio": plausible_date_ratio,
        "median_dt_sec": median_dt_sec,
        "start_sec": float(valid.min()),
        "end_sec": float(valid.max()),
        "timestamp_unit_ok": detected_unit in {"milliseconds", "seconds"},
    }



def _normalize_colname(name: str) -> str:
    return " ".join(str(name).strip().lower().replace("_", " ").replace("-", " ").split())


def _find_column_case_insensitive(columns: list[str], target: str) -> str | None:
    target_norm = _normalize_colname(target)
    for col in columns:
        if _normalize_colname(col) == target_norm:
            return col
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


def _read_timestamp_column(path: Path, column: str) -> pd.Series:
    # Some PPG exports have malformed rows where strict CSV column selection
    # can mis-parse timestamp values; treat field-0 as timestamp for robustness.
    if path.name.lower().endswith("_ppg.csv") and column.lower() == "timestamp":
        return reconstruct_constant_timestamp_packets(read_ppg_timestamp_series(path))

    # Ring logs are packetized .bin files and are not regular CSV.
    if path.suffix.lower() == ".bin" and column.lower() == "timestamp":
        return reconstruct_constant_timestamp_packets(read_ring_timestamp_series(path))

    # Polar ECG exports often use packet-level sys_time; reconstruct per-sample timestamps.
    if path.name.lower().endswith("_polar_ecg.csv") and column.lower() == "sys_time":
        return reconstruct_constant_timestamp_packets(read_polar_ecg_timestamp_series(path))

    header = pd.read_csv(path, nrows=0).columns.tolist()

    # eSense files can carry both a row-like timestamp and true wall-clock in `t`.
    # Prefer t -> Timestamp -> timestamp to match modality loader behavior.
    if path.name.lower().endswith("_esense.csv"):
        aliases = ("t", "Timestamp", "timestamp")
        for alias in aliases:
            if alias in header:
                df = pd.read_csv(path, usecols=[alias])
                series = pd.to_numeric(df[alias], errors="coerce")
                return reconstruct_constant_timestamp_packets(series)
            hit = next((c for c in header if c.lower() == alias.lower()), None)
            if hit is not None:
                df = pd.read_csv(path, usecols=[hit])
                series = pd.to_numeric(df[hit], errors="coerce")
                return reconstruct_constant_timestamp_packets(series)

    selected = next((c for c in header if c == column), None)
    if selected is None:
        selected = next((c for c in header if c.lower() == column.lower()), None)

    if selected is None:
        raise ValueError(
            f"Missing expected timestamp column '{column}' in {path}. "
            f"Available columns: {header}"
        )

    df = pd.read_csv(path, usecols=[selected])
    series = pd.to_numeric(df[selected], errors="coerce")
    return reconstruct_constant_timestamp_packets(series)



def load_engagement_log(path: Path) -> tuple[pd.DataFrame, dict[str, Any], dict[str, float]]:
    df = pd.read_csv(path)
    required_cols = {"video_uid", "video_timestamp_sec", "system_time_sec", "engagement_report"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Engagement log {path} is missing required columns: {sorted(missing)}")

    df = df.copy()
    df["video_uid"] = df["video_uid"].astype(str)
    df["video_timestamp_sec"] = pd.to_numeric(df["video_timestamp_sec"], errors="coerce")
    pulsed_col = _find_column_case_insensitive(list(df.columns), "is_paused")
    if pulsed_col is None:
        df["is_pulsed"] = False
    else:
        df["is_pulsed"] = df[pulsed_col].map(_parse_bool_flag).astype(bool)

    repaired_system = reconstruct_constant_timestamp_packets(df["system_time_sec"])
    normalized_system, detected_unit = detect_and_normalize_timestamps(repaired_system)
    df["system_time_sec_norm"] = normalized_system

    qc = compute_timestamp_qc(df["system_time_sec_norm"], detected_unit)
    qc["path"] = str(path)

    video_start_rows: list[dict[str, Any]] = []
    video_start_map: dict[str, float] = {}
    for video_uid, group in df.groupby("video_uid", sort=False):
        offsets = (group["system_time_sec_norm"] - group["video_timestamp_sec"]).dropna()
        if offsets.empty:
            video_start = np.nan
            mad = np.nan
        else:
            video_start = float(offsets.median())
            mad = float((offsets - video_start).abs().median())
            video_start_map[str(video_uid)] = video_start

        video_start_rows.append(
            {
                "video_uid": str(video_uid),
                "video_start_time_sec": video_start,
                "offset_mad_sec": mad,
                "num_rows": int(len(group)),
                "num_valid_offsets": int(offsets.notna().sum()),
            }
        )

    qc["video_start_rows"] = video_start_rows
    qc["precision_collapse"] = bool((qc["unique_ratio"] or 0.0) < 0.01)
    return df, qc, video_start_map



def discover_sessions(config: RunConfig) -> list[dict[str, Any]]:
    sessions: list[dict[str, Any]] = []

    participant_dirs = sorted(
        [p for p in config.data_root.glob("P*") if p.is_dir()],
        key=lambda p: p.name,
    )
    for participant_dir in participant_dirs:
        for log_path in sorted(participant_dir.glob("*_engagement_log.csv")):
            match = SESSION_PATTERN.match(log_path.name)
            if not match:
                LOGGER.warning("Skipping unmatched engagement log filename: %s", log_path)
                continue

            participant_id = match.group("participant")
            session_id = match.group("session")
            prefix = f"{participant_id}_{session_id}"
            session_key = f"P{participant_id}_S{session_id}"

            row: dict[str, Any] = {
                "session_key": session_key,
                "participant_id": int(participant_id),
                "session_id": int(session_id),
                "data_dir": str(participant_dir),
            }

            for stream_name, spec in STREAM_SPECS.items():
                if stream_name == "ring":
                    ring_candidates = sorted(participant_dir.glob("*.bin"))
                    stream_path = ring_candidates[0] if ring_candidates else None
                else:
                    candidate = participant_dir / f"{prefix}{spec.file_suffix}"
                    stream_path = candidate if candidate.exists() else None

                path_key = f"{stream_name}_path"
                has_key = f"has_{stream_name}"
                device_key = f"{stream_name}_device"
                row[path_key] = str(stream_path) if stream_path is not None else None
                row[has_key] = bool(stream_path is not None)
                row[device_key] = spec.device_name

            sessions.append(row)

    sessions.sort(key=lambda r: (r["participant_id"], r["session_id"]))
    return sessions



def _stream_qc_for_session(row: dict[str, Any], stream_name: str) -> dict[str, Any] | None:
    path_value = row.get(f"{stream_name}_path")
    if not path_value:
        return None

    path = Path(path_value)
    spec = STREAM_SPECS[stream_name]
    series = _read_timestamp_column(path, spec.timestamp_column)
    ts_sec, unit = detect_and_normalize_timestamps(series)
    qc = compute_timestamp_qc(ts_sec, unit)
    qc["stream_name"] = stream_name
    qc["path"] = str(path)
    return qc



def build_session_manifest(
    config: RunConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    session_rows = discover_sessions(config)
    manifest_rows: list[dict[str, Any]] = []
    alignment_rows: list[dict[str, Any]] = []
    normalized_event_rows: list[dict[str, Any]] = []
    video_offset_rows: list[dict[str, Any]] = []

    for row in session_rows:
        notes: list[str] = []
        qc_by_stream: dict[str, dict[str, Any]] = {}

        for stream_name in STREAM_SPECS:
            qc = _stream_qc_for_session(row, stream_name)
            if qc is not None:
                qc_by_stream[stream_name] = qc

        engagement_path = row.get("engagement_log_path")
        if not engagement_path:
            notes.append("missing_engagement_log")
            continue

        engagement_df, engagement_qc, _video_start_map = load_engagement_log(Path(engagement_path))
        qc_by_stream["engagement_log"] = {
            **engagement_qc,
            "stream_name": "engagement_log",
        }

        session_key = str(row["session_key"])
        participant_id = int(row["participant_id"])
        session_id = int(row["session_id"])

        for event_row in engagement_df.to_dict(orient="records"):
            normalized_event_rows.append(
                {
                    "session_key": session_key,
                    "participant_id": participant_id,
                    "session_id": session_id,
                    "video_uid": str(event_row.get("video_uid", "")),
                    "video_timestamp_sec": pd.to_numeric(
                        event_row.get("video_timestamp_sec"),
                        errors="coerce",
                    ),
                    "system_time_sec_norm": pd.to_numeric(
                        event_row.get("system_time_sec_norm"),
                        errors="coerce",
                    ),
                    "engagement_report": (
                        None
                        if pd.isna(event_row.get("engagement_report"))
                        else str(event_row.get("engagement_report"))
                    ),
                    "is_pulsed": bool(event_row.get("is_pulsed", False)),
                }
            )

        for video_qc in engagement_qc.get("video_start_rows", []):
            video_offset_rows.append(
                {
                    "session_key": session_key,
                    "participant_id": participant_id,
                    "session_id": session_id,
                    "video_uid": str(video_qc.get("video_uid", "")),
                    "video_start_time_sec": pd.to_numeric(
                        video_qc.get("video_start_time_sec"),
                        errors="coerce",
                    ),
                    "offset_mad_sec": pd.to_numeric(
                        video_qc.get("offset_mad_sec"),
                        errors="coerce",
                    ),
                    "num_valid_offsets": int(video_qc.get("num_valid_offsets", 0)),
                }
            )

        engagement_start = engagement_qc.get("start_sec")
        engagement_end = engagement_qc.get("end_sec")
        engagement_duration = (
            float(engagement_end - engagement_start)
            if pd.notna(engagement_start) and pd.notna(engagement_end)
            else math.nan
        )

        if engagement_qc.get("precision_collapse"):
            notes.append("engagement_system_time_precision_collapse")

        stream_start_values: list[float] = []
        stream_end_values: list[float] = []
        stream_unit_ok: list[bool] = []

        modality_streams = [
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
        ]

        for stream_name in modality_streams:
            qc = qc_by_stream.get(stream_name)
            if qc is None:
                alignment_rows.append(
                    {
                        "session_key": row["session_key"],
                        "participant_id": row["participant_id"],
                        "session_id": row["session_id"],
                        "modality": stream_name,
                        "has_data": False,
                        "stream_start_sec": np.nan,
                        "stream_end_sec": np.nan,
                        "engagement_start_sec": engagement_start,
                        "engagement_end_sec": engagement_end,
                        "overlap_sec": np.nan,
                        "engagement_coverage_ratio": np.nan,
                        "alignment_ok": False,
                        "notes": "missing_stream",
                    }
                )
                continue

            stream_start = qc["start_sec"]
            stream_end = qc["end_sec"]
            stream_start_values.append(stream_start)
            stream_end_values.append(stream_end)
            stream_unit_ok.append(bool(qc["timestamp_unit_ok"]))

            if pd.notna(stream_start) and pd.notna(stream_end) and pd.notna(engagement_start) and pd.notna(engagement_end):
                overlap_start = max(stream_start, engagement_start)
                overlap_end = min(stream_end, engagement_end)
                overlap = max(0.0, overlap_end - overlap_start)
                coverage_ratio = (
                    float(overlap / engagement_duration)
                    if engagement_duration and engagement_duration > 0
                    else np.nan
                )
            else:
                overlap = np.nan
                coverage_ratio = np.nan

            alignment_ok = bool(pd.notna(coverage_ratio) and coverage_ratio > 0.20)
            note = "ok" if alignment_ok else "low_overlap"
            alignment_rows.append(
                {
                    "session_key": row["session_key"],
                    "participant_id": row["participant_id"],
                    "session_id": row["session_id"],
                    "modality": stream_name,
                    "has_data": True,
                    "stream_start_sec": stream_start,
                    "stream_end_sec": stream_end,
                    "engagement_start_sec": engagement_start,
                    "engagement_end_sec": engagement_end,
                    "overlap_sec": overlap,
                    "engagement_coverage_ratio": coverage_ratio,
                    "alignment_ok": alignment_ok,
                    "notes": note,
                }
            )

        timestamp_unit_ok = bool(all(stream_unit_ok)) if stream_unit_ok else False
        if not timestamp_unit_ok:
            notes.append("timestamp_unit_check_failed")

        participant_is_p2 = int(row["participant_id"]) == 2
        alignment_ok = not engagement_qc.get("precision_collapse", False)
        if config.absolute_time_mode and config.exclude_p2_in_absolute_mode and participant_is_p2:
            alignment_ok = False
            notes.append("p2_alignment_disabled_for_absolute_time")

        manifest_row = dict(row)
        manifest_row.update(
            {
                "t_start_sec": float(min(stream_start_values)) if stream_start_values else np.nan,
                "t_end_sec": float(max(stream_end_values)) if stream_end_values else np.nan,
                "timestamp_unit_ok": timestamp_unit_ok,
                "alignment_ok": alignment_ok,
                "notes": "|".join(dict.fromkeys(notes)),
                "engagement_unique_ratio": engagement_qc.get("unique_ratio", np.nan),
                "engagement_time_unit": engagement_qc.get("detected_unit", "unknown"),
            }
        )

        for stream_name, qc in qc_by_stream.items():
            manifest_row[f"{stream_name}_t_start_sec"] = qc.get("start_sec", np.nan)
            manifest_row[f"{stream_name}_t_end_sec"] = qc.get("end_sec", np.nan)
            manifest_row[f"{stream_name}_detected_unit"] = qc.get("detected_unit", "unknown")

        manifest_rows.append(manifest_row)

        for video_qc in engagement_qc.get("video_start_rows", []):
            alignment_rows.append(
                {
                    "session_key": row["session_key"],
                    "participant_id": row["participant_id"],
                    "session_id": row["session_id"],
                    "modality": "engagement_video_map",
                    "has_data": True,
                    "stream_start_sec": video_qc.get("video_start_time_sec", np.nan),
                    "stream_end_sec": video_qc.get("video_start_time_sec", np.nan),
                    "engagement_start_sec": engagement_start,
                    "engagement_end_sec": engagement_end,
                    "overlap_sec": np.nan,
                    "engagement_coverage_ratio": np.nan,
                    "alignment_ok": not engagement_qc.get("precision_collapse", False),
                    "notes": (
                        f"video_uid={video_qc.get('video_uid')}|"
                        f"offset_mad_sec={video_qc.get('offset_mad_sec')}|"
                        f"num_valid_offsets={video_qc.get('num_valid_offsets')}"
                    ),
                }
            )

    manifest_df = pd.DataFrame(manifest_rows)
    alignment_df = pd.DataFrame(alignment_rows)
    normalized_events_df = pd.DataFrame(normalized_event_rows)
    video_offsets_df = pd.DataFrame(video_offset_rows)
    return manifest_df, alignment_df, normalized_events_df, video_offsets_df



def run_phase_b(config: RunConfig) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Path]]:
    config.ensure_directories()

    (
        manifest_df,
        alignment_df,
        normalized_events_df,
        video_offsets_df,
    ) = build_session_manifest(config)

    manifest_path = _write_table(manifest_df, config.manifests_dir / "session_manifest.parquet")
    alignment_path = config.manifests_dir / "alignment_qc.csv"
    alignment_df.to_csv(alignment_path, index=False)
    normalized_events_path = _write_table(
        normalized_events_df,
        config.manifests_dir / "engagement_events_normalized.parquet",
    )
    video_offsets_path = _write_table(
        video_offsets_df,
        config.manifests_dir / "video_start_offsets.parquet",
    )

    run_manifest_path = _write_table(manifest_df, config.run_dir / "manifests" / "session_manifest.parquet")
    run_alignment_path = config.run_dir / "manifests" / "alignment_qc.csv"
    run_alignment_path.parent.mkdir(parents=True, exist_ok=True)
    alignment_df.to_csv(run_alignment_path, index=False)
    run_normalized_events_path = _write_table(
        normalized_events_df,
        config.run_dir / "manifests" / "engagement_events_normalized.parquet",
    )
    run_video_offsets_path = _write_table(
        video_offsets_df,
        config.run_dir / "manifests" / "video_start_offsets.parquet",
    )

    outputs = {
        "manifest": manifest_path,
        "alignment_qc": alignment_path,
        "engagement_events_normalized": normalized_events_path,
        "video_start_offsets": video_offsets_path,
        "run_manifest": run_manifest_path,
        "run_alignment_qc": run_alignment_path,
        "run_engagement_events_normalized": run_normalized_events_path,
        "run_video_start_offsets": run_video_offsets_path,
    }
    return manifest_df, alignment_df, outputs


