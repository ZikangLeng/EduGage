"""Encoder adapters and embedding cache utilities (Phase D)."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

from .config import ExternalModelSpec, RunConfig
from .io_utils import read_table_with_fallback, write_table_with_fallback
from .loaders import (
    load_session_modality,
    numeric_feature_columns,
    resample_numeric_segment,
    slice_time_window,
)

LOGGER = logging.getLogger(__name__)

EMBEDDING_COLUMNS = [
    "window_id",
    "session_key",
    "participant_id",
    "modality",
    "source_id",
    "encoder_name",
    "encoder_version",
    "encoder_commit",
    "cache_fingerprint",
    "preprocess_hash",
    "config_hash",
    "embedding_dim",
    "embedding",
]


class EmbeddingAdapter(Protocol):
    encoder_name: str
    encoder_version: str
    encoder_commit: str

    def encode(self, segment_df: pd.DataFrame) -> np.ndarray:
        ...


@dataclass(frozen=True)
class StatsEmbeddingAdapter:
    encoder_name: str
    encoder_version: str
    encoder_commit: str
    target_points: int = 128

    def encode(self, segment_df: pd.DataFrame) -> np.ndarray:
        cols = numeric_feature_columns(segment_df)
        matrix = resample_numeric_segment(
            segment_df=segment_df,
            feature_columns=cols,
            target_points=self.target_points,
        )
        return _stats_embedding(matrix)


@dataclass(frozen=True)
class EyeEmbeddingAdapter:
    encoder_name: str
    encoder_version: str
    encoder_commit: str

    def encode(self, segment_df: pd.DataFrame) -> np.ndarray:
        numeric = segment_df.copy()
        for col in [
            "gaze_conf_int",
            "gaze_por_x",
            "gaze_por_y",
            "head_conf_int",
            "head_pos_x_m",
            "head_pos_y_m",
            "head_pos_z_m",
        ]:
            if col not in numeric.columns:
                numeric[col] = 0.0
            numeric[col] = pd.to_numeric(numeric[col], errors="coerce").fillna(0.0)

        t_sec = pd.to_numeric(numeric["t_sec"], errors="coerce").fillna(0.0)
        duration = float(max(0.0, t_sec.max() - t_sec.min())) if len(t_sec) else 0.0

        gaze_conf = numeric["gaze_conf_int"].to_numpy(dtype=float)
        head_conf = numeric["head_conf_int"].to_numpy(dtype=float)
        gaze_x = numeric["gaze_por_x"].to_numpy(dtype=float)
        gaze_y = numeric["gaze_por_y"].to_numpy(dtype=float)
        head_xyz = numeric[["head_pos_x_m", "head_pos_y_m", "head_pos_z_m"]].to_numpy(
            dtype=float
        )

        gaze_speed = np.linalg.norm(np.diff(np.stack([gaze_x, gaze_y], axis=1), axis=0), axis=1)
        head_speed = np.linalg.norm(np.diff(head_xyz, axis=0), axis=1)
        head_norm = np.linalg.norm(head_xyz, axis=1)

        features = np.array(
            [
                float(len(numeric)),
                duration,
                _safe_mean(gaze_conf),
                _safe_std(gaze_conf),
                _safe_mean(head_conf),
                _safe_std(head_conf),
                _safe_mean(gaze_x),
                _safe_std(gaze_x),
                _safe_mean(gaze_y),
                _safe_std(gaze_y),
                _safe_percentile(gaze_x, 10.0),
                _safe_percentile(gaze_x, 90.0),
                _safe_percentile(gaze_y, 10.0),
                _safe_percentile(gaze_y, 90.0),
                _safe_mean(gaze_speed),
                _safe_std(gaze_speed),
                _safe_mean(head_norm),
                _safe_std(head_norm),
                _safe_mean(head_speed),
                _safe_std(head_speed),
            ],
            dtype=float,
        )
        return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


@dataclass(frozen=True)
class EmbeddingCacheState:
    fingerprint: str
    preserved_df: pd.DataFrame
    cached_target_df: pd.DataFrame
    cached_keys: set[tuple[str, str, str, str]]


@dataclass(frozen=True)
class SegmentJob:
    window_id: str
    session_key: str
    participant_id: int
    modality: str
    source_id: str
    t_start_sec: float
    t_end_sec: float


MODEL_KEY_BY_MODALITY = {
    "eeg": "eeg",
    "ecg": "ecg",
    "ppg": "ppg",
    "ring": "eda_imu",
    "eda": "eda_imu",
    "acc": "eda_imu",
    "gyro": "eda_imu",
    "hr": "eda_imu",
    "esense": "eda_imu",
    "eye": "eye",
}


def _safe_mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size else 0.0


def _safe_std(values: np.ndarray) -> float:
    return float(np.std(values)) if values.size else 0.0


def _safe_percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q)) if values.size else 0.0


def _stats_embedding(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return np.empty((0,), dtype=float)

    diff = np.diff(matrix, axis=0)
    flat = matrix.reshape(-1)
    channel_mean = matrix.mean(axis=0)
    channel_std = matrix.std(axis=0)

    head_mean = np.zeros(3, dtype=float)
    head_std = np.zeros(3, dtype=float)
    limit = min(3, matrix.shape[1])
    head_mean[:limit] = channel_mean[:limit]
    head_std[:limit] = channel_std[:limit]

    features = np.array(
        [
            float(matrix.shape[0]),
            float(matrix.shape[1]),
            float(flat.mean()),
            float(flat.std()),
            float(flat.min()),
            float(flat.max()),
            float(np.median(flat)),
            float(np.percentile(flat, 10.0)),
            float(np.percentile(flat, 90.0)),
            float(np.mean(np.abs(diff))) if diff.size else 0.0,
            float(np.std(diff)) if diff.size else 0.0,
            float(np.mean(matrix**2)),
            float(np.mean(np.abs(matrix))),
            float(np.var(channel_mean)),
            float(np.var(channel_std)),
            float(head_mean[0]),
            float(head_mean[1]),
            float(head_mean[2]),
            float(head_std[0]),
            float(head_std[1]),
            float(head_std[2]),
        ],
        dtype=float,
    )
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


def _build_adapter(modality: str, model_spec: ExternalModelSpec | None) -> EmbeddingAdapter:
    if model_spec is None:
        encoder_name = f"{modality}_engineered"
        encoder_commit = "local"
    else:
        encoder_name = f"{model_spec.name}_engineered_fallback"
        encoder_commit = model_spec.commit

    if modality == "eye":
        return EyeEmbeddingAdapter(
            encoder_name=encoder_name,
            encoder_version="eye_feats_v1",
            encoder_commit=encoder_commit,
        )

    return StatsEmbeddingAdapter(
        encoder_name=encoder_name,
        encoder_version="stats_v1",
        encoder_commit=encoder_commit,
        target_points=128,
    )



def _availability_for_window(window_row: dict[str, Any]) -> dict[str, bool]:
    return {
        str(col)[4:]: bool(window_row.get(col, False))
        for col in window_row.keys()
        if str(col).startswith("has_")
    }


def _resolve_absolute_bounds(window_row: dict[str, Any]) -> tuple[float, float]:
    start = pd.to_numeric(window_row.get("t_start_sec"), errors="coerce")
    end = pd.to_numeric(window_row.get("t_end_sec"), errors="coerce")
    if pd.notna(start) and pd.notna(end):
        return float(start), float(end)
    return np.nan, np.nan


def _cache_fingerprint(config: RunConfig) -> str:
    model_payload = {
        key: {
            "name": spec.name,
            "repo_url": spec.repo_url,
            "commit": spec.commit,
            "checkpoint_path": spec.checkpoint_path,
        }
        for key, spec in sorted(config.model_registry.items())
    }
    payload = {
        "seed": config.seed,
        "window_size_sec": config.window_size_sec,
        "stride_sec": config.stride_sec,
        "window_generation_mode": config.window_generation_mode,
        "interval_boundary_mode": config.interval_boundary_mode,
        "exclude_video_uid_patterns": list(config.exclude_video_uid_patterns),
        "short_interval_policy": config.short_interval_policy,
        "absolute_time_mode": config.absolute_time_mode,
        "exclude_p2_in_absolute_mode": config.exclude_p2_in_absolute_mode,
        "enabled_modalities": dict(sorted(config.enabled_modalities.items())),
        "model_registry": model_payload,
        "stats_adapter": "stats_v1",
        "eye_adapter": "eye_feats_v1",
        "resample_points": 128,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:12]


def _cache_key(row: dict[str, Any], fallback_fingerprint: str = "") -> tuple[str, str, str, str]:
    fingerprint = str(row.get("cache_fingerprint", "")).strip() or fallback_fingerprint
    return (
        str(row.get("window_id", "")),
        str(row.get("modality", "")),
        str(row.get("source_id", "")),
        fingerprint,
    )


def prepare_embedding_cache_state(
    existing_df: pd.DataFrame,
    fingerprint: str,
    *,
    force_recompute: bool,
) -> EmbeddingCacheState:
    if existing_df.empty:
        empty = pd.DataFrame(columns=EMBEDDING_COLUMNS)
        return EmbeddingCacheState(
            fingerprint=fingerprint,
            preserved_df=empty,
            cached_target_df=empty,
            cached_keys=set(),
        )

    for col in EMBEDDING_COLUMNS:
        if col not in existing_df.columns:
            existing_df[col] = ""

    cache_fingerprint_series = existing_df["cache_fingerprint"].astype(str).str.strip()
    target_mask = cache_fingerprint_series == fingerprint

    preserved_df = existing_df.loc[~target_mask, EMBEDDING_COLUMNS].copy()
    cached_target_df = existing_df.loc[target_mask, EMBEDDING_COLUMNS].copy()

    if force_recompute:
        cached_target_df = pd.DataFrame(columns=EMBEDDING_COLUMNS)

    cached_keys = {
        _cache_key(row, fallback_fingerprint=fingerprint)
        for row in cached_target_df.to_dict(orient="records")
    }

    return EmbeddingCacheState(
        fingerprint=fingerprint,
        preserved_df=preserved_df,
        cached_target_df=cached_target_df,
        cached_keys=cached_keys,
    )


def merge_cached_and_new_embeddings(
    *,
    cache_state: EmbeddingCacheState,
    new_rows: list[dict[str, Any]],
) -> pd.DataFrame:
    new_df = pd.DataFrame(new_rows, columns=EMBEDDING_COLUMNS)
    target_df = pd.concat([cache_state.cached_target_df, new_df], ignore_index=True)
    if not target_df.empty:
        target_df = target_df.drop_duplicates(
            subset=["window_id", "modality", "source_id", "cache_fingerprint"],
            keep="last",
        )

    final_df = pd.concat([cache_state.preserved_df, target_df], ignore_index=True)
    if final_df.empty:
        return pd.DataFrame(columns=EMBEDDING_COLUMNS)

    return final_df[EMBEDDING_COLUMNS].copy()


def iterate_window_modality_segments(
    *,
    window_index: pd.DataFrame,
    session_rows: dict[str, dict[str, Any]],
    enabled_modalities: list[str],
    cache_state: EmbeddingCacheState,
) -> tuple[list[SegmentJob], int]:
    segment_jobs: list[SegmentJob] = []
    cache_hits = 0

    for window_row in window_index.to_dict(orient="records"):
        session_key = str(window_row.get("session_key", ""))
        session_row = session_rows.get(session_key)
        if session_row is None:
            continue

        t_start_sec, t_end_sec = _resolve_absolute_bounds(window_row)
        if not np.isfinite(t_start_sec) or not np.isfinite(t_end_sec) or t_end_sec <= t_start_sec:
            continue

        availability_mask = _availability_for_window(window_row)

        for modality in enabled_modalities:
            if not availability_mask.get(modality, False):
                continue

            source_id = str(session_row.get(f"{modality}_path") or "")
            if not source_id:
                continue

            candidate_key = (
                str(window_row.get("window_id", "")),
                modality,
                source_id,
                cache_state.fingerprint,
            )
            if candidate_key in cache_state.cached_keys:
                cache_hits += 1
                continue

            participant_id = pd.to_numeric(window_row.get("participant_id"), errors="coerce")
            if pd.isna(participant_id):
                continue

            segment_jobs.append(
                SegmentJob(
                    window_id=str(window_row.get("window_id", "")),
                    session_key=session_key,
                    participant_id=int(participant_id),
                    modality=modality,
                    source_id=source_id,
                    t_start_sec=float(t_start_sec),
                    t_end_sec=float(t_end_sec),
                )
            )

    return segment_jobs, cache_hits


def encode_segments(
    *,
    segment_jobs: list[SegmentJob],
    adapters: dict[str, EmbeddingAdapter],
    session_rows: dict[str, dict[str, Any]],
    interval_boundary_mode: str,
    fingerprint: str,
) -> list[dict[str, Any]]:
    loaded_streams: dict[tuple[str, str], pd.DataFrame] = {}
    new_rows: list[dict[str, Any]] = []

    for job in segment_jobs:
        session_row = session_rows.get(job.session_key)
        if session_row is None:
            continue

        stream_key = (job.session_key, job.modality)
        if stream_key not in loaded_streams:
            try:
                loaded_streams[stream_key] = load_session_modality(session_row, job.modality)
            except (FileNotFoundError, ValueError, pd.errors.EmptyDataError) as exc:
                LOGGER.warning(
                    "Skipping %s/%s due to loader error: %s",
                    job.session_key,
                    job.modality,
                    exc,
                )
                continue

        stream_df = loaded_streams[stream_key]
        segment_df = slice_time_window(
            stream_df,
            t_start_sec=job.t_start_sec,
            t_end_sec=job.t_end_sec,
            boundary_mode=interval_boundary_mode,
        )
        if segment_df.empty:
            continue

        adapter = adapters[job.modality]
        embedding = adapter.encode(segment_df)
        if embedding.size == 0:
            continue

        new_rows.append(
            {
                "window_id": job.window_id,
                "session_key": job.session_key,
                "participant_id": int(job.participant_id),
                "modality": job.modality,
                "source_id": job.source_id,
                "encoder_name": adapter.encoder_name,
                "encoder_version": adapter.encoder_version,
                "encoder_commit": adapter.encoder_commit,
                "cache_fingerprint": fingerprint,
                "preprocess_hash": fingerprint,
                "config_hash": fingerprint,
                "embedding_dim": int(len(embedding)),
                "embedding": json.dumps([float(x) for x in embedding.tolist()]),
            }
        )

    return new_rows


def build_embedding_table(
    config: RunConfig,
    session_manifest: pd.DataFrame,
    window_index: pd.DataFrame,
    force_recompute: bool = False,
) -> tuple[pd.DataFrame, dict[str, Path], dict[str, int]]:
    config.ensure_directories()
    parquet_path = config.embeddings_dir / "embedding_table.parquet"
    run_parquet_path = config.run_dir / "embeddings" / "embedding_table.parquet"

    existing_df = read_table_with_fallback(parquet_path, default_columns=EMBEDDING_COLUMNS)
    fingerprint = _cache_fingerprint(config)
    cache_state = prepare_embedding_cache_state(
        existing_df,
        fingerprint,
        force_recompute=force_recompute,
    )

    session_rows = {
        str(row["session_key"]): row for row in session_manifest.to_dict(orient="records")
    }

    enabled_modalities = [
        modality
        for modality, enabled in config.enabled_modalities.items()
        if enabled and modality != "markers"
    ]

    adapters: dict[str, EmbeddingAdapter] = {}
    for modality in enabled_modalities:
        model_key = MODEL_KEY_BY_MODALITY.get(modality)
        model_spec = config.model_registry.get(model_key) if model_key else None
        adapters[modality] = _build_adapter(modality, model_spec)

    segment_jobs, cache_hits = iterate_window_modality_segments(
        window_index=window_index,
        session_rows=session_rows,
        enabled_modalities=enabled_modalities,
        cache_state=cache_state,
    )

    new_rows = encode_segments(
        segment_jobs=segment_jobs,
        adapters=adapters,
        session_rows=session_rows,
        interval_boundary_mode=config.interval_boundary_mode,
        fingerprint=fingerprint,
    )

    final_df = merge_cached_and_new_embeddings(cache_state=cache_state, new_rows=new_rows)

    output_path = write_table_with_fallback(final_df, parquet_path, logger=LOGGER)
    run_output_path = write_table_with_fallback(final_df, run_parquet_path, logger=LOGGER)

    outputs = {
        "embedding_table": output_path,
        "run_embedding_table": run_output_path,
    }
    stats = {
        "cache_hits": int(cache_hits),
        "new_rows": int(len(new_rows)),
        "total_rows": int(len(final_df)),
    }
    return final_df, outputs, stats


def run_phase_d(
    config: RunConfig,
    session_manifest: pd.DataFrame | None = None,
    window_index: pd.DataFrame | None = None,
    force_recompute: bool = False,
) -> tuple[pd.DataFrame, dict[str, Path], dict[str, int]]:
    if session_manifest is None:
        session_manifest = read_table_with_fallback(
            config.manifests_dir / "session_manifest.parquet"
        )
    if window_index is None:
        window_index = read_table_with_fallback(config.windows_dir / "window_index.parquet")

    if session_manifest.empty:
        raise FileNotFoundError(
            "Session manifest is empty or missing. Run phase B before phase D."
        )
    if window_index.empty:
        raise FileNotFoundError("Window index is empty or missing. Run phase C before phase D.")

    return build_embedding_table(
        config=config,
        session_manifest=session_manifest,
        window_index=window_index,
        force_recompute=force_recompute,
    )






