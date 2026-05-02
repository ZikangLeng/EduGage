from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
import sys
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from engagement.baselines.consensus_zero_shot import (  # noqa: E402
    _compute_metrics,
    _export,
    _resolve_windows,
    run_phase_z1_baseline_consensus,
)
from engagement.config import (  # noqa: E402
    build_default_config,
    validate_baseline_config,
    validate_core_config,
)
from engagement.env_utils import load_env_files  # noqa: E402
from engagement.io_utils import read_table_with_fallback  # noqa: E402


LOGGER = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Resume a ConSensus full-study baseline run by keeping previously completed "
            "window predictions for selected backends and rerunning only the remaining windows."
        )
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--source-run-id",
        help="Existing run id under artifacts/runs/<run-id>/baselines/consensus_zs.",
    )
    source_group.add_argument(
        "--source-predictions",
        help="Optional explicit path to an existing predictions parquet/csv file.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="New run id for the rerun + merged outputs. Default: timestamped resume run id.",
    )
    parser.add_argument(
        "--keep-backends",
        default="llm_consensus_agent",
        help=(
            "Comma-separated backend values to preserve from the source predictions. "
            "Default: llm_consensus_agent"
        ),
    )
    parser.add_argument(
        "--merged-phase-name",
        default="consensus_zs_merged",
        help="Baselines subdirectory name for merged outputs. Default: consensus_zs_merged",
    )
    parser.add_argument(
        "--max-rerun-windows",
        type=int,
        default=0,
        help="Optional cap for testing. <=0 means rerun all remaining windows.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only compute keep/rerun counts and planned paths; do not rerun any windows.",
    )
    parser.add_argument(
        "--provider",
        default=None,
        help="Optional provider override for the rerun subset (e.g. openai, heuristic).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional model override for the rerun subset (e.g. openai:gpt-5-mini or openai:local-qwen).",
    )
    parser.add_argument(
        "--openai-base-url",
        default=None,
        help="Optional OpenAI-compatible base URL.",
    )
    parser.add_argument(
        "--openai-api-key",
        default=None,
        help="Optional API key for OpenAI-compatible endpoints.",
    )
    parser.add_argument(
        "--litellm-base-url",
        default=None,
        help="Optional LiteLLM base URL (used for local-* OpenAI-compatible models).",
    )
    parser.add_argument(
        "--litellm-api-key",
        default=None,
        help="Optional LiteLLM API key for local models.",
    )
    parser.add_argument(
        "--preprocessed-root",
        default=None,
        help="Optional path to preprocessed_data root. Default: <repo>/preprocessed_data",
    )
    parser.add_argument(
        "--window-generation-mode",
        choices=["interval_sliding", "event_trailing"],
        default=None,
        help="Optional override for config.window_generation_mode.",
    )
    parser.add_argument(
        "--window-size-sec",
        type=float,
        default=None,
        help="Optional override for config.window_size_sec.",
    )
    parser.add_argument(
        "--stride-sec",
        type=float,
        default=None,
        help="Optional override for config.stride_sec.",
    )
    parser.add_argument(
        "--examples-per-class",
        type=int,
        default=None,
        help="Optional override for in-context examples per class. Use 0 for strict zero-shot.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def _default_run_id(source_name: str) -> str:
    stamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%S%fZ")
    safe_source = str(source_name).strip().replace(" ", "_")
    return f"{stamp}__resume_{safe_source}"


def _parse_backend_list(raw: str) -> set[str]:
    values = {part.strip() for part in str(raw).split(",") if part.strip()}
    if not values:
        raise ValueError("keep_backends must include at least one backend value.")
    return values


def _resolve_source_paths(args: argparse.Namespace) -> tuple[Path, Path, Path | None, Path | None, str]:
    if args.source_run_id:
        phase_dir = (
            REPO_ROOT
            / "artifacts"
            / "runs"
            / str(args.source_run_id)
            / "baselines"
            / "consensus_zs"
        ).resolve()
        return (
            phase_dir,
            phase_dir / "predictions.parquet",
            phase_dir / "loso_splits.parquet",
            phase_dir / "z1_summary.json",
            str(args.source_run_id),
        )

    predictions_path = Path(args.source_predictions).resolve()
    phase_dir = predictions_path.parent
    source_name = predictions_path.stem
    return (
        phase_dir,
        predictions_path,
        phase_dir / "loso_splits.parquet",
        phase_dir / "z1_summary.json",
        source_name,
    )


def _load_previous_predictions(predictions_path: Path) -> pd.DataFrame:
    df = read_table_with_fallback(predictions_path)
    if df.empty:
        raise FileNotFoundError(f"Could not load prior predictions from: {predictions_path}")
    if "window_id" not in df.columns:
        raise ValueError(f"Prior predictions missing required column 'window_id': {predictions_path}")
    df = df.copy()
    df["window_id"] = df["window_id"].astype(str)
    if "backend" not in df.columns:
        df["backend"] = ""
    return df.drop_duplicates(subset=["window_id"], keep="last").reset_index(drop=True)


def _load_optional_records(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    df = read_table_with_fallback(path)
    if df.empty:
        return []
    return df.to_dict(orient="records")


def _compute_fold_metric_rows(predictions_df: pd.DataFrame) -> list[dict[str, Any]]:
    if predictions_df.empty:
        return []
    required = {"fold_id", "test_participant_id"}
    if not required.issubset(predictions_df.columns):
        return []

    rows: list[dict[str, Any]] = []
    grouped = predictions_df.groupby(["fold_id", "test_participant_id"], sort=True, dropna=False)
    for (fold_id, test_pid), group in grouped:
        metrics, _ = _compute_metrics(group)
        rows.append(
            {
                "experiment": "consensus_zero_shot",
                "backend": "resume_merge",
                "fold_id": int(fold_id),
                "test_participant_id": int(test_pid),
                **metrics,
            }
        )
    return rows


def _minimal_split_rows(predictions_df: pd.DataFrame) -> list[dict[str, Any]]:
    if predictions_df.empty:
        return []
    required = {"fold_id", "test_participant_id"}
    if not required.issubset(predictions_df.columns):
        return []
    rows: list[dict[str, Any]] = []
    grouped = predictions_df.groupby(["fold_id", "test_participant_id"], sort=True, dropna=False)
    for (fold_id, test_pid), group in grouped:
        rows.append(
            {
                "experiment": "consensus_zero_shot",
                "fold_id": int(fold_id),
                "test_participant_id": int(test_pid),
                "num_test_windows": int(len(group)),
                "resume_generated": True,
            }
        )
    return rows


def _build_summary(
    config,
    *,
    source_label: str,
    source_predictions_path: Path,
    keep_backends: set[str],
    source_rows: int,
    kept_rows: int,
    rerun_rows_requested: int,
    merged_rows: int,
    expected_rows: int,
    missing_rows: int,
    rerun_outputs: dict[str, Path] | None,
) -> dict[str, Any]:
    baseline_cfg = config.baseline.consensus_zero_shot
    effective_zero_shot = bool(int(baseline_cfg.examples_per_class) <= 0)
    return {
        "stage": "baseline_consensus_zs_resume_merge",
        "status": (
            "phase_z1_resume_merge_completed"
            if missing_rows == 0
            else "phase_z1_resume_merge_partial"
        ),
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
        "resume_source_label": source_label,
        "resume_source_predictions": str(source_predictions_path),
        "resume_keep_backends": sorted(keep_backends),
        "resume_source_rows": int(source_rows),
        "resume_kept_rows": int(kept_rows),
        "resume_rerun_rows_requested": int(rerun_rows_requested),
        "resume_merged_rows": int(merged_rows),
        "resume_expected_rows": int(expected_rows),
        "resume_missing_rows": int(missing_rows),
        "resume_partial_run_outputs": (
            {k: str(v) for k, v in sorted(rerun_outputs.items())}
            if rerun_outputs
            else {}
        ),
    }


def _print_json(title: str, payload: dict[str, Any]) -> None:
    print(f"\n{title}")
    print(json.dumps(payload, indent=2, sort_keys=True))


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    loaded_env_files = load_env_files(REPO_ROOT)
    if loaded_env_files:
        LOGGER.info("Loaded environment from: %s", ", ".join(str(p) for p in loaded_env_files))

    source_phase_dir, source_predictions_path, source_splits_path, source_summary_path, source_label = _resolve_source_paths(args)
    previous_predictions = _load_previous_predictions(source_predictions_path)
    keep_backends = _parse_backend_list(args.keep_backends)

    config = build_default_config(repo_root=REPO_ROOT)
    config.run_id = args.run_id or _default_run_id(source_label)
    if args.provider:
        config.baseline.consensus_zero_shot.provider = str(args.provider)
    if args.model:
        config.baseline.consensus_zero_shot.model = str(args.model)
    if args.openai_base_url:
        os.environ["OPENAI_BASE_URL"] = str(args.openai_base_url)
    if args.openai_api_key:
        os.environ["OPENAI_API_KEY"] = str(args.openai_api_key)
    if args.litellm_base_url:
        os.environ["LITELLM_BASE_URL"] = str(args.litellm_base_url)
    if args.litellm_api_key:
        os.environ["LITELLM_API_KEY"] = str(args.litellm_api_key)
    if args.window_generation_mode is not None:
        config.window_generation_mode = str(args.window_generation_mode)
    if args.window_size_sec is not None:
        config.window_size_sec = float(args.window_size_sec)
    if args.stride_sec is not None:
        config.stride_sec = float(args.stride_sec)
    if args.examples_per_class is not None:
        config.baseline.consensus_zero_shot.examples_per_class = int(args.examples_per_class)
        config.baseline.consensus_zero_shot.zero_shot = bool(int(args.examples_per_class) <= 0)

    validate_core_config(config)
    validate_baseline_config(config)
    config.ensure_directories()

    preprocessed_root = (
        Path(args.preprocessed_root).resolve()
        if args.preprocessed_root
        else (config.repo_root / "preprocessed_data").resolve()
    )

    all_windows = _resolve_windows(config, window_index=None)
    if all_windows.empty:
        raise RuntimeError("No windows found. Run Phase C first or provide a valid windows artifact.")
    all_windows = all_windows.copy()
    all_windows["window_id"] = all_windows["window_id"].astype(str)

    current_window_ids = set(all_windows["window_id"].tolist())
    previous_predictions = previous_predictions[previous_predictions["window_id"].isin(current_window_ids)].copy()

    keep_mask = previous_predictions["backend"].astype(str).isin(keep_backends)
    kept_predictions = previous_predictions[keep_mask].copy()
    kept_window_ids = set(kept_predictions["window_id"].tolist())

    rerun_windows = all_windows[~all_windows["window_id"].isin(sorted(kept_window_ids))].copy()
    rerun_windows = rerun_windows.sort_values(["participant_id", "session_key", "t_start_sec", "window_id"], kind="mergesort")
    if args.max_rerun_windows > 0:
        rerun_windows = rerun_windows.head(int(args.max_rerun_windows)).copy()

    merged_phase_dir = config.run_dir / "baselines" / str(args.merged_phase_name)
    merged_phase_dir.mkdir(parents=True, exist_ok=True)

    dry_run_payload = {
        "source_phase_dir": str(source_phase_dir),
        "source_predictions": str(source_predictions_path),
        "source_summary_exists": bool(source_summary_path and source_summary_path.exists()),
        "source_rows": int(len(previous_predictions)),
        "keep_backends": sorted(keep_backends),
        "kept_rows": int(len(kept_predictions)),
        "rerun_rows_requested": int(len(rerun_windows)),
        "expected_total_windows": int(len(all_windows)),
        "new_run_id": config.run_id,
        "new_run_dir": str(config.run_dir),
        "merged_phase_dir": str(merged_phase_dir),
        "provider": config.baseline.consensus_zero_shot.provider,
        "model": config.baseline.consensus_zero_shot.model,
        "examples_per_class": int(config.baseline.consensus_zero_shot.examples_per_class),
    }

    if args.dry_run:
        _print_json("Resume Dry Run", dry_run_payload)
        return 0

    rerun_outputs: dict[str, Path] | None = None
    if not rerun_windows.empty:
        LOGGER.info(
            "Rerunning %d windows from %s into new run_id=%s",
            len(rerun_windows),
            source_label,
            config.run_id,
        )
        rerun_outputs, rerun_stats = run_phase_z1_baseline_consensus(
            config,
            window_index=rerun_windows,
            example_window_index=all_windows,
            preprocessed_root=preprocessed_root,
        )
        LOGGER.info("Partial rerun stats: %s", json.dumps(rerun_stats, sort_keys=True))
    else:
        LOGGER.info("No windows need rerun; exporting merged copy from kept predictions only.")

    rerun_predictions = pd.DataFrame()
    if rerun_outputs and "predictions" in rerun_outputs:
        rerun_predictions = read_table_with_fallback(rerun_outputs["predictions"])
        if not rerun_predictions.empty:
            rerun_predictions = rerun_predictions.copy()
            rerun_predictions["window_id"] = rerun_predictions["window_id"].astype(str)

    if rerun_windows.empty:
        merged_predictions = kept_predictions.copy()
    else:
        if rerun_predictions.empty:
            raise RuntimeError(
                "Rerun finished without producing predictions.parquet; cannot build merged results."
            )
        merged_predictions = pd.concat(
            [
                kept_predictions[~kept_predictions["window_id"].isin(set(rerun_predictions["window_id"].tolist()))],
                rerun_predictions,
            ],
            ignore_index=True,
        )

    merged_predictions = merged_predictions.drop_duplicates(subset=["window_id"], keep="last").copy()
    sort_cols = [col for col in ("fold_id", "test_participant_id", "participant_id", "session_key", "t_start_sec", "window_id") if col in merged_predictions.columns]
    if sort_cols:
        merged_predictions = merged_predictions.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    missing_window_ids = sorted(current_window_ids - set(merged_predictions["window_id"].tolist()))
    if missing_window_ids:
        LOGGER.warning(
            "Merged predictions are still missing %d windows. First missing ids: %s",
            len(missing_window_ids),
            ", ".join(missing_window_ids[:10]),
        )

    split_rows = _load_optional_records(source_splits_path)
    if not split_rows:
        split_rows = _minimal_split_rows(merged_predictions)
    fold_metric_rows = _compute_fold_metric_rows(merged_predictions)

    summary = _build_summary(
        config,
        source_label=source_label,
        source_predictions_path=source_predictions_path,
        keep_backends=keep_backends,
        source_rows=len(previous_predictions),
        kept_rows=len(kept_predictions),
        rerun_rows_requested=len(rerun_windows),
        merged_rows=len(merged_predictions),
        expected_rows=len(all_windows),
        missing_rows=len(missing_window_ids),
        rerun_outputs=rerun_outputs,
    )

    merged_outputs, merged_stats = _export(
        merged_phase_dir,
        summary,
        split_rows,
        merged_predictions.to_dict(orient="records"),
        fold_metric_rows,
    )

    payload = {
        "run_id": config.run_id,
        "source_label": source_label,
        "kept_rows": int(len(kept_predictions)),
        "rerun_rows_requested": int(len(rerun_windows)),
        "merged_rows": int(len(merged_predictions)),
        "missing_rows": int(len(missing_window_ids)),
        "partial_rerun_outputs": (
            {k: str(v) for k, v in sorted((rerun_outputs or {}).items())}
        ),
        "merged_outputs": {k: str(v) for k, v in sorted(merged_outputs.items())},
        "merged_stats": merged_stats,
    }
    _print_json("Resume Complete", payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
