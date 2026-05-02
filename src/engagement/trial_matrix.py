from __future__ import annotations

import copy
import json
import logging
import multiprocessing as mp
from pathlib import Path
import re
import traceback
from typing import Any

import numpy as np
import pandas as pd

from .config import RunConfig, persist_run_metadata, validate_core_config
from .io_utils import read_table_with_fallback, write_table_with_fallback
from .multimodal_dataset import MODELED_MODALITIES
from .multimodal_train_eval import run_phase_e_multimodal
from .optuna_tune import (
    DEFAULT_OPTUNA_TRIALS,
    _parse_gpu_indices,
    run_optuna_multimodal_parallel,
    run_optuna_multimodal_study,
)

LOGGER = logging.getLogger(__name__)

TRIAL_MATRIX_COMPLETE_WINDOWS_REFERENCE = "all"
_VALID_MODELED_MODALITIES = set(MODELED_MODALITIES)


def _sanitize_slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", str(value).strip().lower()).strip("_")
    return slug or "trial"


def _study_directory(config: RunConfig, study_name: str) -> Path:
    return config.artifact_root / "trial_matrices" / study_name


def _default_study_name(config: RunConfig) -> str:
    return f"{config.run_id}_trial_matrix"


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)
    if not isinstance(payload, dict):
        raise ValueError("Trial matrix config must be a JSON object.")
    return payload


def _normalize_selected_modalities(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        tokens = [token.strip() for token in value.split(",") if token.strip()]
    elif isinstance(value, (list, tuple)):
        tokens = [str(token).strip() for token in value if str(token).strip()]
    else:
        raise ValueError("selected_modeled_modalities must be a list or comma-separated string.")
    if len(tokens) != len(set(tokens)):
        raise ValueError("selected_modeled_modalities must not contain duplicates.")
    invalid = sorted(token for token in tokens if token not in _VALID_MODELED_MODALITIES)
    if invalid:
        raise ValueError(
            "selected_modeled_modalities contains unsupported entries: "
            + ", ".join(invalid)
        )
    selected_set = set(tokens)
    return tuple(modality for modality in MODELED_MODALITIES if modality in selected_set)


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(int(value))
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    raise ValueError(f"Could not interpret boolean override value: {value!r}")


def _normalize_participant_folds(value: Any) -> tuple[tuple[int, ...], ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError("participant_folds must be a list of participant-id lists.")
    folds: list[tuple[int, ...]] = []
    for fold in value:
        if not isinstance(fold, (list, tuple)):
            raise ValueError("participant_folds must contain only participant-id lists.")
        folds.append(tuple(int(participant_id) for participant_id in fold))
    return tuple(folds)


def _apply_multimodal_override(config: RunConfig, key: str, value: Any) -> None:
    mt = config.multimodal_train
    if key == "use_native_frequency":
        mt.use_native_frequency = _coerce_bool(value)
    elif key == "task_mode":
        mt.task_mode = str(value)
    elif key == "split_mode":
        mt.split_mode = str(value)
    elif key == "num_folds":
        mt.num_folds = int(value)
    elif key == "participant_folds":
        mt.participant_folds = _normalize_participant_folds(value)
    elif key == "target_points":
        mt.target_points = int(value)
    elif key == "epochs":
        mt.epochs = int(value)
    elif key == "batch_size":
        mt.batch_size = int(value)
    elif key == "modality_dropout_prob":
        mt.modality_dropout_prob = float(value)
    elif key == "use_context":
        mt.use_context = _coerce_bool(value)
    elif key == "learning_rate":
        mt.learning_rate = float(value)
    elif key == "weight_decay":
        mt.weight_decay = float(value)
    elif key == "lambda_binary":
        mt.lambda_binary = float(value)
    elif key == "lambda_ordinal":
        mt.lambda_ordinal = float(value)
    elif key == "lambda_regression":
        mt.lambda_regression = float(value)
    elif key == "embedding_dim":
        mt.embedding_dim = int(value)
    elif key == "fusion_hidden_dim":
        mt.fusion_hidden_dim = int(value)
    elif key == "device":
        mt.device = str(value)
    elif key == "selected_modeled_modalities":
        mt.selected_modeled_modalities = _normalize_selected_modalities(value)
    elif key == "complete_windows_only":
        mt.complete_windows_only = _coerce_bool(value)
    elif key == "complete_windows_reference":
        mt.complete_windows_reference = str(value)
    else:
        raise ValueError(f"Unsupported trial override field: {key}")


def _resolve_trial_overrides(trial_payload: dict[str, Any]) -> dict[str, Any]:
    reserved = {
        "name",
        "selected_modeled_modalities",
        "overrides",
        "optuna_trials",
        "optuna_timeout_minutes",
        "optuna_study_name",
    }
    inline_overrides = {
        key: value
        for key, value in trial_payload.items()
        if key not in reserved
    }
    explicit_overrides = trial_payload.get("overrides", {})
    if explicit_overrides is None:
        explicit_overrides = {}
    if not isinstance(explicit_overrides, dict):
        raise ValueError("trial overrides must be a JSON object when provided.")
    return {
        **inline_overrides,
        **explicit_overrides,
    }


def _resolve_trial_specs(config_path: Path) -> dict[str, Any]:
    payload = _read_json(config_path)
    trials_payload = payload.get("trials")
    if not isinstance(trials_payload, list) or not trials_payload:
        raise ValueError("Trial matrix config must contain a non-empty 'trials' list.")

    global_overrides = payload.get("global_overrides", {})
    if global_overrides is None:
        global_overrides = {}
    if not isinstance(global_overrides, dict):
        raise ValueError("global_overrides must be a JSON object when provided.")

    trial_specs: list[dict[str, Any]] = []
    for trial_index, raw_trial in enumerate(trials_payload):
        if not isinstance(raw_trial, dict):
            raise ValueError("Each trial entry must be a JSON object.")
        trial_name = str(raw_trial.get("name") or f"trial_{trial_index + 1:03d}")
        selected_modalities = (
            raw_trial.get("selected_modeled_modalities")
            if "selected_modeled_modalities" in raw_trial
            else global_overrides.get("selected_modeled_modalities")
        )
        overrides = {
            **global_overrides,
            **_resolve_trial_overrides(raw_trial),
        }
        overrides.pop("selected_modeled_modalities", None)
        trial_specs.append(
            {
                "trial_index": int(trial_index),
                "name": trial_name,
                "selected_modeled_modalities": _normalize_selected_modalities(selected_modalities),
                "overrides": overrides,
                "optuna_trials": raw_trial.get("optuna_trials"),
                "optuna_timeout_minutes": raw_trial.get("optuna_timeout_minutes"),
                "optuna_study_name": raw_trial.get("optuna_study_name"),
            }
        )

    return {
        "study_name": payload.get("study_name"),
        "workers_per_gpu": payload.get("workers_per_gpu"),
        "gpu_list": payload.get("gpu_list"),
        "optuna_trials": payload.get("optuna_trials"),
        "optuna_timeout_minutes": payload.get("optuna_timeout_minutes"),
        "trial_specs": trial_specs,
    }


def _build_worker_slots(
    gpu_indices: list[int],
    *,
    workers_per_gpu: int,
) -> list[dict[str, int]]:
    if int(workers_per_gpu) <= 0:
        raise ValueError("trial workers_per_gpu must be positive.")
    slots: list[dict[str, int]] = []
    for gpu_index in gpu_indices:
        for slot_idx in range(int(workers_per_gpu)):
            slots.append({"gpu_index": int(gpu_index), "slot_idx": int(slot_idx)})
    return slots


def _summarize_trial_run(outputs: dict[str, Path]) -> dict[str, float]:
    fold_metrics_df = read_table_with_fallback(Path(outputs["metrics_per_fold"]))
    overall_metrics_df = read_table_with_fallback(Path(outputs["metrics_overall"]))
    binary_overall_df = read_table_with_fallback(Path(outputs["metrics_binary_overall"]))
    if fold_metrics_df.empty:
        raise ValueError("Trial run produced no per-fold metrics.")
    if overall_metrics_df.empty:
        raise ValueError("Trial run produced no overall metrics.")
    if binary_overall_df.empty:
        raise ValueError("Trial run produced no overall binary metrics.")

    overall_row = overall_metrics_df.iloc[0]
    binary_row = binary_overall_df.iloc[0]
    return {
        "mean_fold_mae": float(fold_metrics_df["mae"].mean()),
        "mean_fold_within_1_accuracy": float(fold_metrics_df["within_1_accuracy"].mean()),
        "mean_fold_binary_macro_f1": float(fold_metrics_df["test_binary_macro_f1"].mean()),
        "overall_mae": float(overall_row["mae"]),
        "overall_class_mae": float(overall_row["class_mae"]),
        "overall_within_1_accuracy": float(overall_row["within_1_accuracy"]),
        "overall_macro_f1": float(overall_row["macro_f1"]),
        "overall_accuracy": float(overall_row["accuracy"]),
        "overall_binary_macro_f1": float(binary_row["binary_macro_f1"]),
        "overall_binary_accuracy": float(binary_row["binary_accuracy"]),
        "num_samples": int(overall_row["num_samples"]),
    }


def _trial_sort_key(row: dict[str, Any]) -> tuple[float, float, float, float, float]:
    return (
        float(row["mean_fold_mae"]),
        -float(row["overall_within_1_accuracy"]),
        -float(row["overall_binary_macro_f1"]),
        float(row["overall_class_mae"]),
        -float(row["overall_macro_f1"]),
    )


def _configure_trial_run(
    base_config: RunConfig,
    *,
    trial_index: int,
    trial_name: str,
    selected_modeled_modalities: tuple[str, ...],
    overrides: dict[str, Any],
) -> RunConfig:
    config = copy.deepcopy(base_config)
    for key, value in overrides.items():
        _apply_multimodal_override(config, key, value)
    if selected_modeled_modalities:
        config.multimodal_train.selected_modeled_modalities = selected_modeled_modalities
    config.multimodal_train.complete_windows_only = True
    config.multimodal_train.complete_windows_reference = TRIAL_MATRIX_COMPLETE_WINDOWS_REFERENCE
    config.multimodal_train.use_multi_gpu = False
    config.run_id = (
        f"{base_config.run_id}__trial_matrix__{trial_index + 1:03d}__{_sanitize_slug(trial_name)}"
    )
    validate_core_config(config)
    config.ensure_directories()
    persist_run_metadata(config)
    return config


def _configure_trial_study_name(
    matrix_study_name: str,
    *,
    trial_spec: dict[str, Any],
) -> str:
    explicit_name = trial_spec.get("optuna_study_name")
    if explicit_name:
        return str(explicit_name)
    return (
        f"{matrix_study_name}__"
        f"{int(trial_spec['trial_index']) + 1:03d}__"
        f"{_sanitize_slug(str(trial_spec['name']))}"
    )


def _evaluate_trial(
    *,
    base_config: RunConfig,
    trial_spec: dict[str, Any],
) -> dict[str, Any]:
    trial_config = _configure_trial_run(
        base_config,
        trial_index=int(trial_spec["trial_index"]),
        trial_name=str(trial_spec["name"]),
        selected_modeled_modalities=tuple(trial_spec["selected_modeled_modalities"]),
        overrides=dict(trial_spec["overrides"]),
    )
    _, outputs, stats = run_phase_e_multimodal(trial_config)
    metrics = _summarize_trial_run(outputs)
    return {
        "trial_index": int(trial_spec["trial_index"]),
        "trial_name": str(trial_spec["name"]),
        "run_id": str(trial_config.run_id),
        "selected_modeled_modalities": tuple(
            trial_config.multimodal_train.selected_modeled_modalities or MODELED_MODALITIES
        ),
        "complete_windows_only": bool(trial_config.multimodal_train.complete_windows_only),
        "complete_windows_reference": str(trial_config.multimodal_train.complete_windows_reference),
        "task_mode": str(trial_config.multimodal_train.task_mode),
        "split_mode": str(trial_config.multimodal_train.split_mode),
        "batch_size": int(trial_config.multimodal_train.batch_size),
        "learning_rate": float(trial_config.multimodal_train.learning_rate),
        "weight_decay": float(trial_config.multimodal_train.weight_decay),
        "embedding_dim": int(trial_config.multimodal_train.embedding_dim),
        "fusion_hidden_dim": int(trial_config.multimodal_train.fusion_hidden_dim),
        "modality_dropout_prob": float(trial_config.multimodal_train.modality_dropout_prob),
        "use_context": bool(trial_config.multimodal_train.use_context),
        "lambda_binary": float(trial_config.multimodal_train.lambda_binary),
        "lambda_ordinal": float(trial_config.multimodal_train.lambda_ordinal),
        "lambda_regression": float(trial_config.multimodal_train.lambda_regression),
        "outputs_summary": str(outputs["summary"]),
        "num_prediction_rows": int(stats["num_prediction_rows"]),
        "overrides_json": json.dumps(trial_spec["overrides"], sort_keys=True),
        **metrics,
    }


def _trial_worker_entry(
    *,
    config: RunConfig,
    trial_spec: dict[str, Any],
    gpu_index: int,
    slot_idx: int,
    result_queue: Any,
) -> None:
    try:
        row = _evaluate_trial(base_config=config, trial_spec=trial_spec)
        row.update({"gpu_index": int(gpu_index), "worker_slot": int(slot_idx)})
        result_queue.put({"ok": True, "row": row})
    except Exception as exc:  # noqa: BLE001
        result_queue.put(
            {
                "ok": False,
                "gpu_index": int(gpu_index),
                "worker_slot": int(slot_idx),
                "trial_name": str(trial_spec.get("name", "")),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
        raise


def _evaluate_trial_specs(
    *,
    base_config: RunConfig,
    trial_specs: list[dict[str, Any]],
    workers_per_gpu: int,
    gpu_list: str | None,
) -> list[dict[str, Any]]:
    if not trial_specs:
        return []

    gpu_indices = _parse_gpu_indices(gpu_list)
    if not gpu_indices:
        return [
            {
                **_evaluate_trial(base_config=base_config, trial_spec=trial_spec),
                "gpu_index": np.nan,
                "worker_slot": np.nan,
            }
            for trial_spec in trial_specs
        ]

    worker_slots = _build_worker_slots(gpu_indices, workers_per_gpu=workers_per_gpu)
    if len(worker_slots) == 1:
        rows: list[dict[str, Any]] = []
        slot = worker_slots[0]
        for trial_spec in trial_specs:
            trial_config = copy.deepcopy(base_config)
            trial_config.multimodal_train.device = "cuda"
            trial_config.multimodal_train.use_multi_gpu = False
            trial_config.multimodal_train.cuda_device_index = int(slot["gpu_index"])
            row = _evaluate_trial(base_config=trial_config, trial_spec=trial_spec)
            row.update(
                {
                    "gpu_index": int(slot["gpu_index"]),
                    "worker_slot": int(slot["slot_idx"]),
                }
            )
            rows.append(row)
        return rows

    ctx = mp.get_context("spawn")
    rows: list[dict[str, Any]] = []
    for chunk_start in range(0, len(trial_specs), len(worker_slots)):
        chunk = trial_specs[chunk_start : chunk_start + len(worker_slots)]
        result_queue = ctx.Queue()
        processes: list[Any] = []
        for trial_spec, slot in zip(chunk, worker_slots, strict=False):
            trial_config = copy.deepcopy(base_config)
            trial_config.multimodal_train.device = "cuda"
            trial_config.multimodal_train.use_multi_gpu = False
            trial_config.multimodal_train.cuda_device_index = int(slot["gpu_index"])
            process = ctx.Process(
                target=_trial_worker_entry,
                kwargs={
                    "config": trial_config,
                    "trial_spec": dict(trial_spec),
                    "gpu_index": int(slot["gpu_index"]),
                    "slot_idx": int(slot["slot_idx"]),
                    "result_queue": result_queue,
                },
            )
            process.start()
            processes.append(process)

        chunk_results: list[dict[str, Any]] = []
        for _ in range(len(chunk)):
            chunk_results.append(result_queue.get())
        for process in processes:
            process.join()

        failures = [result for result in chunk_results if not result.get("ok", False)]
        if failures:
            first_failure = failures[0]
            raise RuntimeError(
                "One or more trial-matrix workers failed.\n"
                f"First failure on gpu={first_failure.get('gpu_index')}, "
                f"slot={first_failure.get('worker_slot')}, "
                f"trial={first_failure.get('trial_name')}:\n"
                f"{first_failure.get('error')}\n"
                f"{first_failure.get('traceback', '')}"
            )
        rows.extend(result["row"] for result in chunk_results)
    return rows


def run_trial_matrix(
    config: RunConfig,
    *,
    config_path: str | Path,
    study_name: str | None = None,
    workers_per_gpu: int | None = None,
    gpu_list: str | None = None,
) -> tuple[dict[str, Path], dict[str, Any]]:
    config = copy.deepcopy(config)
    config.multimodal_train.use_multi_gpu = False

    resolved_config_path = Path(config_path).expanduser().resolve()
    if not resolved_config_path.exists():
        raise FileNotFoundError(f"Trial matrix config does not exist: {resolved_config_path}")

    payload = _resolve_trial_specs(resolved_config_path)
    resolved_study_name = str(
        study_name
        or payload.get("study_name")
        or _default_study_name(config)
    )
    resolved_workers_per_gpu = int(
        workers_per_gpu
        if workers_per_gpu is not None
        else (payload.get("workers_per_gpu") or 1)
    )
    resolved_gpu_list = gpu_list if gpu_list is not None else payload.get("gpu_list")
    trial_specs = list(payload["trial_specs"])

    study_dir = _study_directory(config, resolved_study_name)
    study_dir.mkdir(parents=True, exist_ok=True)

    rows = _evaluate_trial_specs(
        base_config=config,
        trial_specs=trial_specs,
        workers_per_gpu=resolved_workers_per_gpu,
        gpu_list=resolved_gpu_list,
    )
    if not rows:
        raise RuntimeError("Trial matrix did not produce any runs.")

    rows_df = pd.DataFrame(
        [
            {
                **row,
                "selected_modeled_modalities": json.dumps(list(row["selected_modeled_modalities"])),
            }
            for row in rows
        ]
    )
    trials_path = write_table_with_fallback(
        rows_df,
        study_dir / "trials.parquet",
        logger=LOGGER,
    )

    requested_payload = {
        "study_name": resolved_study_name,
        "workers_per_gpu": int(resolved_workers_per_gpu),
        "gpu_list": None if resolved_gpu_list is None else str(resolved_gpu_list),
        "config_path": str(resolved_config_path),
        "complete_windows_only": True,
        "complete_windows_reference": TRIAL_MATRIX_COMPLETE_WINDOWS_REFERENCE,
        "trials": [
            {
                "trial_index": int(trial_spec["trial_index"]),
                "name": str(trial_spec["name"]),
                "selected_modeled_modalities": list(
                    trial_spec["selected_modeled_modalities"] or MODELED_MODALITIES
                ),
                "overrides": dict(trial_spec["overrides"]),
            }
            for trial_spec in trial_specs
        ],
    }
    requested_path = study_dir / "requested_trials.json"
    with requested_path.open("w", encoding="utf-8") as fp:
        json.dump(requested_payload, fp, indent=2, sort_keys=True)

    best_row = min(rows, key=_trial_sort_key)
    summary_payload = {
        "study_name": resolved_study_name,
        "config_path": str(resolved_config_path),
        "workers_per_gpu": int(resolved_workers_per_gpu),
        "gpu_list": None if resolved_gpu_list is None else str(resolved_gpu_list),
        "complete_windows_only": True,
        "complete_windows_reference": TRIAL_MATRIX_COMPLETE_WINDOWS_REFERENCE,
        "num_trials": int(len(rows)),
        "best_trial_name": str(best_row["trial_name"]),
        "best_run_id": str(best_row["run_id"]),
        "best_selected_modeled_modalities": list(best_row["selected_modeled_modalities"]),
        "best_mean_fold_mae": float(best_row["mean_fold_mae"]),
        "best_overall_mae": float(best_row["overall_mae"]),
        "best_overall_class_mae": float(best_row["overall_class_mae"]),
        "best_overall_within_1_accuracy": float(best_row["overall_within_1_accuracy"]),
        "best_overall_binary_macro_f1": float(best_row["overall_binary_macro_f1"]),
        "best_overall_macro_f1": float(best_row["overall_macro_f1"]),
    }
    summary_path = study_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(summary_payload, fp, indent=2, sort_keys=True)

    outputs = {
        "trial_matrix_trials": trials_path,
        "trial_matrix_requested": requested_path,
        "trial_matrix_summary": summary_path,
    }
    stats = {
        "study_name": resolved_study_name,
        "num_trials": int(len(rows)),
        "workers_per_gpu": int(resolved_workers_per_gpu),
        "gpu_list": None if resolved_gpu_list is None else str(resolved_gpu_list),
        "best_trial_name": str(best_row["trial_name"]),
        "best_run_id": str(best_row["run_id"]),
        "best_mean_fold_mae": float(best_row["mean_fold_mae"]),
        "best_overall_binary_macro_f1": float(best_row["overall_binary_macro_f1"]),
    }
    return outputs, stats


def run_trial_matrix_optuna(
    config: RunConfig,
    *,
    config_path: str | Path,
    study_name: str | None = None,
    workers_per_gpu: int | None = None,
    gpu_list: str | None = None,
    n_trials: int | None = None,
    timeout_minutes: float | None = None,
) -> tuple[dict[str, Path], dict[str, Any]]:
    config = copy.deepcopy(config)
    config.multimodal_train.use_multi_gpu = False

    resolved_config_path = Path(config_path).expanduser().resolve()
    if not resolved_config_path.exists():
        raise FileNotFoundError(f"Trial matrix config does not exist: {resolved_config_path}")

    payload = _resolve_trial_specs(resolved_config_path)
    resolved_matrix_study_name = str(
        study_name
        or payload.get("study_name")
        or _default_study_name(config)
    )
    resolved_workers_per_gpu = int(
        workers_per_gpu
        if workers_per_gpu is not None
        else (payload.get("workers_per_gpu") or 1)
    )
    resolved_gpu_list = gpu_list if gpu_list is not None else payload.get("gpu_list")
    default_optuna_trials = int(
        n_trials
        if n_trials is not None
        else (payload.get("optuna_trials") or DEFAULT_OPTUNA_TRIALS)
    )
    default_timeout_minutes = (
        timeout_minutes
        if timeout_minutes is not None
        else payload.get("optuna_timeout_minutes")
    )
    trial_specs = list(payload["trial_specs"])

    matrix_dir = _study_directory(config, resolved_matrix_study_name)
    matrix_dir.mkdir(parents=True, exist_ok=True)

    result_rows: list[dict[str, Any]] = []
    for trial_spec in trial_specs:
        trial_config = _configure_trial_run(
            config,
            trial_index=int(trial_spec["trial_index"]),
            trial_name=str(trial_spec["name"]),
            selected_modeled_modalities=tuple(trial_spec["selected_modeled_modalities"]),
            overrides=dict(trial_spec["overrides"]),
        )
        resolved_optuna_trials = int(
            trial_spec.get("optuna_trials")
            if trial_spec.get("optuna_trials") is not None
            else default_optuna_trials
        )
        resolved_trial_timeout = (
            trial_spec.get("optuna_timeout_minutes")
            if trial_spec.get("optuna_timeout_minutes") is not None
            else default_timeout_minutes
        )
        resolved_trial_study_name = _configure_trial_study_name(
            resolved_matrix_study_name,
            trial_spec=trial_spec,
        )

        LOGGER.info(
            "Starting trial-matrix Optuna study | trial=%s | study=%s | modalities=%s | optuna_trials=%s",
            str(trial_spec["name"]),
            resolved_trial_study_name,
            ",".join(
                trial_config.multimodal_train.selected_modeled_modalities or MODELED_MODALITIES
            ),
            resolved_optuna_trials,
        )

        if resolved_gpu_list is None and not _parse_gpu_indices(None):
            outputs, stats = run_optuna_multimodal_study(
                trial_config,
                n_trials=resolved_optuna_trials,
                timeout_minutes=resolved_trial_timeout,
                study_name=resolved_trial_study_name,
            )
        else:
            outputs, stats = run_optuna_multimodal_parallel(
                trial_config,
                n_trials=resolved_optuna_trials,
                workers_per_gpu=resolved_workers_per_gpu,
                gpu_list=resolved_gpu_list,
                timeout_minutes=resolved_trial_timeout,
                study_name=resolved_trial_study_name,
            )

        optuna_summary_path = Path(outputs["optuna_summary"])
        with optuna_summary_path.open("r", encoding="utf-8") as fp:
            optuna_summary = json.load(fp)

        result_rows.append(
            {
                "trial_index": int(trial_spec["trial_index"]),
                "trial_name": str(trial_spec["name"]),
                "matrix_study_name": resolved_matrix_study_name,
                "optuna_study_name": resolved_trial_study_name,
                "selected_modeled_modalities": tuple(
                    trial_config.multimodal_train.selected_modeled_modalities or MODELED_MODALITIES
                ),
                "complete_windows_only": bool(trial_config.multimodal_train.complete_windows_only),
                "complete_windows_reference": str(trial_config.multimodal_train.complete_windows_reference),
                "requested_optuna_trials": int(resolved_optuna_trials),
                "requested_timeout_minutes": (
                    np.nan if resolved_trial_timeout is None else float(resolved_trial_timeout)
                ),
                "best_trial_number": int(stats["best_trial_number"]),
                "best_value": float(stats["best_value"]),
                "best_run_id": str(stats["best_run_id"]),
                "best_mean_fold_mae": float(optuna_summary.get("best_mean_fold_mae", np.nan)),
                "best_mean_fold_binary_macro_f1": float(
                    optuna_summary.get("best_mean_fold_binary_macro_f1", np.nan)
                ),
                "optuna_summary_path": str(outputs["optuna_summary"]),
                "optuna_trials_path": str(outputs["optuna_trials"]),
                "optuna_storage_path": str(outputs["optuna_storage"]),
                "overrides_json": json.dumps(trial_spec["overrides"], sort_keys=True),
            }
        )

    if not result_rows:
        raise RuntimeError("Trial-matrix Optuna did not produce any studies.")

    results_df = pd.DataFrame(
        [
            {
                **row,
                "selected_modeled_modalities": json.dumps(list(row["selected_modeled_modalities"])),
            }
            for row in result_rows
        ]
    )
    studies_path = write_table_with_fallback(
        results_df,
        matrix_dir / "subset_studies.parquet",
        logger=LOGGER,
    )

    requested_payload = {
        "study_name": resolved_matrix_study_name,
        "workers_per_gpu": int(resolved_workers_per_gpu),
        "gpu_list": None if resolved_gpu_list is None else str(resolved_gpu_list),
        "config_path": str(resolved_config_path),
        "complete_windows_only": True,
        "complete_windows_reference": TRIAL_MATRIX_COMPLETE_WINDOWS_REFERENCE,
        "default_optuna_trials": int(default_optuna_trials),
        "default_optuna_timeout_minutes": default_timeout_minutes,
        "trials": [
            {
                "trial_index": int(trial_spec["trial_index"]),
                "name": str(trial_spec["name"]),
                "selected_modeled_modalities": list(
                    trial_spec["selected_modeled_modalities"] or MODELED_MODALITIES
                ),
                "overrides": dict(trial_spec["overrides"]),
                "optuna_trials": trial_spec.get("optuna_trials"),
                "optuna_timeout_minutes": trial_spec.get("optuna_timeout_minutes"),
                "optuna_study_name": trial_spec.get("optuna_study_name"),
            }
            for trial_spec in trial_specs
        ],
    }
    requested_path = matrix_dir / "requested_subset_studies.json"
    with requested_path.open("w", encoding="utf-8") as fp:
        json.dump(requested_payload, fp, indent=2, sort_keys=True)

    best_row = min(
        result_rows,
        key=lambda row: (
            float(row["best_mean_fold_mae"]),
            -float(row["best_mean_fold_binary_macro_f1"]),
        ),
    )
    summary_payload = {
        "study_name": resolved_matrix_study_name,
        "config_path": str(resolved_config_path),
        "workers_per_gpu": int(resolved_workers_per_gpu),
        "gpu_list": None if resolved_gpu_list is None else str(resolved_gpu_list),
        "complete_windows_only": True,
        "complete_windows_reference": TRIAL_MATRIX_COMPLETE_WINDOWS_REFERENCE,
        "default_optuna_trials": int(default_optuna_trials),
        "default_optuna_timeout_minutes": default_timeout_minutes,
        "num_subset_studies": int(len(result_rows)),
        "best_trial_name": str(best_row["trial_name"]),
        "best_optuna_study_name": str(best_row["optuna_study_name"]),
        "best_run_id": str(best_row["best_run_id"]),
        "best_selected_modeled_modalities": list(best_row["selected_modeled_modalities"]),
        "best_mean_fold_mae": float(best_row["best_mean_fold_mae"]),
        "best_mean_fold_binary_macro_f1": float(best_row["best_mean_fold_binary_macro_f1"]),
    }
    summary_path = matrix_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(summary_payload, fp, indent=2, sort_keys=True)

    outputs = {
        "trial_matrix_optuna_studies": studies_path,
        "trial_matrix_optuna_requested": requested_path,
        "trial_matrix_optuna_summary": summary_path,
    }
    stats = {
        "study_name": resolved_matrix_study_name,
        "num_subset_studies": int(len(result_rows)),
        "best_trial_name": str(best_row["trial_name"]),
        "best_optuna_study_name": str(best_row["optuna_study_name"]),
        "best_run_id": str(best_row["best_run_id"]),
        "best_mean_fold_mae": float(best_row["best_mean_fold_mae"]),
    }
    return outputs, stats
