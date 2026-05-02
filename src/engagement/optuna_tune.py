from __future__ import annotations

import copy
import importlib
import inspect
import json
import logging
import multiprocessing as mp
from pathlib import Path
import traceback
from typing import Any

import numpy as np
import pandas as pd

from .config import RunConfig, persist_run_metadata
from .io_utils import read_table_with_fallback, write_table_with_fallback
from .multimodal_train_eval import run_phase_e_multimodal

LOGGER = logging.getLogger(__name__)

DEFAULT_OPTUNA_TRIALS = 20
DEFAULT_SEARCH_SPACE = {
    "learning_rate": {"low": 1e-4, "high": 1e-3, "log": True},
    "weight_decay": {"low": 1e-6, "high": 1e-3, "log": True},
    "batch_size": [8, 16, 32],
    "embedding_dim": [32, 64, 128],
    "fusion_hidden_dim": [64, 96, 128, 192],
}


def _import_optuna() -> Any:
    try:
        return importlib.import_module("optuna")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Optuna is not installed. Install it with `python -m pip install optuna` "
            "before running the tuning stage."
        ) from exc


def _default_study_name(config: RunConfig) -> str:
    return (
        f"{config.run_id}_"
        f"{config.multimodal_train.task_mode}_"
        f"{config.multimodal_train.split_mode}_optuna"
    )


def _study_directory(config: RunConfig, study_name: str) -> Path:
    return config.artifact_root / "optuna" / study_name


def _default_storage_url(study_dir: Path) -> str:
    db_path = (study_dir / "study.db").resolve()
    return f"sqlite:///{db_path.as_posix()}"


def _trial_run_id(base_run_id: str, trial_number: int) -> str:
    return f"{base_run_id}__trial_{trial_number:03d}"


def _apply_trial_hyperparameters(trial: Any, config: RunConfig) -> None:
    config.multimodal_train.learning_rate = float(
        trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True)
    )
    config.multimodal_train.weight_decay = float(
        trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    )
    config.multimodal_train.batch_size = int(
        trial.suggest_categorical("batch_size", [8, 16, 32])
    )
    config.multimodal_train.embedding_dim = int(
        trial.suggest_categorical("embedding_dim", [32, 64, 128])
    )
    config.multimodal_train.fusion_hidden_dim = int(
        trial.suggest_categorical("fusion_hidden_dim", [64, 96, 128, 192])
    )


def _read_metric_summaries(
    outputs: dict[str, Path],
    *,
    task_mode: str,
) -> dict[str, float]:
    fold_metrics_df = read_table_with_fallback(Path(outputs["metrics_per_fold"]))
    binary_fold_metrics_df = read_table_with_fallback(
        Path(outputs["metrics_binary_per_fold"])
    )
    overall_metrics_df = read_table_with_fallback(Path(outputs["metrics_overall"]))
    overall_binary_metrics_df = read_table_with_fallback(
        Path(outputs["metrics_binary_overall"])
    )

    if fold_metrics_df.empty:
        raise ValueError("Optuna trial produced no per-fold metrics.")
    if binary_fold_metrics_df.empty:
        raise ValueError("Optuna trial produced no binary per-fold metrics.")
    if overall_metrics_df.empty:
        raise ValueError("Optuna trial produced no overall ordinal metrics.")
    if overall_binary_metrics_df.empty:
        raise ValueError("Optuna trial produced no overall binary metrics.")

    mean_fold_binary_macro_f1 = float(binary_fold_metrics_df["binary_macro_f1"].mean())
    overall_binary_macro_f1 = float(
        overall_binary_metrics_df.iloc[0]["binary_macro_f1"]
    )

    if task_mode == "binary":
        mean_fold_primary = mean_fold_binary_macro_f1
        mean_fold_mae = float("nan")
        overall_primary = overall_binary_macro_f1
    else:
        mean_fold_primary = float(fold_metrics_df["mae"].mean())
        mean_fold_mae = float(fold_metrics_df["mae"].mean())
        overall_primary = float(overall_metrics_df.iloc[0]["mae"])

    return {
        "mean_fold_primary_metric": mean_fold_primary,
        "mean_fold_mae": mean_fold_mae,
        "mean_fold_binary_macro_f1": mean_fold_binary_macro_f1,
        "overall_primary_metric": overall_primary,
        "overall_binary_macro_f1": overall_binary_macro_f1,
        "overall_mae": (
            float(overall_metrics_df.iloc[0]["mae"])
            if task_mode != "binary"
            else float("nan")
        ),
    }


def _trial_rows(study: Any) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for trial in study.trials:
        row: dict[str, Any] = {
            "trial_number": int(trial.number),
            "state": str(trial.state),
            "value": (
                float(trial.value)
                if trial.value is not None and np.isfinite(float(trial.value))
                else np.nan
            ),
        }
        for key, value in trial.params.items():
            row[f"param_{key}"] = value
        for key, value in trial.user_attrs.items():
            row[key] = value
        rows.append(row)
    return pd.DataFrame(rows)


def _build_tpe_sampler(
    optuna: Any,
    *,
    seed: int,
    constant_liar: bool,
) -> Any:
    sampler_kwargs: dict[str, Any] = {"seed": int(seed)}
    try:
        sampler_signature = inspect.signature(optuna.samplers.TPESampler)
        if constant_liar and "constant_liar" in sampler_signature.parameters:
            sampler_kwargs["constant_liar"] = True
    except (TypeError, ValueError):
        pass
    return optuna.samplers.TPESampler(**sampler_kwargs)


def _initialize_study_storage(
    *,
    config: RunConfig,
    study_name: str,
    storage_url: str,
) -> None:
    optuna = _import_optuna()
    task_mode = str(config.multimodal_train.task_mode)
    sampler = _build_tpe_sampler(
        optuna,
        seed=int(config.seed),
        constant_liar=False,
    )
    optuna.create_study(
        study_name=str(study_name),
        storage=str(storage_url),
        direction="maximize" if task_mode == "binary" else "minimize",
        load_if_exists=True,
        sampler=sampler,
    )


def _available_cuda_gpu_indices() -> list[int]:
    torch = importlib.import_module("torch")
    try:
        count = int(torch.cuda.device_count())
    except Exception:
        count = 0
    return list(range(max(0, count)))


def _parse_gpu_indices(
    gpu_list: str | None,
    *,
    available_gpu_indices: list[int] | None = None,
) -> list[int]:
    available = (
        list(available_gpu_indices)
        if available_gpu_indices is not None
        else _available_cuda_gpu_indices()
    )
    if not available:
        return []
    if gpu_list is None or not str(gpu_list).strip():
        return available

    resolved: list[int] = []
    seen: set[int] = set()
    for raw_part in str(gpu_list).split(","):
        token = raw_part.strip()
        if not token:
            continue
        gpu_index = int(token)
        if gpu_index not in available:
            raise ValueError(
                f"Requested GPU {gpu_index} is not available. Visible GPUs: {available}."
            )
        if gpu_index in seen:
            continue
        resolved.append(gpu_index)
        seen.add(gpu_index)
    if not resolved:
        raise ValueError("No usable GPU indices were parsed for Optuna workers.")
    return resolved


def _distribute_trials(total_trials: int, num_workers: int) -> list[int]:
    if total_trials <= 0:
        raise ValueError("Optuna total trials must be positive.")
    if num_workers <= 0:
        raise ValueError("Optuna worker count must be positive.")
    base = total_trials // num_workers
    remainder = total_trials % num_workers
    return [
        base + (1 if worker_idx < remainder else 0)
        for worker_idx in range(num_workers)
    ]


def _summarize_study(
    *,
    config: RunConfig,
    study: Any,
    study_name: str,
    storage_url: str,
    num_trials_requested: int,
) -> tuple[dict[str, Path], dict[str, Any]]:
    study_dir = _study_directory(config, study_name)
    trials_df = _trial_rows(study)
    trials_path = write_table_with_fallback(
        trials_df,
        study_dir / "trials.parquet",
        logger=LOGGER,
    )

    best_trial = study.best_trial
    task_mode = str(config.multimodal_train.task_mode)
    primary_metric_name = (
        "mean_fold_binary_macro_f1"
        if task_mode == "binary"
        else "mean_fold_mae"
    )
    num_trials_completed = int(
        sum(1 for trial in study.trials if str(trial.state) == "TrialState.COMPLETE")
    )
    summary_payload = {
        "study_name": study_name,
        "storage_url": storage_url,
        "task_mode": task_mode,
        "split_mode": str(config.multimodal_train.split_mode),
        "num_trials_requested": int(num_trials_requested),
        "num_trials_total": int(len(study.trials)),
        "num_trials_completed": num_trials_completed,
        "primary_metric_name": primary_metric_name,
        "best_trial_number": int(best_trial.number),
        "best_value": float(best_trial.value),
        "best_params": dict(best_trial.params),
        "best_run_id": str(best_trial.user_attrs.get("run_id", "")),
        "best_mean_fold_mae": float(best_trial.user_attrs.get("mean_fold_mae", np.nan)),
        "best_mean_fold_binary_macro_f1": float(
            best_trial.user_attrs.get("mean_fold_binary_macro_f1", np.nan)
        ),
        "search_space": DEFAULT_SEARCH_SPACE,
    }
    summary_path = study_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(summary_payload, fp, indent=2, sort_keys=True)

    outputs = {
        "optuna_trials": trials_path,
        "optuna_summary": summary_path,
        "optuna_storage": Path(study_dir / "study.db"),
    }
    stats = {
        "study_name": study_name,
        "num_trials_total": int(len(study.trials)),
        "num_trials_completed": num_trials_completed,
        "primary_metric_name": primary_metric_name,
        "best_trial_number": int(best_trial.number),
        "best_value": float(best_trial.value),
        "best_run_id": str(best_trial.user_attrs.get("run_id", "")),
    }
    return outputs, stats


def run_optuna_multimodal_study(
    config: RunConfig,
    *,
    n_trials: int = DEFAULT_OPTUNA_TRIALS,
    timeout_minutes: float | None = None,
    study_name: str | None = None,
    storage_url: str | None = None,
    sampler_seed: int | None = None,
    use_constant_liar: bool = False,
) -> tuple[dict[str, Path], dict[str, Any]]:
    optuna = _import_optuna()

    resolved_study_name = str(study_name or _default_study_name(config))
    study_dir = _study_directory(config, resolved_study_name)
    study_dir.mkdir(parents=True, exist_ok=True)
    resolved_storage_url = str(storage_url or _default_storage_url(study_dir))
    timeout_seconds = (
        None if timeout_minutes is None else max(0.0, float(timeout_minutes) * 60.0)
    )
    base_run_id = str(config.run_id)
    task_mode = str(config.multimodal_train.task_mode)
    primary_metric_name = (
        "mean_fold_binary_macro_f1"
        if task_mode == "binary"
        else "mean_fold_mae"
    )

    resolved_sampler_seed = int(config.seed if sampler_seed is None else sampler_seed)
    sampler = _build_tpe_sampler(
        optuna,
        seed=resolved_sampler_seed,
        constant_liar=bool(use_constant_liar),
    )
    study = optuna.create_study(
        study_name=resolved_study_name,
        storage=resolved_storage_url,
        direction="maximize" if task_mode == "binary" else "minimize",
        load_if_exists=True,
        sampler=sampler,
    )

    LOGGER.info(
        "Starting Optuna study | study=%s | task=%s | split=%s | trials=%s | storage=%s",
        resolved_study_name,
        task_mode,
        str(config.multimodal_train.split_mode),
        int(n_trials),
        resolved_storage_url,
    )

    def _objective(trial: Any) -> float:
        trial_config = copy.deepcopy(config)
        _apply_trial_hyperparameters(trial, trial_config)
        trial_config.run_id = _trial_run_id(base_run_id, int(trial.number))
        trial_config.ensure_directories()
        persist_run_metadata(trial_config)

        _, outputs, _ = run_phase_e_multimodal(trial_config)
        metrics = _read_metric_summaries(outputs, task_mode=task_mode)

        trial.set_user_attr("run_id", trial_config.run_id)
        trial.set_user_attr(
            "mean_fold_primary_metric", metrics["mean_fold_primary_metric"]
        )
        trial.set_user_attr("mean_fold_mae", metrics["mean_fold_mae"])
        trial.set_user_attr(
            "mean_fold_binary_macro_f1", metrics["mean_fold_binary_macro_f1"]
        )
        trial.set_user_attr(
            "overall_primary_metric", metrics["overall_primary_metric"]
        )
        trial.set_user_attr(
            "overall_binary_macro_f1", metrics["overall_binary_macro_f1"]
        )
        trial.set_user_attr("overall_mae", metrics["overall_mae"])
        trial.set_user_attr("summary_path", str(outputs["summary"]))

        LOGGER.info(
            "Optuna trial complete | trial=%s | run_id=%s | %s=%.4f | mean_fold_mae=%s | mean_fold_binary_macro_f1=%.4f | params=%s",
            int(trial.number),
            trial_config.run_id,
            primary_metric_name,
            metrics["mean_fold_primary_metric"],
            (
                "n/a"
                if task_mode == "binary"
                else f"{metrics['mean_fold_mae']:.4f}"
            ),
            metrics["mean_fold_binary_macro_f1"],
            json.dumps(trial.params, sort_keys=True),
        )
        return float(metrics["mean_fold_primary_metric"])

    study.optimize(
        _objective,
        n_trials=int(n_trials),
        timeout=timeout_seconds,
        gc_after_trial=True,
    )
    return _summarize_study(
        config=config,
        study=study,
        study_name=resolved_study_name,
        storage_url=resolved_storage_url,
        num_trials_requested=int(n_trials),
    )


def _optuna_worker_entry(
    config: RunConfig,
    *,
    n_trials: int,
    timeout_minutes: float | None,
    study_name: str,
    storage_url: str,
    sampler_seed: int,
    use_constant_liar: bool,
    result_queue: Any,
) -> None:
    try:
        outputs, stats = run_optuna_multimodal_study(
            config,
            n_trials=n_trials,
            timeout_minutes=timeout_minutes,
            study_name=study_name,
            storage_url=storage_url,
            sampler_seed=sampler_seed,
            use_constant_liar=use_constant_liar,
        )
        result_queue.put(
            {
                "ok": True,
                "run_id": config.run_id,
                "gpu_index": config.multimodal_train.cuda_device_index,
                "sampler_seed": int(sampler_seed),
                "outputs": {k: str(v) for k, v in outputs.items()},
                "stats": stats,
            }
        )
    except Exception as exc:  # noqa: BLE001
        result_queue.put(
            {
                "ok": False,
                "run_id": config.run_id,
                "gpu_index": config.multimodal_train.cuda_device_index,
                "sampler_seed": int(sampler_seed),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
        raise


def run_optuna_multimodal_parallel(
    config: RunConfig,
    *,
    n_trials: int,
    workers_per_gpu: int,
    gpu_list: str | None = None,
    timeout_minutes: float | None = None,
    study_name: str | None = None,
    storage_url: str | None = None,
) -> tuple[dict[str, Path], dict[str, Any]]:
    if int(workers_per_gpu) <= 0:
        raise ValueError("optuna workers_per_gpu must be positive.")

    gpu_indices = _parse_gpu_indices(gpu_list)
    if not gpu_indices:
        raise RuntimeError(
            "No CUDA GPUs are available for parallel Optuna workers."
        )

    resolved_study_name = str(study_name or _default_study_name(config))
    study_dir = _study_directory(config, resolved_study_name)
    study_dir.mkdir(parents=True, exist_ok=True)
    resolved_storage_url = str(storage_url or _default_storage_url(study_dir))
    _initialize_study_storage(
        config=config,
        study_name=resolved_study_name,
        storage_url=resolved_storage_url,
    )

    num_workers = len(gpu_indices) * int(workers_per_gpu)
    trial_allocation = _distribute_trials(int(n_trials), num_workers)

    LOGGER.info(
        "Starting parallel Optuna launcher | study=%s | gpus=%s | workers_per_gpu=%s | total_workers=%s | total_trials=%s",
        resolved_study_name,
        ",".join(str(idx) for idx in gpu_indices),
        int(workers_per_gpu),
        int(num_workers),
        int(n_trials),
    )

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    worker_processes: list[Any] = []
    worker_specs: list[dict[str, Any]] = []
    worker_idx = 0
    for gpu_index in gpu_indices:
        for slot_idx in range(int(workers_per_gpu)):
            worker_trials = int(trial_allocation[worker_idx])
            sampler_seed = int(config.seed) + worker_idx
            worker_idx += 1
            if worker_trials <= 0:
                continue
            worker_config = copy.deepcopy(config)
            worker_config.multimodal_train.device = "cuda"
            worker_config.multimodal_train.use_multi_gpu = False
            worker_config.multimodal_train.cuda_device_index = int(gpu_index)
            worker_config.run_id = (
                f"{config.run_id}__worker_gpu{gpu_index}_slot{slot_idx}"
            )
            worker_config.ensure_directories()
            persist_run_metadata(worker_config)
            worker_specs.append(
                {
                    "run_id": worker_config.run_id,
                    "gpu_index": int(gpu_index),
                    "slot": int(slot_idx),
                    "n_trials": worker_trials,
                    "sampler_seed": sampler_seed,
                }
            )
            process = ctx.Process(
                target=_optuna_worker_entry,
                kwargs={
                    "config": worker_config,
                    "n_trials": worker_trials,
                    "timeout_minutes": timeout_minutes,
                    "study_name": resolved_study_name,
                    "storage_url": resolved_storage_url,
                    "sampler_seed": sampler_seed,
                    "use_constant_liar": True,
                    "result_queue": result_queue,
                },
                name=f"optuna_gpu{gpu_index}_slot{slot_idx}",
            )
            worker_processes.append(process)

    for spec in worker_specs:
        LOGGER.info(
            "Launching Optuna worker | run_id=%s | gpu=%s | slot=%s | trials=%s | sampler_seed=%s",
            spec["run_id"],
            spec["gpu_index"],
            spec["slot"],
            spec["n_trials"],
            spec["sampler_seed"],
        )

    for process in worker_processes:
        process.start()

    worker_results: list[dict[str, Any]] = []
    for _ in worker_processes:
        worker_results.append(result_queue.get())

    failed_results = [result for result in worker_results if not bool(result.get("ok"))]
    for process in worker_processes:
        process.join()

    if failed_results:
        first_failure = failed_results[0]
        raise RuntimeError(
            "One or more Optuna workers failed.\n"
            f"First failure on gpu={first_failure.get('gpu_index')}, run_id={first_failure.get('run_id')}:\n"
            f"{first_failure.get('error')}\n"
            f"{first_failure.get('traceback', '')}"
        )

    optuna = _import_optuna()
    sampler = _build_tpe_sampler(
        optuna,
        seed=int(config.seed),
        constant_liar=True,
    )
    study = optuna.create_study(
        study_name=resolved_study_name,
        storage=resolved_storage_url,
        direction="maximize",
        load_if_exists=True,
        sampler=sampler,
    )
    outputs, stats = _summarize_study(
        config=config,
        study=study,
        study_name=resolved_study_name,
        storage_url=resolved_storage_url,
        num_trials_requested=int(n_trials),
    )
    stats["worker_count"] = int(len(worker_specs))
    stats["workers_per_gpu"] = int(workers_per_gpu)
    stats["gpu_indices"] = list(gpu_indices)
    stats["worker_specs"] = worker_specs
    return outputs, stats
