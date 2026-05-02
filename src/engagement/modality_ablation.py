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

from .config import RunConfig, persist_run_metadata
from .io_utils import read_table_with_fallback, write_table_with_fallback
from .multimodal_dataset import MODELED_MODALITIES
from .multimodal_train_eval import run_phase_e_multimodal
from .optuna_tune import _parse_gpu_indices

LOGGER = logging.getLogger(__name__)

DEVICE_MODALITY_GROUPS: dict[str, tuple[str, ...]] = {
    "t_ring": ("ring_ppg", "ring_temp", "ring_imu"),
    "microsoft_band_2": ("eda", "hr"),
    "esense": ("imu_esense",),
    "polar_h10": ("ecg",),
    "muse_s_athena": ("eeg", "ppg", "imu_muse"),
    "beam_eye_tracker": ("eye",),
}

COMPLETE_ONLY_REGRESSION_PRESET = {
    "task_mode": "ordinal",
    "complete_windows_only": True,
    "lambda_ordinal": 0.0,
    "lambda_regression": 1.0,
    "batch_size": 16,
    "embedding_dim": 128,
    "fusion_hidden_dim": 64,
    "learning_rate": 6.654422028769585e-4,
    "weight_decay": 1.1977162578030681e-6,
}


def apply_complete_only_regression_preset(config: RunConfig) -> None:
    config.multimodal_train.task_mode = str(COMPLETE_ONLY_REGRESSION_PRESET["task_mode"])
    config.multimodal_train.complete_windows_only = bool(
        COMPLETE_ONLY_REGRESSION_PRESET["complete_windows_only"]
    )
    config.multimodal_train.lambda_ordinal = float(
        COMPLETE_ONLY_REGRESSION_PRESET["lambda_ordinal"]
    )
    config.multimodal_train.lambda_regression = float(
        COMPLETE_ONLY_REGRESSION_PRESET["lambda_regression"]
    )
    config.multimodal_train.batch_size = int(COMPLETE_ONLY_REGRESSION_PRESET["batch_size"])
    config.multimodal_train.embedding_dim = int(
        COMPLETE_ONLY_REGRESSION_PRESET["embedding_dim"]
    )
    config.multimodal_train.fusion_hidden_dim = int(
        COMPLETE_ONLY_REGRESSION_PRESET["fusion_hidden_dim"]
    )
    config.multimodal_train.learning_rate = float(
        COMPLETE_ONLY_REGRESSION_PRESET["learning_rate"]
    )
    config.multimodal_train.weight_decay = float(
        COMPLETE_ONLY_REGRESSION_PRESET["weight_decay"]
    )


def _default_study_name(config: RunConfig, strategy: str, group_scheme: str) -> str:
    return f"{config.run_id}_{group_scheme}_{strategy}_ablation"


def _study_directory(config: RunConfig, study_name: str) -> Path:
    return config.artifact_root / "ablations" / study_name


def _sanitize_slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", str(value).strip().lower()).strip("_")
    return slug or "subset"


def _resolve_modality_groups(group_scheme: str) -> dict[str, tuple[str, ...]]:
    if group_scheme == "device":
        return DEVICE_MODALITY_GROUPS
    if group_scheme == "modality":
        return {modality: (modality,) for modality in MODELED_MODALITIES}
    raise ValueError(f"Unsupported ablation group scheme: {group_scheme}")


def _modalities_for_group_subset(
    groups: dict[str, tuple[str, ...]],
    selected_group_names: tuple[str, ...],
) -> tuple[str, ...]:
    selected_modalities = {
        modality
        for group_name in selected_group_names
        for modality in groups[group_name]
    }
    return tuple(
        modality for modality in MODELED_MODALITIES if modality in selected_modalities
    )


def _summarize_subset_run(outputs: dict[str, Path]) -> dict[str, float]:
    fold_metrics_df = read_table_with_fallback(Path(outputs["metrics_per_fold"]))
    overall_metrics_df = read_table_with_fallback(Path(outputs["metrics_overall"]))
    binary_overall_df = read_table_with_fallback(Path(outputs["metrics_binary_overall"]))
    if fold_metrics_df.empty:
        raise ValueError("Ablation run produced no per-fold metrics.")
    if overall_metrics_df.empty:
        raise ValueError("Ablation run produced no overall metrics.")
    if binary_overall_df.empty:
        raise ValueError("Ablation run produced no overall binary metrics.")

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


def _candidate_sort_key(row: dict[str, Any]) -> tuple[float, float, float, float, float]:
    return (
        float(row["mean_fold_mae"]),
        -float(row["overall_within_1_accuracy"]),
        -float(row["overall_binary_macro_f1"]),
        float(row["overall_class_mae"]),
        -float(row["overall_macro_f1"]),
    )


def _evaluate_subset(
    *,
    base_config: RunConfig,
    groups: dict[str, tuple[str, ...]],
    selected_group_names: tuple[str, ...],
    step_idx: int,
    candidate_label: str,
) -> dict[str, Any]:
    subset_config = copy.deepcopy(base_config)
    selected_modalities = _modalities_for_group_subset(groups, selected_group_names)
    subset_config.multimodal_train.selected_modeled_modalities = selected_modalities
    subset_config.run_id = (
        f"{base_config.run_id}__ablation__step_{step_idx:02d}__{_sanitize_slug(candidate_label)}"
    )
    subset_config.ensure_directories()
    persist_run_metadata(subset_config)

    _, outputs, stats = run_phase_e_multimodal(subset_config)
    metrics = _summarize_subset_run(outputs)
    return {
        "run_id": subset_config.run_id,
        "step_idx": int(step_idx),
        "candidate_label": str(candidate_label),
        "selected_groups": tuple(selected_group_names),
        "selected_modalities": tuple(selected_modalities),
        "num_groups": int(len(selected_group_names)),
        "num_modalities": int(len(selected_modalities)),
        "split_mode": str(subset_config.multimodal_train.split_mode),
        "complete_windows_only": bool(subset_config.multimodal_train.complete_windows_only),
        "outputs_summary": str(outputs["summary"]),
        "num_prediction_rows": int(stats["num_prediction_rows"]),
        **metrics,
    }


def _build_worker_slots(
    gpu_indices: list[int],
    *,
    workers_per_gpu: int,
) -> list[dict[str, int]]:
    if int(workers_per_gpu) <= 0:
        raise ValueError("ablation workers_per_gpu must be positive.")
    slots: list[dict[str, int]] = []
    for gpu_index in gpu_indices:
        for slot_idx in range(int(workers_per_gpu)):
            slots.append({"gpu_index": int(gpu_index), "slot_idx": int(slot_idx)})
    return slots


def _ablation_worker_entry(
    *,
    config: RunConfig,
    groups: dict[str, tuple[str, ...]],
    selected_group_names: tuple[str, ...],
    step_idx: int,
    candidate_label: str,
    candidate_metadata: dict[str, Any],
    gpu_index: int,
    slot_idx: int,
    result_queue: Any,
) -> None:
    try:
        row = _evaluate_subset(
            base_config=config,
            groups=groups,
            selected_group_names=selected_group_names,
            step_idx=step_idx,
            candidate_label=candidate_label,
        )
        row.update(
            {
                "gpu_index": int(gpu_index),
                "worker_slot": int(slot_idx),
                **candidate_metadata,
            }
        )
        result_queue.put({"ok": True, "row": row})
    except Exception as exc:  # noqa: BLE001
        result_queue.put(
            {
                "ok": False,
                "gpu_index": int(gpu_index),
                "worker_slot": int(slot_idx),
                "candidate_label": str(candidate_label),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
        raise


def _evaluate_candidate_specs(
    *,
    base_config: RunConfig,
    groups: dict[str, tuple[str, ...]],
    candidate_specs: list[dict[str, Any]],
    workers_per_gpu: int,
    gpu_list: str | None,
) -> list[dict[str, Any]]:
    if not candidate_specs:
        return []

    gpu_indices = _parse_gpu_indices(gpu_list)
    if not gpu_indices:
        rows: list[dict[str, Any]] = []
        for candidate_spec in candidate_specs:
            row = _evaluate_subset(
                base_config=base_config,
                groups=groups,
                selected_group_names=tuple(candidate_spec["selected_group_names"]),
                step_idx=int(candidate_spec["step_idx"]),
                candidate_label=str(candidate_spec["candidate_label"]),
            )
            row.update(
                {
                    "gpu_index": np.nan,
                    "worker_slot": np.nan,
                    **candidate_spec["candidate_metadata"],
                }
            )
            rows.append(row)
        return rows

    worker_slots = _build_worker_slots(gpu_indices, workers_per_gpu=workers_per_gpu)
    if len(worker_slots) == 1:
        rows = []
        slot = worker_slots[0]
        for candidate_spec in candidate_specs:
            candidate_config = copy.deepcopy(base_config)
            candidate_config.multimodal_train.device = "cuda"
            candidate_config.multimodal_train.use_multi_gpu = False
            candidate_config.multimodal_train.cuda_device_index = int(slot["gpu_index"])
            row = _evaluate_subset(
                base_config=candidate_config,
                groups=groups,
                selected_group_names=tuple(candidate_spec["selected_group_names"]),
                step_idx=int(candidate_spec["step_idx"]),
                candidate_label=str(candidate_spec["candidate_label"]),
            )
            row.update(
                {
                    "gpu_index": int(slot["gpu_index"]),
                    "worker_slot": int(slot["slot_idx"]),
                    **candidate_spec["candidate_metadata"],
                }
            )
            rows.append(row)
        return rows

    ctx = mp.get_context("spawn")
    rows = []
    for chunk_start in range(0, len(candidate_specs), len(worker_slots)):
        chunk = candidate_specs[chunk_start : chunk_start + len(worker_slots)]
        result_queue = ctx.Queue()
        processes: list[Any] = []
        for candidate_spec, slot in zip(chunk, worker_slots, strict=False):
            candidate_config = copy.deepcopy(base_config)
            candidate_config.multimodal_train.device = "cuda"
            candidate_config.multimodal_train.use_multi_gpu = False
            candidate_config.multimodal_train.cuda_device_index = int(slot["gpu_index"])
            process = ctx.Process(
                target=_ablation_worker_entry,
                kwargs={
                    "config": candidate_config,
                    "groups": groups,
                    "selected_group_names": tuple(candidate_spec["selected_group_names"]),
                    "step_idx": int(candidate_spec["step_idx"]),
                    "candidate_label": str(candidate_spec["candidate_label"]),
                    "candidate_metadata": dict(candidate_spec["candidate_metadata"]),
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
                "One or more modality-ablation workers failed.\n"
                f"First failure on gpu={first_failure.get('gpu_index')}, "
                f"slot={first_failure.get('worker_slot')}, "
                f"candidate={first_failure.get('candidate_label')}:\n"
                f"{first_failure.get('error')}\n"
                f"{first_failure.get('traceback', '')}"
            )
        rows.extend(result["row"] for result in chunk_results)
    return rows


def run_modality_ablation_greedy(
    config: RunConfig,
    *,
    strategy: str,
    study_name: str | None = None,
    group_scheme: str = "device",
    min_groups: int = 1,
    max_groups: int | None = None,
    workers_per_gpu: int = 1,
    gpu_list: str | None = None,
) -> tuple[dict[str, Path], dict[str, Any]]:
    config = copy.deepcopy(config)
    config.multimodal_train.use_multi_gpu = False
    if strategy not in {"forward", "backward"}:
        raise ValueError("strategy must be 'forward' or 'backward'.")

    groups = _resolve_modality_groups(group_scheme)
    group_names = tuple(groups.keys())
    if not group_names:
        raise ValueError("No modality groups were defined for the ablation study.")

    resolved_min_groups = max(1, min(int(min_groups), len(group_names)))
    resolved_max_groups = (
        len(group_names)
        if max_groups is None
        else max(1, min(int(max_groups), len(group_names)))
    )
    if resolved_min_groups > resolved_max_groups:
        raise ValueError("ablation min_groups cannot exceed max_groups.")

    resolved_study_name = str(study_name or _default_study_name(config, strategy, group_scheme))
    study_dir = _study_directory(config, resolved_study_name)
    study_dir.mkdir(parents=True, exist_ok=True)

    candidate_rows: list[dict[str, Any]] = []
    chosen_rows: list[dict[str, Any]] = []

    if strategy == "backward":
        current_groups = tuple(group_names)
        full_row = _evaluate_candidate_specs(
            base_config=config,
            groups=groups,
            candidate_specs=[
                {
                    "selected_group_names": current_groups,
                    "step_idx": 0,
                    "candidate_label": "all_groups",
                    "candidate_metadata": {
                        "candidate_operation": "baseline",
                        "candidate_group": None,
                        "selected": True,
                    },
                }
            ],
            workers_per_gpu=workers_per_gpu,
            gpu_list=gpu_list,
        )[0]
        full_row["candidate_operation"] = "baseline"
        full_row["selected"] = True
        candidate_rows.append(full_row)
        chosen_rows.append(dict(full_row))

        step_idx = 1
        while len(current_groups) > resolved_min_groups:
            candidate_specs: list[dict[str, Any]] = []
            for removed_group in current_groups:
                selected_group_names = tuple(
                    group_name for group_name in current_groups if group_name != removed_group
                )
                if len(selected_group_names) < resolved_min_groups:
                    continue
                candidate_specs.append(
                    {
                        "selected_group_names": selected_group_names,
                        "step_idx": int(step_idx),
                        "candidate_label": f"drop_{removed_group}",
                        "candidate_metadata": {
                            "candidate_operation": "remove",
                            "candidate_group": removed_group,
                            "selected": False,
                        },
                    }
                )
            candidate_pool = _evaluate_candidate_specs(
                base_config=config,
                groups=groups,
                candidate_specs=candidate_specs,
                workers_per_gpu=workers_per_gpu,
                gpu_list=gpu_list,
            )
            if not candidate_pool:
                break
            best_row = min(candidate_pool, key=_candidate_sort_key)
            for row in candidate_pool:
                if row["run_id"] == best_row["run_id"]:
                    row["selected"] = True
                candidate_rows.append(row)
            chosen_rows.append(dict(best_row))
            current_groups = tuple(best_row["selected_groups"])
            step_idx += 1
            if len(current_groups) <= resolved_min_groups:
                break
    else:
        current_groups: tuple[str, ...] = ()
        step_idx = 1
        while len(current_groups) < resolved_max_groups:
            remaining_groups = tuple(
                group_name for group_name in group_names if group_name not in set(current_groups)
            )
            candidate_specs: list[dict[str, Any]] = []
            for added_group in remaining_groups:
                selected_group_names = tuple([*current_groups, added_group])
                if len(selected_group_names) > resolved_max_groups:
                    continue
                candidate_specs.append(
                    {
                        "selected_group_names": selected_group_names,
                        "step_idx": int(step_idx),
                        "candidate_label": f"add_{added_group}",
                        "candidate_metadata": {
                            "candidate_operation": "add",
                            "candidate_group": added_group,
                            "selected": False,
                        },
                    }
                )
            candidate_pool = _evaluate_candidate_specs(
                base_config=config,
                groups=groups,
                candidate_specs=candidate_specs,
                workers_per_gpu=workers_per_gpu,
                gpu_list=gpu_list,
            )
            if not candidate_pool:
                break
            best_row = min(candidate_pool, key=_candidate_sort_key)
            for row in candidate_pool:
                if row["run_id"] == best_row["run_id"]:
                    row["selected"] = True
                candidate_rows.append(row)
            if len(best_row["selected_groups"]) >= resolved_min_groups:
                chosen_rows.append(dict(best_row))
            current_groups = tuple(best_row["selected_groups"])
            step_idx += 1
            if len(current_groups) >= resolved_max_groups:
                break

    if not chosen_rows:
        raise RuntimeError("Modality ablation did not produce any selected subsets.")

    candidate_df = pd.DataFrame(
        [
            {
                **row,
                "selected_groups": json.dumps(list(row["selected_groups"])),
                "selected_modalities": json.dumps(list(row["selected_modalities"])),
            }
            for row in candidate_rows
        ]
    )
    chosen_df = pd.DataFrame(
        [
            {
                **row,
                "selected_groups": json.dumps(list(row["selected_groups"])),
                "selected_modalities": json.dumps(list(row["selected_modalities"])),
            }
            for row in chosen_rows
        ]
    )

    candidates_path = write_table_with_fallback(
        candidate_df,
        study_dir / "candidates.parquet",
        logger=LOGGER,
    )
    selected_path = write_table_with_fallback(
        chosen_df,
        study_dir / "selected_path.parquet",
        logger=LOGGER,
    )

    best_row = min(chosen_rows, key=_candidate_sort_key)
    summary_payload = {
        "study_name": resolved_study_name,
        "strategy": strategy,
        "group_scheme": group_scheme,
        "workers_per_gpu": int(workers_per_gpu),
        "gpu_list": None if gpu_list is None else str(gpu_list),
        "groups": {key: list(value) for key, value in groups.items()},
        "preset": COMPLETE_ONLY_REGRESSION_PRESET,
        "min_groups": int(resolved_min_groups),
        "max_groups": int(resolved_max_groups),
        "num_candidates": int(len(candidate_rows)),
        "num_selected_steps": int(len(chosen_rows)),
        "best_run_id": str(best_row["run_id"]),
        "best_groups": list(best_row["selected_groups"]),
        "best_modalities": list(best_row["selected_modalities"]),
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
        "ablation_candidates": candidates_path,
        "ablation_selected_path": selected_path,
        "ablation_summary": summary_path,
    }
    stats = {
        "study_name": resolved_study_name,
        "strategy": strategy,
        "group_scheme": group_scheme,
        "workers_per_gpu": int(workers_per_gpu),
        "gpu_list": None if gpu_list is None else str(gpu_list),
        "num_candidates": int(len(candidate_rows)),
        "num_selected_steps": int(len(chosen_rows)),
        "best_run_id": str(best_row["run_id"]),
        "best_groups": list(best_row["selected_groups"]),
        "best_mean_fold_mae": float(best_row["mean_fold_mae"]),
        "best_overall_binary_macro_f1": float(best_row["overall_binary_macro_f1"]),
    }
    return outputs, stats
