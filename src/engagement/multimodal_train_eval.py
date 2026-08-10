from __future__ import annotations

from dataclasses import dataclass
import hashlib
from itertools import combinations
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import torch
from torch import nn

from .config import RunConfig
from .io_utils import read_table_with_fallback, write_table_with_fallback
from .multimodal_dataset import (
    MODELED_MODALITIES,
    WindowTensorSample,
    build_window_tensor_samples,
    modality_input_dims,
)
from .multimodal_model import CORALLoss, MultimodalGatedFusionModel
from .multimodal_model import coral_predict

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TensorizedWindowSet:
    modality_arrays: dict[str, np.ndarray]
    modality_time_masks: dict[str, np.ndarray]
    modality_mask: np.ndarray
    context: np.ndarray
    labels_5class: np.ndarray
    labels_binary: np.ndarray
    meta: pd.DataFrame


@dataclass(frozen=True)
class PhaseEDirectories:
    phase_dir: Path
    checkpoints_dir: Path
    confusion_dir: Path


@dataclass(frozen=True)
class ParticipantFold:
    fold_id: int
    test_participants: tuple[int, ...]
    train_participants: tuple[int, ...]
    test_indices: np.ndarray
    train_indices: np.ndarray
    test_label_counts: np.ndarray


@dataclass(frozen=True)
class ValidationSplit:
    fit_indices: np.ndarray
    validation_indices: np.ndarray


@dataclass(frozen=True)
class ModalityNormalizationStats:
    mean: dict[str, np.ndarray]
    std: dict[str, np.ndarray]


EXPERIMENT_NAME = "multimodal_gated_context"
DATASET_CACHE_SCHEMA_VERSION = 2
BINARY_LABEL_SCHEMA = "12_vs_345"


def _split_artifact_key(split_mode: str) -> str:
    normalized = str(split_mode).strip().lower()
    if normalized == "loso":
        return "loso_splits"
    if normalized == "fixed_groups":
        return "fixed_group_splits"
    if normalized == "grouped_kfold":
        return "grouped_kfold_splits"
    safe = "".join(ch if ch.isalnum() else "_" for ch in normalized).strip("_")
    return f"{safe or 'evaluation'}_splits"


def _active_modeled_modalities(config: RunConfig) -> tuple[str, ...]:
    selected = tuple(
        str(modality).strip()
        for modality in config.multimodal_train.selected_modeled_modalities
        if str(modality).strip()
    )
    if not selected:
        return MODELED_MODALITIES
    selected_set = set(selected)
    return tuple(modality for modality in MODELED_MODALITIES if modality in selected_set)


def _complete_window_required_modalities(
    config: RunConfig,
    *,
    active_modalities: tuple[str, ...],
) -> tuple[str, ...]:
    reference = str(config.multimodal_train.complete_windows_reference)
    if reference == "all":
        return MODELED_MODALITIES
    return active_modalities


def _ordinal_zero_based_to_binary(labels_zero_based: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels_zero_based, dtype=np.int64).reshape(-1)
    return (labels >= 2).astype(int)


def _read_session_manifest(config: RunConfig) -> pd.DataFrame:
    run_path = config.run_dir / "manifests" / "session_manifest.parquet"
    df = read_table_with_fallback(run_path)
    if not df.empty:
        return df
    return read_table_with_fallback(config.manifests_dir / "session_manifest.parquet")


def _read_window_index(config: RunConfig, window_index: pd.DataFrame | None) -> pd.DataFrame:
    if window_index is not None:
        return window_index.copy()
    run_path = config.run_dir / "windows" / "window_index.parquet"
    df = read_table_with_fallback(run_path)
    if not df.empty:
        return df
    return read_table_with_fallback(config.windows_dir / "window_index.parquet")


def _tensorize_samples(samples: list[WindowTensorSample]) -> TensorizedWindowSet:
    modality_dims = modality_input_dims()
    if not samples:
        empty_meta = pd.DataFrame(
            columns=[
                "window_id",
                "session_key",
                "participant_id",
                "event_id",
                "video_uid",
                "label_5class",
            ]
        )
        empty_arrays = {
            modality: np.empty((0, modality_dims[modality], 0), dtype=np.float32)
            for modality in MODELED_MODALITIES
        }
        empty_time_masks = {
            modality: np.empty((0, 0), dtype=np.float32)
            for modality in MODELED_MODALITIES
        }
        return TensorizedWindowSet(
            modality_arrays=empty_arrays,
            modality_time_masks=empty_time_masks,
            modality_mask=np.empty((0, len(MODELED_MODALITIES)), dtype=np.float32),
            context=np.empty((0, 4), dtype=np.float32),
            labels_5class=np.empty((0,), dtype=np.int64),
            labels_binary=np.empty((0,), dtype=np.int64),
            meta=empty_meta,
        )

    max_lengths = {
        modality: max(
            (sample.modality_arrays[modality].shape[1] for sample in samples if modality in sample.modality_arrays),
            default=0,
        )
        for modality in MODELED_MODALITIES
    }
    modality_arrays = {
        modality: np.zeros(
            (len(samples), modality_dims[modality], max_lengths[modality]),
            dtype=np.float32,
        )
        for modality in MODELED_MODALITIES
    }
    modality_time_masks = {
        modality: np.zeros((len(samples), max_lengths[modality]), dtype=np.float32)
        for modality in MODELED_MODALITIES
    }
    modality_mask = np.zeros((len(samples), len(MODELED_MODALITIES)), dtype=np.float32)
    context = np.zeros((len(samples), len(samples[0].context_vector)), dtype=np.float32)
    labels_5class = np.zeros((len(samples),), dtype=np.int64)
    labels_binary = np.zeros((len(samples),), dtype=np.int64)
    meta_rows: list[dict[str, Any]] = []

    for idx, sample in enumerate(samples):
        modality_mask[idx] = sample.modality_mask
        context[idx] = sample.context_vector
        labels_5class[idx] = int(sample.label_5class - 1)
        labels_binary[idx] = int(sample.label_binary)
        meta_rows.append(
            {
                "window_id": sample.window_id,
                "session_key": sample.session_key,
                "participant_id": sample.participant_id,
                "event_id": sample.event_id,
                "video_uid": sample.video_uid,
                "label_5class": sample.label_5class,
            }
        )
        for modality, array in sample.modality_arrays.items():
            length = int(array.shape[1])
            modality_arrays[modality][idx, :, :length] = array
            modality_time_masks[modality][idx, :length] = 1.0

    return TensorizedWindowSet(
        modality_arrays=modality_arrays,
        modality_time_masks=modality_time_masks,
        modality_mask=modality_mask,
        context=context,
        labels_5class=labels_5class,
        labels_binary=labels_binary,
        meta=pd.DataFrame(meta_rows),
    )


def _subset_tensorized(data: TensorizedWindowSet, indices: np.ndarray) -> TensorizedWindowSet:
    modality_arrays = {modality: array[indices] for modality, array in data.modality_arrays.items()}
    modality_time_masks = {
        modality: array[indices] for modality, array in data.modality_time_masks.items()
    }
    return TensorizedWindowSet(
        modality_arrays=modality_arrays,
        modality_time_masks=modality_time_masks,
        modality_mask=data.modality_mask[indices],
        context=data.context[indices],
        labels_5class=data.labels_5class[indices],
        labels_binary=data.labels_binary[indices],
        meta=data.meta.iloc[indices].reset_index(drop=True),
    )


def _build_stratified_validation_split(
    data: TensorizedWindowSet,
    *,
    validation_fraction: float,
    seed: int,
) -> ValidationSplit:
    """Carve a deterministic, label-stratified validation set from an outer train fold."""

    num_windows = int(len(data.labels_5class))
    if num_windows < 2:
        raise ValueError("At least two outer-training windows are required for validation.")

    target_validation_windows = max(
        1,
        min(num_windows - 1, int(round(float(validation_fraction) * num_windows))),
    )
    all_indices = np.arange(num_windows, dtype=np.int64)
    labels = np.asarray(data.labels_5class, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    class_ids, class_counts = np.unique(labels, return_counts=True)
    quotas = class_counts.astype(np.float64) * target_validation_windows / num_windows
    allocations = np.floor(quotas).astype(int)
    # Keep at least one fit example for every class that has more than one example.
    capacities = np.where(class_counts > 1, class_counts - 1, class_counts)
    allocations = np.minimum(allocations, capacities)
    tie_breakers = rng.random(len(class_ids))

    while int(allocations.sum()) < target_validation_windows:
        available = np.flatnonzero(allocations < capacities)
        if len(available) == 0:
            break
        best_position = min(
            available.tolist(),
            key=lambda position: (
                -(float(quotas[position]) - float(allocations[position])),
                float(tie_breakers[position]),
            ),
        )
        allocations[best_position] += 1

    validation_parts = []
    for class_id, allocation in zip(class_ids.tolist(), allocations.tolist()):
        class_indices = all_indices[labels == int(class_id)]
        validation_parts.append(rng.permutation(class_indices)[: int(allocation)])
    validation_indices = np.sort(np.concatenate(validation_parts)).astype(np.int64)
    validation_mask = np.zeros(num_windows, dtype=bool)
    validation_mask[validation_indices] = True
    return ValidationSplit(
        fit_indices=all_indices[~validation_mask],
        validation_indices=validation_indices,
    )


def _restrict_to_selected_modalities(
    data: TensorizedWindowSet,
    *,
    selected_modalities: tuple[str, ...],
) -> TensorizedWindowSet:
    if not selected_modalities or tuple(selected_modalities) == MODELED_MODALITIES:
        return data

    selected_set = set(selected_modalities)
    modality_arrays = {
        modality: np.asarray(array, dtype=np.float32).copy()
        for modality, array in data.modality_arrays.items()
    }
    modality_time_masks = {
        modality: np.asarray(array, dtype=np.float32).copy()
        for modality, array in data.modality_time_masks.items()
    }
    modality_mask = np.asarray(data.modality_mask, dtype=np.float32).copy()

    for modality_idx, modality in enumerate(MODELED_MODALITIES):
        if modality in selected_set:
            continue
        modality_arrays[modality][:] = 0.0
        modality_time_masks[modality][:] = 0.0
        modality_mask[:, modality_idx] = 0.0

    return TensorizedWindowSet(
        modality_arrays=modality_arrays,
        modality_time_masks=modality_time_masks,
        modality_mask=modality_mask,
        context=data.context.copy(),
        labels_5class=data.labels_5class.copy(),
        labels_binary=data.labels_binary.copy(),
        meta=data.meta.copy(),
    )


def _disable_context_features(data: TensorizedWindowSet) -> TensorizedWindowSet:
    return TensorizedWindowSet(
        modality_arrays={modality: array.copy() for modality, array in data.modality_arrays.items()},
        modality_time_masks={modality: array.copy() for modality, array in data.modality_time_masks.items()},
        modality_mask=data.modality_mask.copy(),
        context=np.zeros_like(data.context, dtype=np.float32),
        labels_5class=data.labels_5class.copy(),
        labels_binary=data.labels_binary.copy(),
        meta=data.meta.copy(),
    )


def _filter_complete_windows(
    data: TensorizedWindowSet,
    *,
    required_modalities: tuple[str, ...] | None = None,
) -> TensorizedWindowSet:
    if required_modalities:
        required_indices = [
            MODELED_MODALITIES.index(modality)
            for modality in required_modalities
        ]
    else:
        required_indices = list(range(len(MODELED_MODALITIES)))
    complete_mask = np.all(
        np.asarray(data.modality_mask, dtype=np.float32)[:, required_indices] > 0.0,
        axis=1,
    )
    indices = np.flatnonzero(complete_mask)
    return _subset_tensorized(data, indices)


def _fit_modality_normalization(train_set: TensorizedWindowSet) -> ModalityNormalizationStats:
    means: dict[str, np.ndarray] = {}
    stds: dict[str, np.ndarray] = {}
    for modality in MODELED_MODALITIES:
        array = np.asarray(train_set.modality_arrays[modality], dtype=np.float32)
        if array.size == 0:
            means[modality] = np.zeros((array.shape[1],), dtype=np.float32)
            stds[modality] = np.ones((array.shape[1],), dtype=np.float32)
            continue

        time_mask = np.asarray(train_set.modality_time_masks[modality], dtype=np.float32)
        modality_idx = MODELED_MODALITIES.index(modality)
        modality_present = np.asarray(train_set.modality_mask[:, modality_idx], dtype=np.float32)
        valid = time_mask[:, None, :] * modality_present[:, None, None]
        valid_count = np.sum(valid, axis=(0, 2), dtype=np.float64)
        safe_count = np.maximum(valid_count, 1.0)

        mean = np.sum(array * valid, axis=(0, 2), dtype=np.float64) / safe_count
        centered = array - mean[None, :, None]
        var = np.sum((centered * centered) * valid, axis=(0, 2), dtype=np.float64) / safe_count
        std = np.sqrt(np.maximum(var, 1e-6))

        mean = np.where(valid_count > 0.0, mean, 0.0).astype(np.float32)
        std = np.where(valid_count > 0.0, std, 1.0).astype(np.float32)
        std = np.where(std < 1e-3, 1.0, std).astype(np.float32)

        means[modality] = mean
        stds[modality] = std

    return ModalityNormalizationStats(mean=means, std=stds)


def _apply_modality_normalization(
    data: TensorizedWindowSet,
    stats: ModalityNormalizationStats,
) -> TensorizedWindowSet:
    modality_arrays: dict[str, np.ndarray] = {}
    for modality in MODELED_MODALITIES:
        array = np.asarray(data.modality_arrays[modality], dtype=np.float32)
        if array.size == 0:
            modality_arrays[modality] = array.copy()
            continue
        time_mask = np.asarray(data.modality_time_masks[modality], dtype=np.float32)
        mean = stats.mean[modality][None, :, None]
        std = stats.std[modality][None, :, None]
        normalized = (array - mean) / std
        normalized *= time_mask[:, None, :]
        modality_arrays[modality] = normalized.astype(np.float32, copy=False)

    return TensorizedWindowSet(
        modality_arrays=modality_arrays,
        modality_time_masks={modality: array.copy() for modality, array in data.modality_time_masks.items()},
        modality_mask=data.modality_mask.copy(),
        context=data.context.copy(),
        labels_5class=data.labels_5class.copy(),
        labels_binary=data.labels_binary.copy(),
        meta=data.meta.copy(),
    )


def _apply_modality_dropout(
    data: TensorizedWindowSet,
    *,
    dropout_prob: float,
    rng: np.random.Generator,
) -> TensorizedWindowSet:
    dropout_prob = float(dropout_prob)
    if dropout_prob <= 0.0:
        return data

    modality_mask = np.asarray(data.modality_mask, dtype=np.float32).copy()
    modality_arrays = {
        modality: np.asarray(array, dtype=np.float32).copy()
        for modality, array in data.modality_arrays.items()
    }
    modality_time_masks = {
        modality: np.asarray(array, dtype=np.float32).copy()
        for modality, array in data.modality_time_masks.items()
    }

    for sample_idx in range(modality_mask.shape[0]):
        available_indices = np.flatnonzero(modality_mask[sample_idx] > 0.0)
        if len(available_indices) <= 1:
            continue
        drop_flags = rng.random(len(available_indices)) < dropout_prob
        if np.all(drop_flags):
            keep_idx = int(rng.integers(0, len(available_indices)))
            drop_flags[keep_idx] = False
        dropped_indices = available_indices[drop_flags]
        if len(dropped_indices) == 0:
            continue
        modality_mask[sample_idx, dropped_indices] = 0.0
        for dropped_idx in dropped_indices.tolist():
            modality = MODELED_MODALITIES[int(dropped_idx)]
            modality_arrays[modality][sample_idx, :, :] = 0.0
            modality_time_masks[modality][sample_idx, :] = 0.0

    return TensorizedWindowSet(
        modality_arrays=modality_arrays,
        modality_time_masks=modality_time_masks,
        modality_mask=modality_mask,
        context=data.context.copy(),
        labels_5class=data.labels_5class.copy(),
        labels_binary=data.labels_binary.copy(),
        meta=data.meta.copy(),
    )


def _to_device_batch(data: TensorizedWindowSet, device: torch.device) -> dict[str, Any]:
    return {
        "modality_inputs": {
            modality: torch.from_numpy(array).to(device=device, dtype=torch.float32)
            for modality, array in data.modality_arrays.items()
        },
        "modality_time_masks": {
            modality: torch.from_numpy(array).to(device=device, dtype=torch.float32)
            for modality, array in data.modality_time_masks.items()
        },
        "modality_mask": torch.from_numpy(data.modality_mask).to(device=device, dtype=torch.float32),
        "context": torch.from_numpy(data.context).to(device=device, dtype=torch.float32),
        "labels_5class": torch.from_numpy(data.labels_5class).to(device=device, dtype=torch.long),
        "labels_binary": torch.from_numpy(data.labels_binary).to(device=device, dtype=torch.float32),
    }


def _iter_minibatches(
    size: int,
    batch_size: int,
    rng: np.random.Generator,
    *,
    sample_weights: np.ndarray | None = None,
) -> list[np.ndarray]:
    if sample_weights is None:
        order = rng.permutation(size)
    else:
        weights = np.asarray(sample_weights, dtype=np.float64).reshape(-1)
        if weights.shape[0] != size:
            raise ValueError("sample_weights must have length equal to size.")
        weights = np.clip(weights, a_min=0.0, a_max=None)
        if not np.any(weights > 0):
            order = rng.permutation(size)
        else:
            probs = weights / np.sum(weights)
            order = rng.choice(size, size=size, replace=True, p=probs)
    return [order[start : start + batch_size] for start in range(0, size, batch_size)]


def _ordinal_threshold_pos_weight(labels_5class_zero_based: np.ndarray, num_classes: int = 5) -> torch.Tensor:
    labels = np.asarray(labels_5class_zero_based, dtype=np.int64).reshape(-1)
    thresholds = np.arange(num_classes - 1, dtype=np.int64).reshape(1, -1)
    targets = (labels.reshape(-1, 1) > thresholds).astype(np.float32)
    positives = np.sum(targets, axis=0, dtype=np.float64)
    negatives = float(len(labels)) - positives
    pos_weight = np.ones((num_classes - 1,), dtype=np.float32)
    valid = positives > 0.0
    pos_weight[valid] = (negatives[valid] / positives[valid]).astype(np.float32)
    return torch.from_numpy(pos_weight.astype(np.float32))


def _ordinal_sample_weights(labels_5class_zero_based: np.ndarray, num_classes: int = 5) -> np.ndarray:
    labels = np.asarray(labels_5class_zero_based, dtype=np.int64).reshape(-1)
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    inv = np.zeros((num_classes,), dtype=np.float64)
    valid = counts > 0.0
    inv[valid] = 1.0 / counts[valid]
    weights = inv[labels]
    if not np.any(weights > 0.0):
        return np.ones((len(labels),), dtype=np.float64)
    return weights / np.mean(weights)


def _compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "binary_accuracy": float(accuracy_score(y_true, y_pred)),
        "binary_macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "binary_weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "num_samples": int(len(y_true)),
    }


def _compute_ordinal_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true_arr = np.asarray(y_true, dtype=np.float32)
    y_pred_arr = np.asarray(y_pred, dtype=np.float32)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "mae": float(np.mean(np.abs(y_true_arr - y_pred_arr))),
        "within_1_accuracy": float(np.mean(np.abs(y_true_arr - y_pred_arr) <= 1.0)),
        "num_samples": int(len(y_true)),
    }


def _compute_regression_mae(y_true: np.ndarray, y_pred_score: np.ndarray) -> float:
    y_true_float = np.asarray(y_true, dtype=np.float32).reshape(-1)
    y_pred_float = np.asarray(y_pred_score, dtype=np.float32).reshape(-1)
    return float(np.mean(np.abs(y_true_float - y_pred_float)))


def _evaluate_random_baseline(
    *,
    train_set: TensorizedWindowSet,
    test_set: TensorizedWindowSet,
    task_mode: str,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    y_pred_5class: np.ndarray | None = None
    ordinal_metrics: dict[str, float] | None = None

    if task_mode == "binary":
        binary_counts = np.bincount(train_set.labels_binary.astype(int), minlength=2).astype(np.float64)
        binary_probs = binary_counts / max(float(binary_counts.sum()), 1.0)
        y_pred_binary = rng.choice(2, size=len(test_set.labels_binary), p=binary_probs).astype(int)
    else:
        class_counts = np.bincount(train_set.labels_5class.astype(int), minlength=5).astype(np.float64)
        class_probs = class_counts / max(float(class_counts.sum()), 1.0)
        y_pred_5class = rng.choice(5, size=len(test_set.labels_5class), p=class_probs).astype(int)
        ordinal_metrics = _compute_ordinal_metrics(test_set.labels_5class, y_pred_5class)
        y_pred_binary = _ordinal_zero_based_to_binary(y_pred_5class)

    binary_metrics = _compute_binary_metrics(test_set.labels_binary, y_pred_binary)
    return {
        "y_pred_5class": y_pred_5class,
        "y_pred_binary": y_pred_binary,
        "ordinal_metrics": ordinal_metrics,
        "binary_metrics": binary_metrics,
    }


def _evaluate_constant_baseline(
    *,
    train_set: TensorizedWindowSet,
    test_set: TensorizedWindowSet,
    task_mode: str,
    strategy: str,
) -> dict[str, Any]:
    if strategy not in {"mean", "median"}:
        raise ValueError(f"Unsupported constant baseline strategy: {strategy}")

    if task_mode == "binary":
        train_values = train_set.labels_binary.astype(np.float64)
        if strategy == "mean":
            constant_value = int(np.clip(np.round(float(np.mean(train_values))), 0, 1))
        else:
            constant_value = int(np.clip(np.round(float(np.median(train_values))), 0, 1))
        y_pred_binary = np.full((len(test_set.labels_binary),), constant_value, dtype=int)
        y_pred_5class: np.ndarray | None = None
        ordinal_metrics: dict[str, float] | None = None
    else:
        train_values = train_set.labels_5class.astype(np.float64)
        if strategy == "mean":
            constant_value = int(np.clip(np.round(float(np.mean(train_values))), 0, 4))
        else:
            constant_value = int(np.clip(np.round(float(np.median(train_values))), 0, 4))
        y_pred_5class = np.full((len(test_set.labels_5class),), constant_value, dtype=int)
        ordinal_metrics = _compute_ordinal_metrics(test_set.labels_5class, y_pred_5class)
        y_pred_binary = _ordinal_zero_based_to_binary(y_pred_5class)

    binary_metrics = _compute_binary_metrics(test_set.labels_binary, y_pred_binary)
    return {
        "y_pred_5class": y_pred_5class,
        "y_pred_binary": y_pred_binary,
        "ordinal_metrics": ordinal_metrics,
        "binary_metrics": binary_metrics,
    }


def _primary_metric_name(task_mode: str) -> str:
    return "binary_macro_f1" if task_mode == "binary" else "mae"


def _participant_label_counts(data: TensorizedWindowSet) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    participant_ids = sorted({int(v) for v in data.meta["participant_id"].tolist()})
    participant_series = data.meta["participant_id"].to_numpy(dtype=int)
    for participant_id in participant_ids:
        mask = participant_series == participant_id
        label_counts = np.bincount(data.labels_5class[mask], minlength=5)
        rows.append(
            {
                "participant_id": int(participant_id),
                "num_windows": int(mask.sum()),
                **{f"class_{class_id + 1}_count": int(label_counts[class_id]) for class_id in range(5)},
            }
        )
    return pd.DataFrame(rows)


def _assignment_objective(
    assignment: dict[int, int],
    participant_counts: dict[int, np.ndarray],
    participant_windows: dict[int, int],
    *,
    num_folds: int,
    global_label_counts: np.ndarray,
) -> tuple[float, float, float, float, float]:
    fold_counts = [np.zeros(5, dtype=np.int64) for _ in range(num_folds)]
    fold_windows = np.zeros(num_folds, dtype=np.int64)
    fold_sizes = np.zeros(num_folds, dtype=np.int64)

    for participant_id, fold_id in assignment.items():
        fold_counts[fold_id] += participant_counts[participant_id]
        fold_windows[fold_id] += int(participant_windows[participant_id])
        fold_sizes[fold_id] += 1

    missing_classes = float(sum(int(np.sum(counts == 0)) for counts in fold_counts))
    incomplete_folds = float(sum(int(np.any(counts == 0)) for counts in fold_counts))
    window_std = float(np.std(fold_windows.astype(np.float64)))
    size_std = float(np.std(fold_sizes.astype(np.float64)))

    total_global = float(np.sum(global_label_counts))
    global_ratio = (
        global_label_counts.astype(np.float64) / total_global
        if total_global > 0
        else np.zeros_like(global_label_counts, dtype=np.float64)
    )
    ratio_penalties: list[float] = []
    for counts in fold_counts:
        total = float(np.sum(counts))
        if total <= 0:
            ratio_penalties.append(float(np.sum(np.abs(global_ratio))))
            continue
        ratio = counts.astype(np.float64) / total
        ratio_penalties.append(float(np.sum(np.abs(ratio - global_ratio))))
    ratio_penalty = float(np.mean(ratio_penalties)) if ratio_penalties else 0.0

    return (
        missing_classes,
        incomplete_folds,
        ratio_penalty,
        window_std,
        size_std,
    )


def _target_fold_sizes(num_participants: int, num_folds: int) -> list[int]:
    num_folds = max(2, min(int(num_folds), int(num_participants)))
    base = int(num_participants) // int(num_folds)
    remainder = int(num_participants) % int(num_folds)
    return [base + (1 if fold_id < remainder else 0) for fold_id in range(num_folds)]


def _build_grouped_kfold_assignments(
    data: TensorizedWindowSet,
    *,
    num_folds: int,
    seed: int,
) -> list[tuple[int, ...]]:
    participant_summary = _participant_label_counts(data)
    participant_ids = [int(v) for v in participant_summary["participant_id"].tolist()]
    num_folds = max(2, min(int(num_folds), len(participant_ids)))
    participant_counts = {
        int(row["participant_id"]): np.asarray(
            [row[f"class_{class_id}_count"] for class_id in range(1, 6)],
            dtype=np.int64,
        )
        for row in participant_summary.to_dict(orient="records")
    }
    participant_windows = {
        int(row["participant_id"]): int(row["num_windows"])
        for row in participant_summary.to_dict(orient="records")
    }
    target_sizes = _target_fold_sizes(len(participant_ids), num_folds)
    global_label_counts = np.sum(
        np.stack([participant_counts[participant_id] for participant_id in participant_ids], axis=0),
        axis=0,
    )
    rarity_weights = 1.0 / np.maximum(global_label_counts.astype(np.float64), 1.0)
    target_windows_per_fold = float(sum(participant_windows.values())) / float(num_folds)
    global_ratio = global_label_counts.astype(np.float64) / max(
        float(np.sum(global_label_counts)),
        1.0,
    )

    if len(participant_ids) <= 20 and max(target_sizes) <= 5:
        ordered_participants = tuple(sorted(participant_ids))
        participant_to_bit = {
            participant_id: (1 << idx) for idx, participant_id in enumerate(ordered_participants)
        }
        full_mask = (1 << len(ordered_participants)) - 1

        candidate_groups_by_size: dict[int, dict[int, list[dict[str, Any]]]] = {}
        for group_size in sorted(set(target_sizes)):
            candidates_for_size: dict[int, list[dict[str, Any]]] = {
                participant_id: [] for participant_id in ordered_participants
            }
            for group in combinations(ordered_participants, group_size):
                counts = np.sum(
                    np.stack([participant_counts[participant_id] for participant_id in group], axis=0),
                    axis=0,
                )
                mask = 0
                for participant_id in group:
                    mask |= participant_to_bit[participant_id]
                ratio_penalty = float(
                    np.sum(
                        np.abs(
                            counts.astype(np.float64) / max(float(np.sum(counts)), 1.0)
                            - global_ratio
                        )
                    )
                )
                candidate = {
                    "group": tuple(int(v) for v in group),
                    "mask": int(mask),
                    "counts": counts,
                    "missing_classes": int(np.sum(counts == 0)),
                    "window_delta": abs(
                        float(sum(participant_windows[participant_id] for participant_id in group))
                        - target_windows_per_fold
                    ),
                    "ratio_penalty": ratio_penalty,
                }
                candidates_for_size[int(group[0])].append(candidate)
            for participant_id in candidates_for_size:
                candidates_for_size[participant_id].sort(
                    key=lambda candidate: (
                        candidate["missing_classes"],
                        candidate["ratio_penalty"],
                        candidate["window_delta"],
                    )
                )
            candidate_groups_by_size[group_size] = candidates_for_size

        def evaluate_partition(groups: list[dict[str, Any]]) -> tuple[float, float, float, float, float]:
            fold_counts = [group["counts"] for group in groups]
            fold_windows = np.asarray(
                [sum(participant_windows[participant_id] for participant_id in group["group"]) for group in groups],
                dtype=np.float64,
            )
            ratio_penalties = [float(group["ratio_penalty"]) for group in groups]
            return (
                float(sum(int(np.sum(counts == 0)) for counts in fold_counts)),
                float(sum(int(np.any(counts == 0)) for counts in fold_counts)),
                float(np.mean(ratio_penalties)) if ratio_penalties else 0.0,
                float(np.std(fold_windows)),
                0.0,
            )

        def search(require_complete_groups: bool) -> list[tuple[int, ...]] | None:
            best_partition: list[dict[str, Any]] | None = None
            best_objective: tuple[float, float, float, float, float] | None = None

            def recurse(remaining_mask: int, fold_id: int, chosen_groups: list[dict[str, Any]]) -> None:
                nonlocal best_partition, best_objective
                if fold_id == len(target_sizes):
                    if remaining_mask != 0:
                        return
                    objective = evaluate_partition(chosen_groups)
                    if best_objective is None or objective < best_objective:
                        best_objective = objective
                        best_partition = list(chosen_groups)
                    return

                remaining_group_sizes = target_sizes[fold_id:]
                remaining_participant_count = int(remaining_mask.bit_count())
                if remaining_participant_count != int(sum(remaining_group_sizes)):
                    return

                remaining_ids = [
                    participant_id
                    for participant_id in ordered_participants
                    if remaining_mask & participant_to_bit[participant_id]
                ]
                if not remaining_ids:
                    return
                anchor = int(remaining_ids[0])
                group_size = target_sizes[fold_id]
                candidates = candidate_groups_by_size[group_size][anchor]
                for candidate in candidates:
                    if require_complete_groups and int(candidate["missing_classes"]) > 0:
                        continue
                    candidate_mask = int(candidate["mask"])
                    if (candidate_mask & remaining_mask) != candidate_mask:
                        continue
                    partial_missing = sum(int(group["missing_classes"]) for group in chosen_groups) + int(candidate["missing_classes"])
                    if best_objective is not None and float(partial_missing) > float(best_objective[0]):
                        continue
                    recurse(
                        remaining_mask ^ candidate_mask,
                        fold_id + 1,
                        [*chosen_groups, candidate],
                    )

            recurse(full_mask, 0, [])
            if best_partition is None:
                return None
            return [tuple(sorted(group["group"])) for group in best_partition]

        exact_partition = search(require_complete_groups=True)
        if exact_partition is None:
            exact_partition = search(require_complete_groups=False)
        if exact_partition is not None:
            return exact_partition

    def participant_priority(participant_id: int, rng: np.random.Generator) -> tuple[float, float, float]:
        counts = participant_counts[participant_id]
        rarity_score = float(np.sum((counts > 0).astype(np.float64) * rarity_weights))
        class_count = int(np.sum(counts > 0))
        random_tiebreak = float(rng.uniform(0.0, 1.0))
        return (-rarity_score, float(class_count), -float(participant_windows[participant_id]) - random_tiebreak)

    def build_candidate_assignment(rng: np.random.Generator) -> dict[int, int]:
        assignment: dict[int, int] = {}
        fold_counts = [np.zeros(5, dtype=np.int64) for _ in range(num_folds)]
        fold_windows = np.zeros(num_folds, dtype=np.int64)
        fold_sizes = np.zeros(num_folds, dtype=np.int64)
        ordered_participants = sorted(
            participant_ids,
            key=lambda participant_id: participant_priority(participant_id, rng),
        )

        for fold_id, participant_id in enumerate(ordered_participants[:num_folds]):
            assignment[participant_id] = fold_id
            fold_counts[fold_id] += participant_counts[participant_id]
            fold_windows[fold_id] += participant_windows[participant_id]
            fold_sizes[fold_id] += 1

        for participant_id in ordered_participants[num_folds:]:
            counts = participant_counts[participant_id]
            best_score: tuple[float, float, float, float, float] | None = None
            best_fold_id = 0
            for fold_id in range(num_folds):
                projected_counts = fold_counts[fold_id] + counts
                missing_after = float(np.sum(projected_counts == 0))
                coverage_gain = float(
                    np.sum(
                        rarity_weights[
                            (fold_counts[fold_id] == 0) & (counts > 0)
                        ]
                    )
                )
                if fold_sizes[fold_id] + 1 > target_sizes[fold_id]:
                    continue
                projected_windows = float(fold_windows[fold_id] + participant_windows[participant_id])
                projected_size = float(fold_sizes[fold_id] + 1)
                projected_ratio = projected_counts.astype(np.float64) / max(
                    float(np.sum(projected_counts)),
                    1.0,
                )
                ratio_penalty = float(np.sum(np.abs(projected_ratio - global_ratio)))
                score = (
                    missing_after - coverage_gain,
                    abs(projected_size - float(target_sizes[fold_id])),
                    abs(projected_windows - target_windows_per_fold),
                    ratio_penalty,
                    float(fold_id),
                )
                if best_score is None or score < best_score:
                    best_score = score
                    best_fold_id = fold_id
            assignment[participant_id] = best_fold_id
            fold_counts[best_fold_id] += counts
            fold_windows[best_fold_id] += participant_windows[participant_id]
            fold_sizes[best_fold_id] += 1
        return assignment

    rng_master = np.random.default_rng(seed)
    best_assignment: dict[int, int] | None = None
    best_objective: tuple[float, float, float, float, float] | None = None
    num_restarts = max(64, num_folds * len(participant_ids) * 8)

    for _ in range(num_restarts):
        trial_seed = int(rng_master.integers(0, np.iinfo(np.int32).max))
        assignment = build_candidate_assignment(np.random.default_rng(trial_seed))
        objective = _assignment_objective(
            assignment,
            participant_counts,
            participant_windows,
            num_folds=num_folds,
            global_label_counts=global_label_counts,
        )
        if best_objective is None or objective < best_objective:
            best_assignment = assignment
            best_objective = objective
            if objective[0] <= 0.0 and objective[1] <= 0.0:
                break

    if best_assignment is None:
        raise RuntimeError("Failed to build grouped participant folds.")

    folds: list[tuple[int, ...]] = []
    for fold_id in range(num_folds):
        test_participants = sorted(
            participant_id
            for participant_id, assigned_fold_id in best_assignment.items()
            if assigned_fold_id == fold_id
        )
        if test_participants:
            folds.append(tuple(int(v) for v in test_participants))
    return folds


def _build_participant_folds(
    data: TensorizedWindowSet,
    *,
    config: RunConfig,
    split_mode: str,
    num_folds: int,
    seed: int,
) -> list[ParticipantFold]:
    participant_ids = sorted({int(v) for v in data.meta["participant_id"].tolist()})
    if len(participant_ids) < 2:
        raise ValueError("Grouped participant evaluation requires at least 2 participants.")

    if split_mode == "loso":
        fold_groups = [tuple([participant_id]) for participant_id in participant_ids]
    elif split_mode == "fixed_groups":
        participant_id_set = set(int(participant_id) for participant_id in participant_ids)
        configured_folds = [
            tuple(
                int(participant_id)
                for participant_id in fold
                if int(participant_id) in participant_id_set
            )
            for fold in config.multimodal_train.participant_folds
        ]
        missing_participants = sorted(
            participant_id
            for participant_id in participant_ids
            if all(participant_id not in fold for fold in configured_folds)
        )
        if missing_participants:
            raise ValueError(
                "Configured fixed participant folds do not cover all usable participants: "
                + ", ".join(str(participant_id) for participant_id in missing_participants)
            )
        fold_groups = configured_folds
    else:
        fold_groups = _build_grouped_kfold_assignments(
            data,
            num_folds=num_folds,
            seed=seed,
        )

    all_indices = np.arange(len(data.labels_5class))
    participant_array = data.meta["participant_id"].to_numpy(dtype=int)
    folds: list[ParticipantFold] = []
    for fold_id, test_participants in enumerate(fold_groups):
        test_mask = np.isin(participant_array, np.asarray(test_participants, dtype=int))
        test_indices = all_indices[test_mask]
        train_indices = all_indices[~test_mask]
        if len(test_indices) == 0 or len(train_indices) == 0:
            continue
        train_participants = tuple(
            sorted(int(v) for v in participant_ids if int(v) not in set(test_participants))
        )
        test_label_counts = np.bincount(data.labels_5class[test_indices], minlength=5).astype(np.int64)
        folds.append(
            ParticipantFold(
                fold_id=int(fold_id),
                test_participants=tuple(int(v) for v in test_participants),
                train_participants=train_participants,
                test_indices=test_indices,
                train_indices=train_indices,
                test_label_counts=test_label_counts,
            )
        )

    if len(folds) < 2:
        raise ValueError("Evaluation requires at least 2 non-empty participant folds.")
    return folds


def _aggregate_report_level(predictions_df: pd.DataFrame) -> pd.DataFrame:
    if predictions_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    group_cols = ["experiment"]
    if "baseline" in predictions_df.columns:
        group_cols.append("baseline")
    group_cols.extend(["fold_id", "test_group", "event_id"])
    for keys, group in predictions_df.groupby(group_cols, sort=False):
        baseline_name: str | None = None
        if "baseline" in predictions_df.columns:
            experiment, baseline_name, fold_id, test_group, event_id = keys
        else:
            experiment, fold_id, test_group, event_id = keys
        y_true = group["y_true"].mode().iloc[0]
        y_pred = group["y_pred"].dropna().mode().iloc[0] if group["y_pred"].notna().any() else np.nan
        y_true_binary = group["y_true_binary"].mode().iloc[0]
        y_pred_binary = group["y_pred_binary"].mode().iloc[0]
        row = {
            "experiment": str(experiment),
            "fold_id": int(fold_id),
            "test_group": str(test_group),
            "event_id": str(event_id),
            "y_true": int(y_true),
            "y_pred": int(y_pred) if pd.notna(y_pred) else np.nan,
            "y_true_binary": int(y_true_binary),
            "y_pred_binary": int(y_pred_binary),
            "num_windows": int(len(group)),
        }
        if "regression_score" in group.columns:
            valid_regression = pd.to_numeric(group["regression_score"], errors="coerce")
            row["regression_score"] = (
                float(valid_regression.mean())
                if valid_regression.notna().any()
                else np.nan
            )
        if baseline_name is not None:
            row["baseline"] = str(baseline_name)
        rows.append(row)
    return pd.DataFrame(rows)


def _prepare_directories(config: RunConfig) -> PhaseEDirectories:
    phase_dir = config.run_dir / "phase_e"
    checkpoints_dir = phase_dir / "checkpoints"
    confusion_dir = phase_dir / "confusion"
    phase_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    confusion_dir.mkdir(parents=True, exist_ok=True)
    return PhaseEDirectories(
        phase_dir=phase_dir,
        checkpoints_dir=checkpoints_dir,
        confusion_dir=confusion_dir,
    )


def _dataset_cache_root(config: RunConfig) -> Path:
    path = config.artifact_root / "cache" / "multimodal_phase_e"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _hash_dataframe(df: pd.DataFrame) -> str:
    normalized = df.copy()
    normalized = normalized.reindex(sorted(normalized.columns), axis=1)
    if "window_id" in normalized.columns:
        normalized = normalized.sort_values("window_id", kind="mergesort").reset_index(drop=True)
    elif "session_key" in normalized.columns:
        normalized = normalized.sort_values("session_key", kind="mergesort").reset_index(drop=True)
    else:
        normalized = normalized.reset_index(drop=True)
    normalized = normalized.fillna("__nan__")
    hashed = pd.util.hash_pandas_object(normalized.astype(str), index=False).to_numpy(dtype=np.uint64)
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def _dataset_cache_key(
    *,
    windows: pd.DataFrame,
    session_manifest: pd.DataFrame,
    config: RunConfig,
) -> str:
    payload = {
        "schema_version": DATASET_CACHE_SCHEMA_VERSION,
        "binary_label_schema": BINARY_LABEL_SCHEMA,
        "modeled_modalities": list(MODELED_MODALITIES),
        "windows_hash": _hash_dataframe(windows),
        "session_manifest_hash": _hash_dataframe(session_manifest),
        "use_native_frequency": bool(config.multimodal_train.use_native_frequency),
        "target_points": int(config.multimodal_train.target_points),
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


def _dataset_cache_paths(cache_root: Path, cache_key: str) -> dict[str, Path]:
    return {
        "arrays": cache_root / f"{cache_key}.npz",
        "meta": cache_root / f"{cache_key}_meta.parquet",
        "manifest": cache_root / f"{cache_key}_manifest.json",
    }


def _save_tensorized_cache(
    *,
    cache_paths: dict[str, Path],
    data: TensorizedWindowSet,
    manifest_payload: dict[str, Any],
) -> None:
    arrays_payload: dict[str, np.ndarray] = {
        "modality_mask": data.modality_mask,
        "context": data.context,
        "labels_5class": data.labels_5class,
        "labels_binary": data.labels_binary,
    }
    for modality in MODELED_MODALITIES:
        arrays_payload[f"array__{modality}"] = data.modality_arrays[modality]
        arrays_payload[f"time_mask__{modality}"] = data.modality_time_masks[modality]

    cache_paths["arrays"].parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_paths["arrays"], **arrays_payload)
    write_table_with_fallback(data.meta, cache_paths["meta"], logger=LOGGER)
    with cache_paths["manifest"].open("w", encoding="utf-8") as fp:
        json.dump(manifest_payload, fp, indent=2, sort_keys=True)


def _load_tensorized_cache(cache_paths: dict[str, Path]) -> TensorizedWindowSet | None:
    arrays_path = cache_paths["arrays"]
    meta_path = cache_paths["meta"]
    manifest_path = cache_paths["manifest"]
    if not arrays_path.exists() or not manifest_path.exists():
        return None
    meta_df = read_table_with_fallback(meta_path)
    if meta_df.empty and not meta_path.with_suffix(".csv").exists() and not meta_path.exists():
        return None

    with arrays_path.open("rb") as fp:
        loaded = np.load(fp, allow_pickle=False)
        modality_arrays = {
            modality: np.asarray(loaded[f"array__{modality}"], dtype=np.float32)
            for modality in MODELED_MODALITIES
        }
        modality_time_masks = {
            modality: np.asarray(loaded[f"time_mask__{modality}"], dtype=np.float32)
            for modality in MODELED_MODALITIES
        }
        modality_mask = np.asarray(loaded["modality_mask"], dtype=np.float32)
        context = np.asarray(loaded["context"], dtype=np.float32)
        labels_5class = np.asarray(loaded["labels_5class"], dtype=np.int64)
        labels_binary = np.asarray(loaded["labels_binary"], dtype=np.int64)

    return TensorizedWindowSet(
        modality_arrays=modality_arrays,
        modality_time_masks=modality_time_masks,
        modality_mask=modality_mask,
        context=context,
        labels_5class=labels_5class,
        labels_binary=labels_binary,
        meta=meta_df,
    )


def _build_or_load_tensorized_dataset(
    *,
    config: RunConfig,
    windows: pd.DataFrame,
    session_manifest: pd.DataFrame,
) -> TensorizedWindowSet:
    cache_root = _dataset_cache_root(config)
    cache_key = _dataset_cache_key(
        windows=windows,
        session_manifest=session_manifest,
        config=config,
    )
    cache_paths = _dataset_cache_paths(cache_root, cache_key)
    cached = _load_tensorized_cache(cache_paths)
    if cached is not None:
        LOGGER.info(
            "Phase E multimodal dataset cache hit: %s",
            cache_paths["arrays"],
        )
        return cached

    LOGGER.info("Phase E multimodal dataset cache miss: %s", cache_key)
    samples = build_window_tensor_samples(
        window_index=windows,
        session_manifest=session_manifest,
        use_native_frequency=bool(config.multimodal_train.use_native_frequency),
        target_points=int(config.multimodal_train.target_points),
    )
    if not samples:
        raise ValueError("No usable multimodal samples could be built for Phase E.")

    data = _tensorize_samples(samples)
    manifest_payload = {
        "schema_version": DATASET_CACHE_SCHEMA_VERSION,
        "binary_label_schema": BINARY_LABEL_SCHEMA,
        "cache_key": cache_key,
        "num_samples": int(len(data.meta)),
        "use_native_frequency": bool(config.multimodal_train.use_native_frequency),
        "target_points": int(config.multimodal_train.target_points),
    }
    _save_tensorized_cache(
        cache_paths=cache_paths,
        data=data,
        manifest_payload=manifest_payload,
    )
    LOGGER.info(
        "Phase E multimodal dataset cache saved: %s",
        cache_paths["arrays"],
    )
    return data


def _positive_class_weight_binary(labels_binary: np.ndarray) -> torch.Tensor:
    positives = float((labels_binary == 1).sum())
    negatives = float((labels_binary == 0).sum())
    if positives <= 0 or negatives <= 0:
        return torch.tensor(1.0, dtype=torch.float32)
    return torch.tensor(negatives / positives, dtype=torch.float32)


def _resolve_device(config: RunConfig) -> torch.device:
    preference = str(config.multimodal_train.device).lower()
    explicit_cuda_index = config.multimodal_train.cuda_device_index
    if preference == "cuda":
        if torch.cuda.is_available():
            if explicit_cuda_index is not None:
                if explicit_cuda_index >= _cuda_device_count():
                    raise RuntimeError(
                        "multimodal_train.cuda_device_index exceeds available CUDA devices."
                    )
                return torch.device(f"cuda:{explicit_cuda_index}")
            return torch.device("cuda")
        raise RuntimeError("multimodal_train.device='cuda' requested, but CUDA is unavailable.")
    if preference == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        raise RuntimeError("multimodal_train.device='mps' requested, but MPS is unavailable.")
    if preference == "cpu":
        return torch.device("cpu")

    if torch.cuda.is_available():
        if explicit_cuda_index is not None:
            if explicit_cuda_index >= _cuda_device_count():
                raise RuntimeError(
                    "multimodal_train.cuda_device_index exceeds available CUDA devices."
                )
            return torch.device(f"cuda:{explicit_cuda_index}")
        return torch.device("cuda")
    if explicit_cuda_index is not None:
        raise RuntimeError(
            "multimodal_train.cuda_device_index was set, but CUDA is unavailable."
        )
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _cuda_device_count() -> int:
    try:
        return int(torch.cuda.device_count())
    except Exception:
        return 0


def _unwrap_model(model: nn.Module) -> nn.Module:
    if isinstance(model, nn.DataParallel):
        return model.module
    return model


def _build_training_model(
    *,
    config: RunConfig,
    modality_dims: dict[str, int],
    context_dim: int,
    device: torch.device,
) -> tuple[nn.Module, int]:
    base_model = MultimodalGatedFusionModel(
        modality_input_dims=modality_dims,
        modality_order=MODELED_MODALITIES,
        context_dim=context_dim,
        embedding_dim=config.multimodal_train.embedding_dim,
        fusion_hidden_dim=config.multimodal_train.fusion_hidden_dim,
    ).to(device)

    num_cuda_devices = _cuda_device_count() if device.type == "cuda" else 0
    if (
        device.type == "cuda"
        and bool(config.multimodal_train.use_multi_gpu)
        and config.multimodal_train.cuda_device_index is None
        and num_cuda_devices > 1
    ):
        LOGGER.info(
            "Phase E multimodal enabling DataParallel across %s CUDA devices.",
            num_cuda_devices,
        )
        return nn.DataParallel(base_model), num_cuda_devices

    return base_model, max(1, num_cuda_devices) if device.type == "cuda" else 0


def _evaluate_model(
    model: MultimodalGatedFusionModel,
    data: TensorizedWindowSet,
    *,
    device: torch.device,
    task_mode: str,
    lambda_binary: float,
    lambda_ordinal: float,
    lambda_regression: float,
    ordinal_pos_weight: torch.Tensor | None = None,
) -> dict[str, Any]:
    base_model = _unwrap_model(model)
    model.eval()
    with torch.no_grad():
        tensors = _to_device_batch(data, device)
        outputs = model(
            modality_inputs=tensors["modality_inputs"],
            modality_time_masks=tensors["modality_time_masks"],
            modality_mask=tensors["modality_mask"],
            context_inputs=tensors["context"],
        )
        ordinal_proba = outputs["ordinal_proba"].detach().cpu().numpy()
        binary_proba = torch.sigmoid(outputs["binary_logits"]).detach().cpu().numpy()
        regression_score = outputs["regression_score"].detach().cpu().numpy()
        gates = outputs["gates"].detach().cpu().numpy()
        binary_pos_weight = _positive_class_weight_binary(data.labels_binary).to(device)
        bce_loss = nn.BCEWithLogitsLoss(pos_weight=binary_pos_weight)
        coral_loss = CORALLoss(
            pos_weight=ordinal_pos_weight.to(device) if ordinal_pos_weight is not None else None
        )
        regression_loss = nn.SmoothL1Loss()
        loss_bin = float(bce_loss(outputs["binary_logits"], tensors["labels_binary"]).detach().cpu())
        loss_ord = float(coral_loss(outputs["ordinal_logits"], tensors["labels_5class"]).detach().cpu())
        loss_reg = float(
            regression_loss(
                outputs["regression_score"],
                tensors["labels_5class"].to(dtype=torch.float32),
            ).detach().cpu()
        )
        if task_mode == "binary":
            loss_total = float(lambda_binary * loss_bin)
        elif task_mode == "ordinal":
            loss_total = float(lambda_ordinal * loss_ord + lambda_regression * loss_reg)
        else:
            loss_total = float(
                lambda_binary * loss_bin
                + lambda_ordinal * loss_ord
                + lambda_regression * loss_reg
            )

    y_pred_5class: np.ndarray | None = None
    y_pred_5class_ordinal_head: np.ndarray | None = None
    ordinal_metrics: dict[str, float] | None = None
    regression_mae = float("nan")
    if task_mode != "binary":
        y_pred_5class_ordinal_head = (
            coral_predict(outputs["ordinal_logits"]).detach().cpu().numpy().astype(int)
        )
        regression_score = np.clip(
            regression_score,
            0.0,
            float(base_model.num_classes - 1),
        )
        y_pred_5class = np.clip(
            np.rint(regression_score),
            0.0,
            float(base_model.num_classes - 1),
        ).astype(int)
        ordinal_metrics = _compute_ordinal_metrics(data.labels_5class, y_pred_5class)
        regression_mae = _compute_regression_mae(data.labels_5class, regression_score)

    if task_mode == "ordinal":
        if y_pred_5class is None:
            raise RuntimeError("Ordinal task mode requires ordinal predictions.")
        y_pred_binary = _ordinal_zero_based_to_binary(y_pred_5class)
    else:
        y_pred_binary = (binary_proba >= 0.5).astype(int)

    y_true_5class = data.labels_5class
    y_true_binary = data.labels_binary
    binary_metrics = _compute_binary_metrics(y_true_binary, y_pred_binary)

    return {
        "y_pred_5class": y_pred_5class,
        "y_pred_5class_ordinal_head": y_pred_5class_ordinal_head,
        "y_pred_binary": y_pred_binary,
        "ordinal_proba": ordinal_proba,
        "binary_proba": binary_proba,
        "regression_score": regression_score,
        "gates": gates,
        "ordinal_metrics": ordinal_metrics,
        "regression_mae": regression_mae,
        "binary_metrics": binary_metrics,
        "primary_metric_name": _primary_metric_name(task_mode),
        "test_loss_total": loss_total,
        "test_loss_binary": loss_bin,
        "test_loss_ordinal": loss_ord,
        "test_loss_regression": loss_reg,
    }


def _train_fold_model(
    *,
    config: RunConfig,
    train_set: TensorizedWindowSet,
    validation_set: TensorizedWindowSet,
    test_set: TensorizedWindowSet,
    modality_dims: dict[str, int],
    context_dim: int,
) -> tuple[MultimodalGatedFusionModel, dict[str, Any], dict[str, Any]]:
    device = _resolve_device(config)
    task_mode = str(config.multimodal_train.task_mode)
    model, num_cuda_devices = _build_training_model(
        config=config,
        modality_dims=modality_dims,
        context_dim=context_dim,
        device=device,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.multimodal_train.learning_rate,
        weight_decay=config.multimodal_train.weight_decay,
    )
    ordinal_pos_weight = _ordinal_threshold_pos_weight(train_set.labels_5class)
    coral_loss = CORALLoss(pos_weight=ordinal_pos_weight.to(device))
    binary_pos_weight = _positive_class_weight_binary(train_set.labels_binary).to(device)
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=binary_pos_weight)
    regression_loss = nn.SmoothL1Loss()
    ordinal_sample_weights = _ordinal_sample_weights(train_set.labels_5class)

    batch_size = min(
        int(config.multimodal_train.batch_size),
        max(1, int(len(train_set.labels_5class))),
    )
    rng = np.random.default_rng(config.seed)

    last_total = 0.0
    last_bin = 0.0
    last_ord = 0.0
    last_reg = 0.0
    best_epoch = -1
    best_metric_name = (
        "validation_binary_macro_f1" if task_mode == "binary" else "validation_class_mae"
    )
    best_primary_metric = np.inf if task_mode != "binary" else -np.inf
    best_validation_eval: dict[str, Any] | None = None
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    epochs_trained = 0
    early_stopped = False

    for epoch in range(int(config.multimodal_train.epochs)):
        model.train()
        batch_losses: list[float] = []
        batch_bin_losses: list[float] = []
        batch_ord_losses: list[float] = []
        batch_reg_losses: list[float] = []
        batch_sample_weights = ordinal_sample_weights if task_mode != "binary" else None
        for batch_indices in _iter_minibatches(
            len(train_set.labels_5class),
            batch_size,
            rng,
            sample_weights=batch_sample_weights,
        ):
            batch = _subset_tensorized(train_set, batch_indices)
            batch = _apply_modality_dropout(
                batch,
                dropout_prob=float(config.multimodal_train.modality_dropout_prob),
                rng=rng,
            )
            tensors = _to_device_batch(batch, device)
            optimizer.zero_grad()
            outputs = model(
                modality_inputs=tensors["modality_inputs"],
                modality_time_masks=tensors["modality_time_masks"],
                modality_mask=tensors["modality_mask"],
                context_inputs=tensors["context"],
            )
            loss_bin = bce_loss(outputs["binary_logits"], tensors["labels_binary"])
            loss_ord = coral_loss(outputs["ordinal_logits"], tensors["labels_5class"])
            loss_reg = regression_loss(
                outputs["regression_score"],
                tensors["labels_5class"].to(dtype=torch.float32),
            )
            if task_mode == "binary":
                loss = float(config.multimodal_train.lambda_binary) * loss_bin
            elif task_mode == "ordinal":
                loss = (
                    float(config.multimodal_train.lambda_ordinal) * loss_ord
                    + float(config.multimodal_train.lambda_regression) * loss_reg
                )
            else:
                loss = (
                    float(config.multimodal_train.lambda_binary) * loss_bin
                    + float(config.multimodal_train.lambda_ordinal) * loss_ord
                    + float(config.multimodal_train.lambda_regression) * loss_reg
                )
            loss.backward()
            optimizer.step()

            batch_losses.append(float(loss.detach().cpu()))
            batch_bin_losses.append(float(loss_bin.detach().cpu()))
            batch_ord_losses.append(float(loss_ord.detach().cpu()))
            batch_reg_losses.append(float(loss_reg.detach().cpu()))

        if batch_losses:
            last_total = float(np.mean(batch_losses))
            last_bin = float(np.mean(batch_bin_losses))
            last_ord = float(np.mean(batch_ord_losses))
            last_reg = float(np.mean(batch_reg_losses))

        epochs_trained = epoch + 1
        validation_result = _evaluate_model(
            model,
            validation_set,
            device=device,
            task_mode=task_mode,
            lambda_binary=float(config.multimodal_train.lambda_binary),
            lambda_ordinal=float(config.multimodal_train.lambda_ordinal),
            lambda_regression=float(config.multimodal_train.lambda_regression),
            ordinal_pos_weight=ordinal_pos_weight,
        )
        ordinal_macro_f1 = (
            float(validation_result["ordinal_metrics"]["macro_f1"])
            if validation_result["ordinal_metrics"] is not None
            else float("nan")
        )
        ordinal_class_mae = (
            float(validation_result["ordinal_metrics"]["mae"])
            if validation_result["ordinal_metrics"] is not None
            else float("nan")
        )
        ordinal_mae = float(validation_result["regression_mae"])
        current_validation_binary_macro_f1 = float(
            validation_result["binary_metrics"]["binary_macro_f1"]
        )
        current_primary_metric = (
            current_validation_binary_macro_f1
            if task_mode == "binary"
            else ordinal_class_mae
        )
        improved = (
            current_primary_metric > best_primary_metric
            if task_mode == "binary"
            else current_primary_metric < best_primary_metric
        )
        LOGGER.info(
            "Phase E multimodal epoch %s/%s | task=%s | "
            "train_loss=%.4f bin_loss=%.4f ord_loss=%.4f reg_loss=%.4f | "
            "validation_loss=%.4f validation_bin_loss=%.4f validation_ord_loss=%.4f "
            "validation_reg_loss=%.4f | validation_macro_f1=%s validation_mae=%s "
            "validation_class_mae=%s validation_bin_macro_f1=%.4f%s",
            epoch + 1,
            int(config.multimodal_train.epochs),
            task_mode,
            last_total,
            last_bin,
            last_ord,
            last_reg,
            float(validation_result["test_loss_total"]),
            float(validation_result["test_loss_binary"]),
            float(validation_result["test_loss_ordinal"]),
            float(validation_result["test_loss_regression"]),
            "n/a" if task_mode == "binary" else f"{ordinal_macro_f1:.4f}",
            "n/a" if task_mode == "binary" else f"{ordinal_mae:.4f}",
            "n/a" if task_mode == "binary" else f"{ordinal_class_mae:.4f}",
            current_validation_binary_macro_f1,
            " | new_best" if improved else "",
        )
        if improved:
            best_primary_metric = current_primary_metric
            best_epoch = epoch
            best_validation_eval = validation_result
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in _unwrap_model(model).state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= int(
                config.multimodal_train.early_stopping_patience
            ):
                early_stopped = True
                break

    if best_state is None or best_validation_eval is None:
        raise RuntimeError("Training completed without producing a best checkpoint.")

    _unwrap_model(model).load_state_dict(best_state)
    model.to(device)
    # The outer test fold is evaluated once, only after validation chose the checkpoint.
    test_eval = _evaluate_model(
        model,
        test_set,
        device=device,
        task_mode=task_mode,
        lambda_binary=float(config.multimodal_train.lambda_binary),
        lambda_ordinal=float(config.multimodal_train.lambda_ordinal),
        lambda_regression=float(config.multimodal_train.lambda_regression),
        ordinal_pos_weight=ordinal_pos_weight,
    )
    return model, {
        "train_loss_total": last_total,
        "train_loss_binary": last_bin,
        "train_loss_ordinal": last_ord,
        "train_loss_regression": last_reg,
        "epochs": float(epochs_trained),
        "best_epoch": float(best_epoch),
        "best_primary_metric": float(best_primary_metric),
        "best_primary_metric_name": best_metric_name,
        "checkpoint_selection_source": "validation",
        "early_stopped": bool(early_stopped),
        "best_test_macro_f1": float(
            test_eval["ordinal_metrics"]["macro_f1"]
            if task_mode != "binary" and test_eval["ordinal_metrics"] is not None
            else np.nan
        ),
        "best_test_mae": float(test_eval["regression_mae"] if task_mode != "binary" else np.nan),
        "best_test_class_mae": float(
            test_eval["ordinal_metrics"]["mae"]
            if task_mode != "binary" and test_eval["ordinal_metrics"] is not None
            else np.nan
        ),
        "best_test_bin_macro_f1": float(
            test_eval["binary_metrics"]["binary_macro_f1"]
        ),
        "device": str(device),
        "multi_gpu_enabled": bool(isinstance(model, nn.DataParallel)),
        "num_cuda_devices": int(num_cuda_devices),
        "task_mode": task_mode,
    }, test_eval


def run_phase_e_multimodal(
    config: RunConfig,
    *,
    window_index: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, Path], dict[str, Any]]:
    task_mode = str(config.multimodal_train.task_mode)
    split_mode = str(config.multimodal_train.split_mode)
    experiment_name = f"{EXPERIMENT_NAME}_{task_mode}"
    config.ensure_directories()
    windows = _read_window_index(config, window_index)
    session_manifest = _read_session_manifest(config)
    if windows.empty:
        raise FileNotFoundError("Window index is empty or missing. Run phase C before phase E.")
    if session_manifest.empty:
        raise FileNotFoundError("Session manifest is empty or missing. Run phase B before phase E.")

    data = _build_or_load_tensorized_dataset(
        config=config,
        windows=windows,
        session_manifest=session_manifest,
    )
    active_modalities = _active_modeled_modalities(config)
    if bool(config.multimodal_train.complete_windows_only):
        original_count = int(len(data.labels_5class))
        required_modalities = _complete_window_required_modalities(
            config,
            active_modalities=active_modalities,
        )
        data = _filter_complete_windows(
            data,
            required_modalities=required_modalities,
        )
        filtered_count = int(len(data.labels_5class))
        LOGGER.info(
            "Phase E multimodal complete-windows-only filter applied | kept=%s/%s windows | required_modalities=%s",
            filtered_count,
            original_count,
            ",".join(required_modalities),
        )
        if filtered_count == 0:
            raise ValueError(
                "No complete multimodal windows remain after applying complete-windows-only filtering."
            )
    data = _restrict_to_selected_modalities(
        data,
        selected_modalities=active_modalities,
    )
    if not bool(config.multimodal_train.use_context):
        data = _disable_context_features(data)
    participants = sorted({int(v) for v in data.meta["participant_id"].tolist()})
    folds = _build_participant_folds(
        data,
        config=config,
        split_mode=split_mode,
        num_folds=int(config.multimodal_train.num_folds),
        seed=int(config.seed),
    )

    directories = _prepare_directories(config)
    modeled_dims = modality_input_dims()
    context_dim = int(data.context.shape[1])

    split_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    fold_metric_rows: list[dict[str, Any]] = []
    binary_metric_rows: list[dict[str, Any]] = []
    baseline_prediction_rows: list[dict[str, Any]] = []
    baseline_fold_metric_rows: list[dict[str, Any]] = []
    baseline_binary_metric_rows: list[dict[str, Any]] = []

    for fold in folds:
        fold_id = int(fold.fold_id)
        test_group_label = ",".join(str(v) for v in fold.test_participants)
        test_participant_value: float | int = (
            int(fold.test_participants[0]) if len(fold.test_participants) == 1 else np.nan
        )
        outer_train_set_raw = _subset_tensorized(data, fold.train_indices)
        test_set_raw = _subset_tensorized(data, fold.test_indices)
        validation_split = _build_stratified_validation_split(
            outer_train_set_raw,
            validation_fraction=float(config.multimodal_train.validation_fraction),
            seed=int(config.seed) + fold_id,
        )
        train_set_raw = _subset_tensorized(
            outer_train_set_raw, validation_split.fit_indices
        )
        validation_set_raw = _subset_tensorized(
            outer_train_set_raw, validation_split.validation_indices
        )
        # Fit normalization on the inner fit split only; validation and test remain held out.
        normalization_stats = _fit_modality_normalization(train_set_raw)
        train_set = _apply_modality_normalization(train_set_raw, normalization_stats)
        validation_set = _apply_modality_normalization(
            validation_set_raw, normalization_stats
        )
        test_set = _apply_modality_normalization(test_set_raw, normalization_stats)

        split_rows.append(
            {
                "experiment": experiment_name,
                "task_mode": task_mode,
                "split_mode": split_mode,
                "fold_id": fold_id,
                "test_group": test_group_label,
                "test_participant_id": test_participant_value,
                "test_participants": json.dumps(list(fold.test_participants)),
                "train_participants": json.dumps(list(fold.train_participants)),
                "num_test_participants": int(len(fold.test_participants)),
                "num_train_participants": int(len(fold.train_participants)),
                "test_label_counts": json.dumps([int(v) for v in fold.test_label_counts.tolist()]),
                "test_present_labels": json.dumps(
                    [class_id + 1 for class_id, count in enumerate(fold.test_label_counts.tolist()) if int(count) > 0]
                ),
                "train_fold_normalized": True,
                "normalization_fit_source": "inner_train",
                "checkpoint_selection_source": "validation",
                "validation_fraction": float(config.multimodal_train.validation_fraction),
                "num_outer_train_windows": int(len(fold.train_indices)),
                "num_train_windows": int(len(train_set.labels_5class)),
                "num_validation_windows": int(len(validation_set.labels_5class)),
                "num_test_windows": int(len(fold.test_indices)),
            }
        )

        LOGGER.info(
            "Phase E multimodal fold %s/%s | split=%s | test_group=%s | "
            "train_windows=%s | validation_windows=%s | test_windows=%s | test_labels=%s",
            fold_id + 1,
            len(folds),
            split_mode,
            test_group_label,
            int(len(train_set.labels_5class)),
            int(len(validation_set.labels_5class)),
            int(len(fold.test_indices)),
            ",".join(str(class_id + 1) for class_id, count in enumerate(fold.test_label_counts.tolist()) if int(count) > 0),
        )

        model, train_stats, eval_result = _train_fold_model(
            config=config,
            train_set=train_set,
            validation_set=validation_set,
            test_set=test_set,
            modality_dims=modeled_dims,
            context_dim=context_dim,
        )
        baseline_results = {
            "random_train_distribution": _evaluate_random_baseline(
                train_set=train_set,
                test_set=test_set,
                task_mode=task_mode,
                seed=int(config.seed) + 1000 + fold_id,
            ),
            "train_mean_class": _evaluate_constant_baseline(
                train_set=train_set,
                test_set=test_set,
                task_mode=task_mode,
                strategy="mean",
            ),
            "train_median_class": _evaluate_constant_baseline(
                train_set=train_set,
                test_set=test_set,
                task_mode=task_mode,
                strategy="median",
            ),
        }
        y_pred_5 = eval_result["y_pred_5class"]
        y_pred_bin = eval_result["y_pred_binary"]
        ordinal_proba = eval_result["ordinal_proba"]
        gates = eval_result["gates"]
        y_true_5 = test_set.labels_5class
        y_true_bin = test_set.labels_binary

        fold_ordinal_metrics = dict(eval_result["ordinal_metrics"] or {})
        if task_mode != "binary":
            fold_ordinal_metrics["class_mae"] = float(
                fold_ordinal_metrics.get("mae", np.nan)
            )
            fold_ordinal_metrics["mae"] = float(eval_result["regression_mae"])
        fold_metric_rows.append(
            {
                "experiment": experiment_name,
                "task_mode": task_mode,
                "split_mode": split_mode,
                "fold_id": fold_id,
                "test_group": test_group_label,
                "test_participant_id": test_participant_value,
                "test_participants": json.dumps(list(fold.test_participants)),
                "train_fold_normalized": True,
                **fold_ordinal_metrics,
                **{
                    f"test_{k}": v
                    for k, v in eval_result["binary_metrics"].items()
                    if k != "num_samples"
                },
                **train_stats,
            }
        )
        binary_metric_rows.append(
            {
                "experiment": experiment_name,
                "task_mode": task_mode,
                "split_mode": split_mode,
                "fold_id": fold_id,
                "test_group": test_group_label,
                "test_participant_id": test_participant_value,
                "test_participants": json.dumps(list(fold.test_participants)),
                "train_fold_normalized": True,
                **eval_result["binary_metrics"],
            }
        )
        for baseline_name, baseline_result in baseline_results.items():
            baseline_fold_metric_rows.append(
                {
                    "experiment": experiment_name,
                    "baseline": baseline_name,
                    "task_mode": task_mode,
                    "split_mode": split_mode,
                    "fold_id": fold_id,
                    "test_group": test_group_label,
                    "test_participant_id": test_participant_value,
                    "test_participants": json.dumps(list(fold.test_participants)),
                    "train_fold_normalized": True,
                    **(baseline_result["ordinal_metrics"] or {}),
                    **{
                        f"test_{k}": v
                        for k, v in baseline_result["binary_metrics"].items()
                        if k != "num_samples"
                    },
                }
            )
            baseline_binary_metric_rows.append(
                {
                    "experiment": experiment_name,
                    "baseline": baseline_name,
                    "task_mode": task_mode,
                    "split_mode": split_mode,
                    "fold_id": fold_id,
                    "test_group": test_group_label,
                    "test_participant_id": test_participant_value,
                    "test_participants": json.dumps(list(fold.test_participants)),
                    "train_fold_normalized": True,
                    **baseline_result["binary_metrics"],
                }
            )

        LOGGER.info(
            "Phase E multimodal fold complete | task=%s | split=%s | test_group=%s | best_epoch=%s | "
            "best_primary_metric=%.4f | best_test_mae=%s | best_test_bin_macro_f1=%.4f",
            task_mode,
            split_mode,
            test_group_label,
            int(train_stats["best_epoch"]) + 1,
            float(train_stats["best_primary_metric"]),
            "n/a" if task_mode == "binary" else f"{float(train_stats['best_test_mae']):.4f}",
            float(eval_result["binary_metrics"]["binary_macro_f1"]),
        )
        for baseline_name, baseline_result in baseline_results.items():
            baseline_macro_f1 = (
                float(baseline_result["ordinal_metrics"]["macro_f1"])
                if baseline_result["ordinal_metrics"] is not None
                else float("nan")
            )
            baseline_mae = (
                float(baseline_result["ordinal_metrics"]["mae"])
                if baseline_result["ordinal_metrics"] is not None
                else float("nan")
            )
            LOGGER.info(
                "Phase E baseline=%s | task=%s | split=%s | test_group=%s | "
                "macro_f1=%s mae=%s bin_macro_f1=%.4f",
                baseline_name,
                task_mode,
                split_mode,
                test_group_label,
                "n/a" if task_mode == "binary" else f"{baseline_macro_f1:.4f}",
                "n/a" if task_mode == "binary" else f"{baseline_mae:.4f}",
                float(baseline_result["binary_metrics"]["binary_macro_f1"]),
            )

        cm_ordinal = (
            confusion_matrix(y_true_5, y_pred_5, labels=list(range(5)))
            if y_pred_5 is not None
            else None
        )
        cm_binary = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])
        if cm_ordinal is not None:
            pd.DataFrame(cm_ordinal).to_csv(
                directories.confusion_dir / f"{experiment_name}__fold_{fold_id}__ordinal.csv",
                index=False,
            )
        pd.DataFrame(cm_binary).to_csv(
            directories.confusion_dir / f"{experiment_name}__fold_{fold_id}__binary.csv",
            index=False,
        )

        checkpoint_path = directories.checkpoints_dir / f"{experiment_name}__fold_{fold_id}.pt"
        torch.save(
            {
                "state_dict": {
                    key: value.detach().cpu()
                    for key, value in _unwrap_model(model).state_dict().items()
                },
                "modality_order": list(MODELED_MODALITIES),
                "selected_modeled_modalities": list(active_modalities),
                "modality_input_dims": modeled_dims,
                "context_dim": context_dim,
                "use_multi_gpu": bool(config.multimodal_train.use_multi_gpu),
                "cuda_device_index": config.multimodal_train.cuda_device_index,
                "multi_gpu_enabled": bool(train_stats["multi_gpu_enabled"]),
                "num_cuda_devices": int(train_stats["num_cuda_devices"]),
                "use_native_frequency": bool(config.multimodal_train.use_native_frequency),
                "embedding_dim": int(config.multimodal_train.embedding_dim),
                "fusion_hidden_dim": int(config.multimodal_train.fusion_hidden_dim),
                "train_stats": train_stats,
                "normalization": {
                    modality: {
                        "mean": normalization_stats.mean[modality],
                        "std": normalization_stats.std[modality],
                    }
                    for modality in MODELED_MODALITIES
                },
            },
            checkpoint_path,
        )

        for idx in range(len(test_set.meta)):
            row = test_set.meta.iloc[idx]
            prediction_rows.append(
                {
                    "experiment": experiment_name,
                    "task_mode": task_mode,
                    "split_mode": split_mode,
                    "fold_id": fold_id,
                    "test_group": test_group_label,
                    "test_participant_id": test_participant_value,
                    "test_participants": json.dumps(list(fold.test_participants)),
                    "train_fold_normalized": True,
                    "window_id": str(row["window_id"]),
                    "session_key": str(row["session_key"]),
                    "participant_id": int(row["participant_id"]),
                    "event_id": str(row["event_id"]),
                    "video_uid": str(row["video_uid"]),
                    "y_true": int(y_true_5[idx] + 1),
                    "y_pred": (
                        int(y_pred_5[idx] + 1)
                        if y_pred_5 is not None
                        else np.nan
                    ),
                    "y_pred_ordinal_head": (
                        int(eval_result["y_pred_5class_ordinal_head"][idx] + 1)
                        if task_mode != "binary"
                        and eval_result["y_pred_5class_ordinal_head"] is not None
                        else np.nan
                    ),
                    "y_true_binary": int(y_true_bin[idx]),
                    "y_pred_binary": int(y_pred_bin[idx]),
                    "proba_1": float(ordinal_proba[idx, 0]) if task_mode != "binary" else np.nan,
                    "proba_2": float(ordinal_proba[idx, 1]) if task_mode != "binary" else np.nan,
                    "proba_3": float(ordinal_proba[idx, 2]) if task_mode != "binary" else np.nan,
                    "proba_4": float(ordinal_proba[idx, 3]) if task_mode != "binary" else np.nan,
                    "proba_5": float(ordinal_proba[idx, 4]) if task_mode != "binary" else np.nan,
                    "regression_score": float(eval_result["regression_score"][idx]) if task_mode != "binary" else np.nan,
                    "regression_pred_rounded": (
                        int(y_pred_5[idx] + 1)
                        if task_mode != "binary" and y_pred_5 is not None
                        else np.nan
                    ),
                    "proba_low": float(np.sum(ordinal_proba[idx, :2])) if task_mode != "binary" else np.nan,
                    "proba_high": float(np.sum(ordinal_proba[idx, 2:])) if task_mode != "binary" else np.nan,
                    "context_progress_center": float(test_set.context[idx, 0]),
                    "context_progress_sin": float(test_set.context[idx, 1]),
                    "context_progress_cos": float(test_set.context[idx, 2]),
                    "modalities": json.dumps(list(MODELED_MODALITIES)),
                    "active_modalities": json.dumps(list(active_modalities)),
                    "modality_mask": json.dumps([float(v) for v in test_set.modality_mask[idx].tolist()]),
                    "gates": json.dumps([float(v) for v in gates[idx].tolist()]),
                }
            )
            for baseline_name, baseline_result in baseline_results.items():
                baseline_y_pred_5 = baseline_result["y_pred_5class"]
                baseline_prediction_rows.append(
                    {
                        "experiment": experiment_name,
                        "baseline": baseline_name,
                        "task_mode": task_mode,
                        "split_mode": split_mode,
                        "fold_id": fold_id,
                        "test_group": test_group_label,
                        "test_participant_id": test_participant_value,
                        "test_participants": json.dumps(list(fold.test_participants)),
                        "train_fold_normalized": True,
                        "window_id": str(row["window_id"]),
                        "session_key": str(row["session_key"]),
                        "participant_id": int(row["participant_id"]),
                        "event_id": str(row["event_id"]),
                        "video_uid": str(row["video_uid"]),
                        "y_true": int(y_true_5[idx] + 1),
                        "y_pred": (
                            int(baseline_y_pred_5[idx] + 1)
                            if baseline_y_pred_5 is not None
                            else np.nan
                        ),
                        "y_true_binary": int(y_true_bin[idx]),
                        "y_pred_binary": int(baseline_result["y_pred_binary"][idx]),
                    }
                )

    predictions_df = pd.DataFrame(prediction_rows)
    if predictions_df.empty:
        raise ValueError("Phase E did not produce predictions. Check modality availability and inputs.")

    report_predictions_df = _aggregate_report_level(predictions_df)
    split_df = pd.DataFrame(split_rows)
    fold_metrics_df = pd.DataFrame(fold_metric_rows)
    binary_metrics_df = pd.DataFrame(binary_metric_rows)
    baseline_predictions_df = pd.DataFrame(baseline_prediction_rows)
    baseline_report_predictions_df = _aggregate_report_level(baseline_predictions_df)
    baseline_fold_metrics_df = pd.DataFrame(baseline_fold_metric_rows)
    baseline_binary_metrics_df = pd.DataFrame(baseline_binary_metric_rows)

    model_overall_ordinal = (
        _compute_ordinal_metrics(
            predictions_df["y_true"].to_numpy(dtype=int) - 1,
            predictions_df["y_pred"].to_numpy(dtype=int) - 1,
        )
        if task_mode != "binary"
        else {
            "accuracy": np.nan,
            "macro_f1": np.nan,
            "weighted_f1": np.nan,
            "mae": np.nan,
            "class_mae": np.nan,
            "num_samples": int(len(predictions_df)),
        }
    )
    if task_mode != "binary":
        model_overall_ordinal["class_mae"] = float(
            np.mean(
                np.abs(
                    predictions_df["y_true"].to_numpy(dtype=np.float32) - 1.0
                    - (predictions_df["y_pred"].to_numpy(dtype=np.float32) - 1.0)
                )
            )
        )
        model_overall_ordinal["mae"] = _compute_regression_mae(
            predictions_df["y_true"].to_numpy(dtype=np.int64) - 1,
            predictions_df["regression_score"].to_numpy(dtype=np.float32),
        )
    model_overall_binary = _compute_binary_metrics(
        predictions_df["y_true_binary"].to_numpy(dtype=int),
        predictions_df["y_pred_binary"].to_numpy(dtype=int),
    )
    model_report_overall_ordinal = (
        _compute_ordinal_metrics(
            report_predictions_df["y_true"].to_numpy(dtype=int) - 1,
            report_predictions_df["y_pred"].to_numpy(dtype=int) - 1,
        )
        if task_mode != "binary"
        else {
            "accuracy": np.nan,
            "macro_f1": np.nan,
            "weighted_f1": np.nan,
            "mae": np.nan,
            "class_mae": np.nan,
            "num_samples": int(len(report_predictions_df)),
        }
    )
    if task_mode != "binary":
        model_report_overall_ordinal["class_mae"] = float(
            np.mean(
                np.abs(
                    report_predictions_df["y_true"].to_numpy(dtype=np.float32) - 1.0
                    - (report_predictions_df["y_pred"].to_numpy(dtype=np.float32) - 1.0)
                )
            )
        )
        model_report_overall_ordinal["mae"] = _compute_regression_mae(
            report_predictions_df["y_true"].to_numpy(dtype=np.int64) - 1,
            report_predictions_df["regression_score"].to_numpy(dtype=np.float32),
        )
    model_report_overall_binary = _compute_binary_metrics(
        report_predictions_df["y_true_binary"].to_numpy(dtype=int),
        report_predictions_df["y_pred_binary"].to_numpy(dtype=int),
    )
    baseline_overall_rows: list[dict[str, Any]] = []
    baseline_binary_overall_rows: list[dict[str, Any]] = []
    baseline_report_overall_rows: list[dict[str, Any]] = []
    baseline_report_binary_overall_rows: list[dict[str, Any]] = []
    for baseline_name, baseline_group in baseline_predictions_df.groupby("baseline", sort=False):
        baseline_group = baseline_group.reset_index(drop=True)
        baseline_report_group = baseline_report_predictions_df[
            baseline_report_predictions_df["baseline"] == baseline_name
        ].reset_index(drop=True)
        baseline_overall_ordinal = (
            _compute_ordinal_metrics(
                baseline_group["y_true"].to_numpy(dtype=int) - 1,
                baseline_group["y_pred"].to_numpy(dtype=int) - 1,
            )
            if task_mode != "binary"
            else {
                "accuracy": np.nan,
                "macro_f1": np.nan,
                "weighted_f1": np.nan,
                "mae": np.nan,
                "num_samples": int(len(baseline_group)),
            }
        )
        baseline_overall_binary = _compute_binary_metrics(
            baseline_group["y_true_binary"].to_numpy(dtype=int),
            baseline_group["y_pred_binary"].to_numpy(dtype=int),
        )
        baseline_report_overall_ordinal = (
            _compute_ordinal_metrics(
                baseline_report_group["y_true"].to_numpy(dtype=int) - 1,
                baseline_report_group["y_pred"].to_numpy(dtype=int) - 1,
            )
            if task_mode != "binary"
            else {
                "accuracy": np.nan,
                "macro_f1": np.nan,
                "weighted_f1": np.nan,
                "mae": np.nan,
                "num_samples": int(len(baseline_report_group)),
            }
        )
        baseline_report_overall_binary = _compute_binary_metrics(
            baseline_report_group["y_true_binary"].to_numpy(dtype=int),
            baseline_report_group["y_pred_binary"].to_numpy(dtype=int),
        )
        baseline_overall_rows.append(
            {
                "experiment": experiment_name,
                "baseline": str(baseline_name),
                "task_mode": task_mode,
                **baseline_overall_ordinal,
            }
        )
        baseline_binary_overall_rows.append(
            {
                "experiment": experiment_name,
                "baseline": str(baseline_name),
                "task_mode": task_mode,
                **baseline_overall_binary,
            }
        )
        baseline_report_overall_rows.append(
            {
                "experiment": experiment_name,
                "baseline": str(baseline_name),
                "task_mode": task_mode,
                **baseline_report_overall_ordinal,
            }
        )
        baseline_report_binary_overall_rows.append(
            {
                "experiment": experiment_name,
                "baseline": str(baseline_name),
                "task_mode": task_mode,
                **baseline_report_overall_binary,
            }
        )

    predictions_path = write_table_with_fallback(
        predictions_df,
        directories.phase_dir / "predictions.parquet",
        logger=LOGGER,
    )
    baseline_predictions_path = write_table_with_fallback(
        baseline_predictions_df,
        directories.phase_dir / "predictions_baselines.parquet",
        logger=LOGGER,
    )
    random_predictions_path = write_table_with_fallback(
        baseline_predictions_df[baseline_predictions_df["baseline"] == "random_train_distribution"].reset_index(drop=True),
        directories.phase_dir / "predictions_random_baseline.parquet",
        logger=LOGGER,
    )
    report_predictions_path = write_table_with_fallback(
        report_predictions_df,
        directories.phase_dir / "predictions_report_level.parquet",
        logger=LOGGER,
    )
    baseline_report_predictions_path = write_table_with_fallback(
        baseline_report_predictions_df,
        directories.phase_dir / "predictions_report_level_baselines.parquet",
        logger=LOGGER,
    )
    random_report_predictions_path = write_table_with_fallback(
        baseline_report_predictions_df[baseline_report_predictions_df["baseline"] == "random_train_distribution"].reset_index(drop=True),
        directories.phase_dir / "predictions_report_level_random_baseline.parquet",
        logger=LOGGER,
    )
    split_artifact_key = _split_artifact_key(split_mode)
    splits_path = write_table_with_fallback(
        split_df,
        directories.phase_dir / f"{split_artifact_key}.parquet",
        logger=LOGGER,
    )
    fold_metrics_path = write_table_with_fallback(
        fold_metrics_df,
        directories.phase_dir / "metrics_per_fold.parquet",
        logger=LOGGER,
    )
    binary_fold_metrics_path = write_table_with_fallback(
        binary_metrics_df,
        directories.phase_dir / "metrics_binary_per_fold.parquet",
        logger=LOGGER,
    )
    baseline_fold_metrics_path = write_table_with_fallback(
        baseline_fold_metrics_df,
        directories.phase_dir / "metrics_baselines_per_fold.parquet",
        logger=LOGGER,
    )
    baseline_binary_fold_metrics_path = write_table_with_fallback(
        baseline_binary_metrics_df,
        directories.phase_dir / "metrics_baselines_binary_per_fold.parquet",
        logger=LOGGER,
    )
    random_fold_metrics_path = write_table_with_fallback(
        baseline_fold_metrics_df[baseline_fold_metrics_df["baseline"] == "random_train_distribution"].reset_index(drop=True),
        directories.phase_dir / "metrics_random_per_fold.parquet",
        logger=LOGGER,
    )
    random_binary_fold_metrics_path = write_table_with_fallback(
        baseline_binary_metrics_df[baseline_binary_metrics_df["baseline"] == "random_train_distribution"].reset_index(drop=True),
        directories.phase_dir / "metrics_random_binary_per_fold.parquet",
        logger=LOGGER,
    )
    overall_ordinal_path = write_table_with_fallback(
        pd.DataFrame(
            [{"experiment": experiment_name, "task_mode": task_mode, **model_overall_ordinal}]
        ),
        directories.phase_dir / "metrics_overall.parquet",
        logger=LOGGER,
    )
    overall_binary_path = write_table_with_fallback(
        pd.DataFrame(
            [{"experiment": experiment_name, "task_mode": task_mode, **model_overall_binary}]
        ),
        directories.phase_dir / "metrics_binary_overall.parquet",
        logger=LOGGER,
    )
    report_overall_ordinal_path = write_table_with_fallback(
        pd.DataFrame(
            [
                {
                    "experiment": experiment_name,
                    "task_mode": task_mode,
                    **model_report_overall_ordinal,
                }
            ]
        ),
        directories.phase_dir / "metrics_report_level_overall.parquet",
        logger=LOGGER,
    )
    report_overall_binary_path = write_table_with_fallback(
        pd.DataFrame(
            [
                {
                    "experiment": experiment_name,
                    "task_mode": task_mode,
                    **model_report_overall_binary,
                }
            ]
        ),
        directories.phase_dir / "metrics_report_level_binary_overall.parquet",
        logger=LOGGER,
    )
    baseline_overall_ordinal_path = write_table_with_fallback(
        pd.DataFrame(baseline_overall_rows),
        directories.phase_dir / "metrics_baselines_overall.parquet",
        logger=LOGGER,
    )
    baseline_overall_binary_path = write_table_with_fallback(
        pd.DataFrame(baseline_binary_overall_rows),
        directories.phase_dir / "metrics_baselines_binary_overall.parquet",
        logger=LOGGER,
    )
    baseline_report_overall_ordinal_path = write_table_with_fallback(
        pd.DataFrame(baseline_report_overall_rows),
        directories.phase_dir / "metrics_report_level_baselines_overall.parquet",
        logger=LOGGER,
    )
    baseline_report_overall_binary_path = write_table_with_fallback(
        pd.DataFrame(baseline_report_binary_overall_rows),
        directories.phase_dir / "metrics_report_level_baselines_binary_overall.parquet",
        logger=LOGGER,
    )
    random_overall_ordinal_path = write_table_with_fallback(
        pd.DataFrame([row for row in baseline_overall_rows if row["baseline"] == "random_train_distribution"]),
        directories.phase_dir / "metrics_random_overall.parquet",
        logger=LOGGER,
    )
    random_overall_binary_path = write_table_with_fallback(
        pd.DataFrame([row for row in baseline_binary_overall_rows if row["baseline"] == "random_train_distribution"]),
        directories.phase_dir / "metrics_random_binary_overall.parquet",
        logger=LOGGER,
    )
    random_report_overall_ordinal_path = write_table_with_fallback(
        pd.DataFrame([row for row in baseline_report_overall_rows if row["baseline"] == "random_train_distribution"]),
        directories.phase_dir / "metrics_report_level_random_overall.parquet",
        logger=LOGGER,
    )
    random_report_overall_binary_path = write_table_with_fallback(
        pd.DataFrame([row for row in baseline_report_binary_overall_rows if row["baseline"] == "random_train_distribution"]),
        directories.phase_dir / "metrics_report_level_random_binary_overall.parquet",
        logger=LOGGER,
    )

    summary_payload = {
        "experiment": experiment_name,
        "num_predictions": int(len(predictions_df)),
        "num_report_predictions": int(len(report_predictions_df)),
        "num_baseline_predictions": int(len(baseline_predictions_df)),
        "participants": participants,
        "modeled_modalities": list(MODELED_MODALITIES),
        "active_modeled_modalities": list(active_modalities),
        "multimodal_train": {
            "task_mode": task_mode,
            "split_mode": split_mode,
            "num_folds": int(config.multimodal_train.num_folds),
            "participant_folds": [list(fold) for fold in config.multimodal_train.participant_folds],
            "train_fold_normalization": True,
            "complete_windows_only": bool(config.multimodal_train.complete_windows_only),
            "complete_windows_reference": str(config.multimodal_train.complete_windows_reference),
            "selected_modeled_modalities": list(active_modalities),
            "use_context": bool(config.multimodal_train.use_context),
            "use_multi_gpu": bool(config.multimodal_train.use_multi_gpu),
            "cuda_device_index": config.multimodal_train.cuda_device_index,
            "target_points": int(config.multimodal_train.target_points),
            "use_native_frequency": bool(config.multimodal_train.use_native_frequency),
            "epochs": int(config.multimodal_train.epochs),
            "batch_size": int(config.multimodal_train.batch_size),
            "learning_rate": float(config.multimodal_train.learning_rate),
            "weight_decay": float(config.multimodal_train.weight_decay),
            "lambda_binary": float(config.multimodal_train.lambda_binary),
            "lambda_ordinal": float(config.multimodal_train.lambda_ordinal),
            "lambda_regression": float(config.multimodal_train.lambda_regression),
            "embedding_dim": int(config.multimodal_train.embedding_dim),
            "fusion_hidden_dim": int(config.multimodal_train.fusion_hidden_dim),
            "device": str(config.multimodal_train.device),
        },
    }
    summary_path = directories.phase_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(summary_payload, fp, indent=2, sort_keys=True)

    outputs = {
        "predictions": predictions_path,
        "predictions_baselines": baseline_predictions_path,
        "predictions_random_baseline": random_predictions_path,
        "report_predictions": report_predictions_path,
        "report_predictions_baselines": baseline_report_predictions_path,
        "report_predictions_random_baseline": random_report_predictions_path,
        "evaluation_splits": splits_path,
        split_artifact_key: splits_path,
        "metrics_per_fold": fold_metrics_path,
        "metrics_binary_per_fold": binary_fold_metrics_path,
        "metrics_baselines_per_fold": baseline_fold_metrics_path,
        "metrics_baselines_binary_per_fold": baseline_binary_fold_metrics_path,
        "metrics_random_per_fold": random_fold_metrics_path,
        "metrics_random_binary_per_fold": random_binary_fold_metrics_path,
        "metrics_overall": overall_ordinal_path,
        "metrics_binary_overall": overall_binary_path,
        "metrics_report_level_overall": report_overall_ordinal_path,
        "metrics_report_level_binary_overall": report_overall_binary_path,
        "metrics_baselines_overall": baseline_overall_ordinal_path,
        "metrics_baselines_binary_overall": baseline_overall_binary_path,
        "metrics_report_level_baselines_overall": baseline_report_overall_ordinal_path,
        "metrics_report_level_baselines_binary_overall": baseline_report_overall_binary_path,
        "metrics_random_overall": random_overall_ordinal_path,
        "metrics_random_binary_overall": random_overall_binary_path,
        "metrics_report_level_random_overall": random_report_overall_ordinal_path,
        "metrics_report_level_random_binary_overall": random_report_overall_binary_path,
        "summary": summary_path,
    }
    stats = {
        "experiments": [experiment_name],
        "num_windows": int(predictions_df["window_id"].nunique()),
        "num_prediction_rows": int(len(predictions_df)),
        "num_report_rows": int(len(report_predictions_df)),
        "participants": participants,
        "split_mode": split_mode,
        "num_folds": int(len(folds)),
    }
    return predictions_df, outputs, stats
