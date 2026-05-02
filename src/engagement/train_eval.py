"""Training loop, LOSO splits, and evaluation metrics (Phase E)."""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from .config import RunConfig
from .io_utils import read_table_with_fallback, write_table_with_fallback
from .model import (
    LinearClassifierHead,
    SourceAwareModalityFusion,
    WindowEmbeddingIndex,
    save_fusion_metadata,
)
from .multimodal_train_eval import run_phase_e_multimodal

LOGGER = logging.getLogger(__name__)

LABEL_ORDER = ("1", "2", "3", "4", "5")
LABEL_TO_ID = {label: idx for idx, label in enumerate(LABEL_ORDER)}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    modalities: tuple[str, ...]


@dataclass(frozen=True)
class PhaseEDirectories:
    phase_dir: Path
    checkpoints_dir: Path
    confusion_dir: Path


@dataclass
class FeatureStandardizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> "FeatureStandardizer":
        if x.ndim != 2:
            raise ValueError("x must be 2D.")
        if x.shape[0] == 0:
            return cls(mean=np.zeros((x.shape[1],), dtype=float), std=np.ones((x.shape[1],), dtype=float))

        mean = np.mean(x, axis=0)
        std = np.std(x, axis=0)
        std = np.where(std < 1e-8, 1.0, std)
        return cls(mean=mean.astype(float), std=std.astype(float))

    def transform(self, x: np.ndarray) -> np.ndarray:
        if x.ndim != 2:
            raise ValueError("x must be 2D.")
        if x.shape[1] != self.mean.shape[0]:
            raise ValueError("x has incompatible feature dimension for this standardizer.")
        return (x - self.mean) / self.std


class _MajorityVote:
    @staticmethod
    def vote(class_ids: list[int]) -> int:
        if not class_ids:
            return 0
        arr = np.asarray(class_ids, dtype=int)
        counts = np.bincount(arr, minlength=len(LABEL_ORDER))
        return int(np.argmax(counts))


def _normalize_label_5class(value: Any) -> str | None:
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        return None
    rounded = int(round(float(number)))
    if rounded in {1, 2, 3, 4, 5} and abs(float(number) - rounded) < 1e-6:
        return str(rounded)
    return None


def _coerce_embedding(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value.astype(float).reshape(-1)
    if isinstance(value, list):
        return np.asarray(value, dtype=float).reshape(-1)
    if not isinstance(value, str):
        return np.empty((0,), dtype=float)
    text = value.strip()
    if not text:
        return np.empty((0,), dtype=float)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return np.empty((0,), dtype=float)
    if not isinstance(parsed, list):
        return np.empty((0,), dtype=float)
    try:
        arr = np.asarray(parsed, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return np.empty((0,), dtype=float)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def _build_embedding_index(
    embedding_table: pd.DataFrame,
    allowed_window_ids: set[str],
) -> tuple[WindowEmbeddingIndex, dict[str, int]]:
    index: WindowEmbeddingIndex = {}
    modality_dims: dict[str, int] = {}

    for row in embedding_table.to_dict(orient="records"):
        window_id = str(row.get("window_id", ""))
        if not window_id or window_id not in allowed_window_ids:
            continue

        modality = str(row.get("modality", ""))
        source_id = str(row.get("source_id", ""))
        if not modality:
            continue

        emb = _coerce_embedding(row.get("embedding"))
        if emb.size == 0:
            continue

        index.setdefault(window_id, {}).setdefault(modality, []).append((source_id, emb))
        modality_dims[modality] = max(modality_dims.get(modality, 0), int(emb.size))

    return index, modality_dims


def _build_experiments(modality_dims: dict[str, int]) -> list[ExperimentSpec]:
    modalities = sorted([m for m, dim in modality_dims.items() if int(dim) > 0])
    experiments: list[ExperimentSpec] = []

    for modality in modalities:
        experiments.append(ExperimentSpec(name=f"unimodal_{modality}", modalities=(modality,)))

    without_eye = tuple([m for m in modalities if m != "eye"])
    if without_eye:
        experiments.append(ExperimentSpec(name="multimodal_without_eye", modalities=without_eye))

    with_eye = tuple(modalities)
    if with_eye:
        experiments.append(ExperimentSpec(name="multimodal_with_eye", modalities=with_eye))

    return experiments


def _compute_class_weights(y: np.ndarray) -> dict[int, float]:
    counts = np.bincount(y.astype(int), minlength=len(LABEL_ORDER))
    total = int(len(y))
    weights: dict[int, float] = {}
    for class_id, count in enumerate(counts):
        if count <= 0:
            weights[class_id] = 0.0
        else:
            weights[class_id] = float(total / (len(LABEL_ORDER) * count))
    return weights


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[dict[str, float], list[dict[str, Any]], np.ndarray]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    macro_f1 = float(f1_score(y_true, y_pred, average="macro", labels=list(range(len(LABEL_ORDER))), zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", labels=list(range(len(LABEL_ORDER))), zero_division=0))
    acc = float(accuracy_score(y_true, y_pred))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(LABEL_ORDER))),
        zero_division=0,
    )

    per_class_rows: list[dict[str, Any]] = []
    for class_id, class_name in enumerate(LABEL_ORDER):
        per_class_rows.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "precision": float(precision[class_id]),
                "recall": float(recall[class_id]),
                "f1": float(f1[class_id]),
                "support": int(support[class_id]),
            }
        )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(LABEL_ORDER))),
    )

    return {
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "accuracy": acc,
        "num_samples": int(len(y_true)),
    }, per_class_rows, cm


def _aggregate_report_level(predictions_df: pd.DataFrame) -> pd.DataFrame:
    if predictions_df.empty:
        return pd.DataFrame()

    grouped_rows: list[dict[str, Any]] = []
    group_cols = ["experiment", "fold_id", "test_participant_id", "event_id"]

    for keys, group in predictions_df.groupby(group_cols, sort=False):
        experiment, fold_id, test_participant_id, event_id = keys
        y_true_ids = [LABEL_TO_ID.get(str(v), 0) for v in group["y_true"].tolist()]
        y_pred_ids = [LABEL_TO_ID.get(str(v), 0) for v in group["y_pred"].tolist()]

        y_true_major = _MajorityVote.vote(y_true_ids)
        y_pred_major = _MajorityVote.vote(y_pred_ids)

        grouped_rows.append(
            {
                "experiment": str(experiment),
                "fold_id": int(fold_id),
                "test_participant_id": int(test_participant_id),
                "event_id": str(event_id),
                "y_true": ID_TO_LABEL[y_true_major],
                "y_pred": ID_TO_LABEL[y_pred_major],
                "num_windows": int(len(group)),
            }
        )

    return pd.DataFrame(grouped_rows)


def _build_matrices_for_rows(
    rows: pd.DataFrame,
    embedding_index: WindowEmbeddingIndex,
    fusion: SourceAwareModalityFusion,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    x_embed: list[np.ndarray] = []
    x_mask: list[np.ndarray] = []
    y: list[int] = []
    meta_rows: list[dict[str, Any]] = []

    for row in rows.to_dict(orient="records"):
        window_id = str(row.get("window_id", ""))
        label_name = _normalize_label_5class(row.get("label_5class", row.get("label_3class")))
        if label_name is None:
            continue
        label_id = LABEL_TO_ID.get(label_name)
        if label_id is None:
            continue

        modalities = embedding_index.get(window_id, {})
        emb_vec, mask_vec = fusion.fuse_window(modalities)

        x_embed.append(emb_vec)
        x_mask.append(mask_vec)
        y.append(label_id)

        meta_rows.append(
            {
                "window_id": window_id,
                "session_key": str(row.get("session_key", "")),
                "participant_id": int(row.get("participant_id")),
                "event_id": str(row.get("event_id", "")),
                "video_uid": str(row.get("video_uid", "")),
                "label_5class": label_name,
                "label_3class": label_name,
            }
        )

    if not x_embed:
        return (
            np.empty((0, sum(fusion.modality_dims.get(m, 0) for m in fusion.selected_modalities)), dtype=float),
            np.empty((0, len(fusion.selected_modalities)), dtype=float),
            np.empty((0,), dtype=int),
            pd.DataFrame(columns=["window_id", "session_key", "participant_id", "event_id", "video_uid", "label_5class", "label_3class"]),
        )

    return (
        np.stack(x_embed, axis=0),
        np.stack(x_mask, axis=0),
        np.asarray(y, dtype=int),
        pd.DataFrame(meta_rows),
    )


def prepare_phase_e_inputs(
    config: RunConfig,
    window_index: pd.DataFrame | None,
    embedding_table: pd.DataFrame | None,
) -> tuple[pd.DataFrame, WindowEmbeddingIndex, dict[str, int], list[ExperimentSpec], list[int], PhaseEDirectories]:
    config.ensure_directories()

    if window_index is None:
        run_window_parquet = config.run_dir / "windows" / "window_index.parquet"
        window_index = read_table_with_fallback(run_window_parquet)
        if window_index.empty:
            window_index = read_table_with_fallback(config.windows_dir / "window_index.parquet")

    if embedding_table is None:
        run_embedding_parquet = config.run_dir / "embeddings" / "embedding_table.parquet"
        embedding_table = read_table_with_fallback(run_embedding_parquet)
        if embedding_table.empty:
            embedding_table = read_table_with_fallback(config.embeddings_dir / "embedding_table.parquet")

    if window_index.empty:
        raise FileNotFoundError("Window index is empty or missing. Run phase C before phase E.")
    if embedding_table.empty:
        raise FileNotFoundError("Embedding table is empty or missing. Run phase D before phase E.")

    windows = window_index.copy()
    label_source = windows.get("label_5class") if "label_5class" in windows.columns else windows.get("label_3class")
    windows["label_5class"] = pd.to_numeric(label_source, errors="coerce")
    windows = windows[windows["label_5class"].notna()].copy()
    windows["label_5class"] = windows["label_5class"].astype(int).astype(str)
    windows = windows[windows["label_5class"].isin(LABEL_ORDER)].copy()
    windows = windows[windows["participant_id"].notna()].copy()
    if windows.empty:
        raise ValueError("No supervised windows are available for Phase E training.")

    windows["participant_id"] = pd.to_numeric(windows["participant_id"], errors="coerce")
    windows = windows[windows["participant_id"].notna()].copy()
    windows["participant_id"] = windows["participant_id"].astype(int)

    allowed_window_ids = {str(wid) for wid in windows["window_id"].astype(str).tolist()}
    embedding_index, modality_dims = _build_embedding_index(embedding_table, allowed_window_ids)
    experiments = _build_experiments(modality_dims)
    if not experiments:
        raise ValueError("No valid modality embeddings found for Phase E experiments.")

    participants = sorted({int(v) for v in windows["participant_id"].tolist()})
    if len(participants) < 2:
        raise ValueError("LOSO requires at least 2 participants with windows.")

    phase_dir = config.run_dir / "phase_e"
    checkpoints_dir = phase_dir / "checkpoints"
    confusion_dir = phase_dir / "confusion"
    phase_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    confusion_dir.mkdir(parents=True, exist_ok=True)

    return windows, embedding_index, modality_dims, experiments, participants, PhaseEDirectories(
        phase_dir=phase_dir,
        checkpoints_dir=checkpoints_dir,
        confusion_dir=confusion_dir,
    )


def evaluate_fold(
    *,
    config: RunConfig,
    experiment: ExperimentSpec,
    fold_id: int,
    test_pid: int,
    train_rows: pd.DataFrame,
    test_rows: pd.DataFrame,
    embedding_index: WindowEmbeddingIndex,
    dims_for_exp: dict[str, int],
    directories: PhaseEDirectories,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    prediction_rows: list[dict[str, Any]] = []
    fold_metric_rows: list[dict[str, Any]] = []
    per_class_metric_rows: list[dict[str, Any]] = []

    fusion = SourceAwareModalityFusion(
        selected_modalities=experiment.modalities,
        modality_dims=dims_for_exp,
    )
    fusion.fit()

    x_train_embed, x_train_mask, y_train, _train_meta = _build_matrices_for_rows(
        train_rows,
        embedding_index=embedding_index,
        fusion=fusion,
    )
    x_test_embed, x_test_mask, y_test, test_meta = _build_matrices_for_rows(
        test_rows,
        embedding_index=embedding_index,
        fusion=fusion,
    )

    if x_train_embed.shape[0] == 0 or x_test_embed.shape[0] == 0:
        LOGGER.warning(
            "Skipping %s fold %s due to empty train/test matrices.",
            experiment.name,
            fold_id,
        )
        return prediction_rows, fold_metric_rows, per_class_metric_rows

    standardizer = FeatureStandardizer.fit(x_train_embed)
    x_train_embed_std = standardizer.transform(x_train_embed)
    x_test_embed_std = standardizer.transform(x_test_embed)

    class_weights = _compute_class_weights(y_train)

    model = LinearClassifierHead(
        input_dim=x_train_embed_std.shape[1],
        n_classes=len(LABEL_ORDER),
        seed=config.seed,
        max_iter=400,
    )
    train_stats = model.fit(
        x_embed=x_train_embed_std,
        x_mask=x_train_mask,
        y=y_train,
        class_weights=class_weights,
    )

    y_pred = model.predict(x_embed=x_test_embed_std, x_mask=x_test_mask)
    y_proba = model.predict_proba(x_embed=x_test_embed_std, x_mask=x_test_mask)

    metrics, per_class, cm = _compute_metrics(y_test, y_pred)
    fold_metric_rows.append(
        {
            "experiment": experiment.name,
            "fold_id": int(fold_id),
            "test_participant_id": int(test_pid),
            **metrics,
            **{f"train_{k}": v for k, v in train_stats.items()},
        }
    )

    for row in per_class:
        per_class_metric_rows.append(
            {
                "experiment": experiment.name,
                "fold_id": int(fold_id),
                "test_participant_id": int(test_pid),
                **row,
            }
        )

    cm_df = pd.DataFrame(cm, index=LABEL_ORDER, columns=LABEL_ORDER)
    cm_path = directories.confusion_dir / f"{experiment.name}__fold_{fold_id}__participant_{test_pid}.csv"
    cm_df.to_csv(cm_path, index=True)

    for idx in range(len(test_meta)):
        prediction_rows.append(
            {
                "experiment": experiment.name,
                "fold_id": int(fold_id),
                "test_participant_id": int(test_pid),
                "window_id": str(test_meta.iloc[idx]["window_id"]),
                "session_key": str(test_meta.iloc[idx]["session_key"]),
                "participant_id": int(test_meta.iloc[idx]["participant_id"]),
                "event_id": str(test_meta.iloc[idx]["event_id"]),
                "video_uid": str(test_meta.iloc[idx]["video_uid"]),
                "y_true": ID_TO_LABEL[int(y_test[idx])],
                "y_pred": ID_TO_LABEL[int(y_pred[idx])],
                "proba_1": float(y_proba[idx, LABEL_TO_ID["1"]]),
                "proba_2": float(y_proba[idx, LABEL_TO_ID["2"]]),
                "proba_3": float(y_proba[idx, LABEL_TO_ID["3"]]),
                "proba_4": float(y_proba[idx, LABEL_TO_ID["4"]]),
                "proba_5": float(y_proba[idx, LABEL_TO_ID["5"]]),
                "proba_low": float(y_proba[idx, LABEL_TO_ID["1"]] + y_proba[idx, LABEL_TO_ID["2"]]),
                "proba_neutral": float(y_proba[idx, LABEL_TO_ID["3"]]),
                "proba_high": float(y_proba[idx, LABEL_TO_ID["4"]] + y_proba[idx, LABEL_TO_ID["5"]]),
                "modalities": json.dumps(list(experiment.modalities)),
            }
        )

    checkpoint_path = directories.checkpoints_dir / f"{experiment.name}__fold_{fold_id}__participant_{test_pid}.npz"
    model.save(checkpoint_path)
    save_fusion_metadata(
        directories.checkpoints_dir / f"{experiment.name}__fold_{fold_id}__participant_{test_pid}.json",
        {
            "artifact_schema_version": 2,
            "model_impl": model.model_impl,
            "experiment": experiment.name,
            "fold_id": int(fold_id),
            "test_participant_id": int(test_pid),
            "modalities": list(experiment.modalities),
            "modality_dims": {k: int(v) for k, v in dims_for_exp.items()},
            "source_fusion": fusion.to_dict(),
            "standardizer": {
                "mean": [float(v) for v in standardizer.mean.tolist()],
                "std": [float(v) for v in standardizer.std.tolist()],
            },
            "class_weights": {str(k): float(v) for k, v in class_weights.items()},
            "train_stats": train_stats,
        },
    )

    return prediction_rows, fold_metric_rows, per_class_metric_rows


def run_loso_for_experiment(
    *,
    config: RunConfig,
    experiment: ExperimentSpec,
    windows: pd.DataFrame,
    participants: list[int],
    modality_dims: dict[str, int],
    embedding_index: WindowEmbeddingIndex,
    directories: PhaseEDirectories,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    split_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    fold_metric_rows: list[dict[str, Any]] = []
    per_class_metric_rows: list[dict[str, Any]] = []

    dims_for_exp = {m: modality_dims[m] for m in experiment.modalities}

    for fold_id, test_pid in enumerate(participants):
        train_rows = windows[windows["participant_id"] != test_pid].copy()
        test_rows = windows[windows["participant_id"] == test_pid].copy()

        if train_rows.empty or test_rows.empty:
            continue

        overlap = set(train_rows["participant_id"].tolist()) & set(test_rows["participant_id"].tolist())
        if overlap:
            raise AssertionError(f"LOSO leakage detected in fold {fold_id}: {sorted(overlap)}")

        split_rows.append(
            {
                "experiment": experiment.name,
                "fold_id": fold_id,
                "test_participant_id": int(test_pid),
                "train_participants": json.dumps(sorted(set(int(v) for v in train_rows["participant_id"].tolist()))),
                "num_train_windows": int(len(train_rows)),
                "num_test_windows": int(len(test_rows)),
            }
        )

        fold_predictions, fold_metrics, fold_per_class = evaluate_fold(
            config=config,
            experiment=experiment,
            fold_id=fold_id,
            test_pid=test_pid,
            train_rows=train_rows,
            test_rows=test_rows,
            embedding_index=embedding_index,
            dims_for_exp=dims_for_exp,
            directories=directories,
        )

        prediction_rows.extend(fold_predictions)
        fold_metric_rows.extend(fold_metrics)
        per_class_metric_rows.extend(fold_per_class)

    return split_rows, prediction_rows, fold_metric_rows, per_class_metric_rows


def export_phase_e_artifacts(
    *,
    directories: PhaseEDirectories,
    split_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    fold_metric_rows: list[dict[str, Any]],
    per_class_metric_rows: list[dict[str, Any]],
) -> tuple[pd.DataFrame, dict[str, Path], dict[str, Any]]:
    predictions_df = pd.DataFrame(prediction_rows)
    split_df = pd.DataFrame(split_rows)
    fold_metrics_df = pd.DataFrame(fold_metric_rows)
    per_class_df = pd.DataFrame(per_class_metric_rows)

    if predictions_df.empty:
        raise ValueError("Phase E did not produce predictions. Check modality availability and inputs.")

    report_predictions_df = _aggregate_report_level(predictions_df)

    overall_rows: list[dict[str, Any]] = []
    report_overall_rows: list[dict[str, Any]] = []

    for experiment_name, group in predictions_df.groupby("experiment", sort=False):
        y_true = np.asarray([LABEL_TO_ID[v] for v in group["y_true"].tolist()], dtype=int)
        y_pred = np.asarray([LABEL_TO_ID[v] for v in group["y_pred"].tolist()], dtype=int)
        metrics, _per_class, _cm = _compute_metrics(y_true, y_pred)
        overall_rows.append({"experiment": str(experiment_name), **metrics})

    for experiment_name, group in report_predictions_df.groupby("experiment", sort=False):
        y_true = np.asarray([LABEL_TO_ID[v] for v in group["y_true"].tolist()], dtype=int)
        y_pred = np.asarray([LABEL_TO_ID[v] for v in group["y_pred"].tolist()], dtype=int)
        metrics, _per_class, _cm = _compute_metrics(y_true, y_pred)
        report_overall_rows.append({"experiment": str(experiment_name), **metrics})

    overall_df = pd.DataFrame(overall_rows)
    report_overall_df = pd.DataFrame(report_overall_rows)

    predictions_path = write_table_with_fallback(predictions_df, directories.phase_dir / "predictions.parquet", logger=LOGGER)
    report_predictions_path = write_table_with_fallback(
        report_predictions_df,
        directories.phase_dir / "predictions_report_level.parquet",
        logger=LOGGER,
    )
    splits_path = write_table_with_fallback(split_df, directories.phase_dir / "loso_splits.parquet", logger=LOGGER)
    fold_metrics_path = write_table_with_fallback(
        fold_metrics_df,
        directories.phase_dir / "metrics_per_fold.parquet",
        logger=LOGGER,
    )
    per_class_path = write_table_with_fallback(
        per_class_df,
        directories.phase_dir / "metrics_per_class.parquet",
        logger=LOGGER,
    )
    overall_path = write_table_with_fallback(overall_df, directories.phase_dir / "metrics_overall.parquet", logger=LOGGER)
    report_overall_path = write_table_with_fallback(
        report_overall_df,
        directories.phase_dir / "metrics_report_level_overall.parquet",
        logger=LOGGER,
    )

    summary_payload = {
        "num_predictions": int(len(predictions_df)),
        "num_report_predictions": int(len(report_predictions_df)),
        "num_experiments": int(predictions_df["experiment"].nunique()),
        "num_folds": int(fold_metrics_df[["experiment", "fold_id"]].drop_duplicates().shape[0]),
        "label_order": list(LABEL_ORDER),
    }
    summary_json_path = directories.phase_dir / "summary.json"
    with summary_json_path.open("w", encoding="utf-8") as fp:
        json.dump(summary_payload, fp, indent=2, sort_keys=True)

    outputs = {
        "predictions": predictions_path,
        "report_predictions": report_predictions_path,
        "loso_splits": splits_path,
        "metrics_per_fold": fold_metrics_path,
        "metrics_per_class": per_class_path,
        "metrics_overall": overall_path,
        "metrics_report_level_overall": report_overall_path,
        "summary": summary_json_path,
    }

    stats = {
        "experiments": sorted(predictions_df["experiment"].unique().tolist()),
        "num_windows": int(predictions_df["window_id"].nunique()),
        "num_prediction_rows": int(len(predictions_df)),
        "num_report_rows": int(len(report_predictions_df)),
        "participants": sorted({int(v) for v in predictions_df["participant_id"].tolist()}),
    }

    return predictions_df, outputs, stats


def _run_phase_e_legacy(
    config: RunConfig,
    window_index: pd.DataFrame | None = None,
    embedding_table: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, Path], dict[str, Any]]:
    (
        windows,
        embedding_index,
        modality_dims,
        experiments,
        participants,
        directories,
    ) = prepare_phase_e_inputs(config, window_index, embedding_table)

    split_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    fold_metric_rows: list[dict[str, Any]] = []
    per_class_metric_rows: list[dict[str, Any]] = []

    for experiment in experiments:
        exp_split_rows, exp_prediction_rows, exp_fold_metric_rows, exp_per_class_rows = run_loso_for_experiment(
            config=config,
            experiment=experiment,
            windows=windows,
            participants=participants,
            modality_dims=modality_dims,
            embedding_index=embedding_index,
            directories=directories,
        )
        split_rows.extend(exp_split_rows)
        prediction_rows.extend(exp_prediction_rows)
        fold_metric_rows.extend(exp_fold_metric_rows)
        per_class_metric_rows.extend(exp_per_class_rows)

    return export_phase_e_artifacts(
        directories=directories,
        split_rows=split_rows,
        prediction_rows=prediction_rows,
        fold_metric_rows=fold_metric_rows,
        per_class_metric_rows=per_class_metric_rows,
    )


def _should_use_multimodal_backend(
    config: RunConfig,
    window_index: pd.DataFrame | None,
    embedding_table: pd.DataFrame | None,
) -> bool:
    candidate = window_index
    if candidate is None:
        run_window_parquet = config.run_dir / "windows" / "window_index.parquet"
        candidate = read_table_with_fallback(run_window_parquet)
        if candidate.empty:
            candidate = read_table_with_fallback(config.windows_dir / "window_index.parquet")

    if candidate is None or candidate.empty:
        return False

    required_cols = {"t_start_sec", "t_end_sec", "t_start_video_sec", "t_end_video_sec"}
    if not required_cols.issubset(set(candidate.columns)):
        return False

    run_manifest = config.run_dir / "manifests" / "session_manifest.parquet"
    manifest_df = read_table_with_fallback(run_manifest)
    if manifest_df.empty:
        manifest_df = read_table_with_fallback(config.manifests_dir / "session_manifest.parquet")
    if manifest_df.empty:
        return False

    # Prefer the new raw-window backend whenever the necessary artifacts exist.
    # The embedding-based path remains available as a fallback for older tests
    # and callers that only provide window labels plus cached embeddings.
    _ = embedding_table
    return True


def run_phase_e(
    config: RunConfig,
    window_index: pd.DataFrame | None = None,
    embedding_table: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, Path], dict[str, Any]]:
    if _should_use_multimodal_backend(config, window_index, embedding_table):
        return run_phase_e_multimodal(config, window_index=window_index)
    return _run_phase_e_legacy(
        config,
        window_index=window_index,
        embedding_table=embedding_table,
    )

