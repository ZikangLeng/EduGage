from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error


REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINES_ROOT = REPO_ROOT / "src" / "engagement" / "baselines"

FIXED_COLUMNS = [
    "baseline_family",
    "split_name",
    "model",
    "fold_id",
    "test_participant_ids",
    "accuracy",
    "within_1_accuracy",
    "binary_accuracy",
    "binary_macro_f1",
    "mae",
    "raw_mae",
    "ordinal_macro_f1",
    "ordinal_weighted_f1",
    "num_samples",
    "train_best_loss",
    "train_epochs",
]

METRIC_COLUMNS = [
    "accuracy",
    "within_1_accuracy",
    "binary_accuracy",
    "binary_macro_f1",
    "mae",
    "raw_mae",
    "ordinal_macro_f1",
    "ordinal_weighted_f1",
    "train_best_loss",
]


def _low_high(labels: np.ndarray) -> np.ndarray:
    return (np.asarray(labels, dtype=int) >= 3).astype(int)


def _architecture_metrics() -> pd.DataFrame:
    arch_root = BASELINES_ROOT / "foundational_architecture_baseline" / "results"
    preferred = arch_root / "architecture_models_fixed4_regression_mae_full"
    fallback = arch_root / "architecture_models_fixed4_official_full"
    arch_dir = preferred if (preferred / "metrics_per_fold.csv").exists() else fallback
    metrics_path = arch_dir / "metrics_per_fold.csv"
    predictions_path = arch_dir / "predictions.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing architecture metrics: {metrics_path}")

    metrics = pd.read_csv(metrics_path)
    if {"mae", "within_1_accuracy", "binary_accuracy", "binary_macro_f1"}.issubset(metrics.columns):
        out = metrics.copy()
        out.insert(0, "baseline_family", "foundational_architecture")
        out["ordinal_macro_f1"] = out.get("macro_f1", np.nan)
        out["ordinal_weighted_f1"] = out.get("weighted_f1", np.nan)
        return out.reindex(columns=FIXED_COLUMNS)

    preds = pd.read_csv(predictions_path)
    rows = []
    for keys, group in preds.groupby(["split_name", "model", "fold_id", "test_participant_ids"], sort=False):
        split_name, model, fold_id, test_ids = keys
        y_true = group["y_true"].astype(int).to_numpy()
        y_pred = group["y_pred"].astype(int).to_numpy()
        train_row = metrics[
            (metrics["split_name"] == split_name)
            & (metrics["model"] == model)
            & (metrics["fold_id"] == fold_id)
            & (metrics["test_participant_ids"] == test_ids)
        ].iloc[0]
        rows.append(
            {
                "baseline_family": "foundational_architecture",
                "split_name": split_name,
                "model": model,
                "fold_id": int(fold_id),
                "test_participant_ids": test_ids,
                "accuracy": accuracy_score(y_true, y_pred),
                "within_1_accuracy": np.mean(np.abs(y_true - y_pred) <= 1),
                "binary_accuracy": accuracy_score(_low_high(y_true), _low_high(y_pred)),
                "binary_macro_f1": f1_score(_low_high(y_true), _low_high(y_pred), average="macro", zero_division=0),
                "mae": mean_absolute_error(y_true, y_pred),
                "raw_mae": np.nan,
                "ordinal_macro_f1": train_row.get("macro_f1", np.nan),
                "ordinal_weighted_f1": train_row.get("weighted_f1", np.nan),
                "num_samples": len(group),
                "train_best_loss": train_row.get("train_best_loss", np.nan),
                "train_epochs": train_row.get("train_epochs", np.nan),
            }
        )
    return pd.DataFrame(rows).reindex(columns=FIXED_COLUMNS)


def _ml_metrics() -> pd.DataFrame:
    path = BASELINES_ROOT / "ml_baseline" / "results" / "ml_baselines_fixed4" / "metrics_per_fold.csv"
    df = pd.read_csv(path)
    df.insert(0, "baseline_family", "ml_statistical_regressor")
    return df.reindex(columns=FIXED_COLUMNS)


def _embedding_metrics() -> pd.DataFrame:
    path = (
        BASELINES_ROOT
        / "foundational_embeddings_concat_baseline"
        / "results"
        / "concat_foundational_embeddings_linear_head_fixed4_metrics.csv"
    )
    return pd.read_csv(path).reindex(columns=FIXED_COLUMNS)


def main() -> int:
    out_dir = BASELINES_ROOT / "ml_baseline" / "results"
    full = pd.concat([_ml_metrics(), _architecture_metrics(), _embedding_metrics()], ignore_index=True)
    for column in METRIC_COLUMNS:
        full[column] = pd.to_numeric(full[column], errors="coerce").round(4)

    full_path = out_dir / "all_baselines_fixed4_metrics.csv"
    summary_path = out_dir / "all_baselines_fixed4_summary.csv"
    full.to_csv(full_path, index=False)
    summary = (
        full.groupby(["baseline_family", "model"], sort=False)[
            ["mae", "accuracy", "within_1_accuracy", "binary_accuracy", "binary_macro_f1", "ordinal_macro_f1", "ordinal_weighted_f1"]
        ]
        .mean()
        .round(4)
        .reset_index()
    )
    summary.to_csv(summary_path, index=False)
    print(full_path)
    print(summary_path)
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
