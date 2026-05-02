from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error

REPO_ROOT = Path(__file__).resolve().parents[4]
BASELINE_ROOT = Path(__file__).resolve().parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from engagement.architecture_data import build_architecture_dataset_from_preprocessed  # noqa: E402
from engagement.train_eval import ID_TO_LABEL, LABEL_ORDER  # noqa: E402

LOGGER = logging.getLogger(__name__)

FIXED_4_FOLDS: tuple[tuple[int, ...], ...] = (
    (1, 2, 3, 10),
    (4, 5, 7, 9),
    (6, 8, 11, 12),
    (13, 14, 15, 16),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run simple label-only comparison baselines.")
    parser.add_argument("--preprocessed-root", type=Path, default=REPO_ROOT / "preprocessed_data")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASELINE_ROOT / "results" / "mean_mode_random_baselines",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def _mode_label(y_train: np.ndarray) -> int:
    counts = np.bincount(y_train.astype(int), minlength=len(LABEL_ORDER))
    return int(np.argmax(counts))


def _constant_predictions(y_train: np.ndarray, n_test: int) -> dict[str, np.ndarray]:
    return {
        "mean": np.full((n_test,), int(np.clip(np.rint(np.mean(y_train)), 0, len(LABEL_ORDER) - 1)), dtype=int),
        "mode": np.full((n_test,), _mode_label(y_train), dtype=int),
    }


def _random_distribution_predictions(y_train: np.ndarray, n_test: int, rng: np.random.Generator) -> np.ndarray:
    counts = np.bincount(y_train.astype(int), minlength=len(LABEL_ORDER)).astype(float)
    if counts.sum() <= 0:
        probs = np.full((len(LABEL_ORDER),), 1.0 / len(LABEL_ORDER), dtype=float)
    else:
        probs = counts / counts.sum()
    return rng.choice(np.arange(len(LABEL_ORDER), dtype=int), size=n_test, replace=True, p=probs).astype(int)


def _one_hot_proba(y_pred: np.ndarray) -> np.ndarray:
    proba = np.zeros((len(y_pred), len(LABEL_ORDER)), dtype=float)
    proba[np.arange(len(y_pred)), np.clip(y_pred, 0, len(LABEL_ORDER) - 1)] = 1.0
    return proba


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_architecture_dataset_from_preprocessed(args.preprocessed_root)
    y = dataset.y
    metadata = dataset.metadata.copy()
    pid_values = metadata["participant_id"].astype(int).to_numpy()

    LOGGER.info("Loaded %s labeled windows for simple baselines.", len(y))

    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []

    for fold_id, test_participants in enumerate(FIXED_4_FOLDS):
        test_mask = np.isin(pid_values, test_participants)
        train_mask = ~test_mask
        if not np.any(test_mask) or not np.any(train_mask):
            continue

        y_train = y[train_mask]
        y_test = y[test_mask]
        rng = np.random.default_rng(int(args.seed) + int(fold_id))
        fold_predictions = _constant_predictions(y_train, len(y_test))
        fold_predictions["random_distribution"] = _random_distribution_predictions(y_train, len(y_test), rng)

        for model_name, y_pred in fold_predictions.items():
            macro_f1 = float(f1_score(y_test, y_pred, average="macro", labels=list(range(len(LABEL_ORDER))), zero_division=0))
            weighted_f1 = float(f1_score(y_test, y_pred, average="weighted", labels=list(range(len(LABEL_ORDER))), zero_division=0))
            accuracy = float(accuracy_score(y_test, y_pred))
            mae = float(mean_absolute_error(y_test, y_pred))
            within_1_accuracy = float(np.mean(np.abs(y_test.astype(int) - y_pred.astype(int)) <= 1))
            y_test_binary = (y_test.astype(int) >= 2).astype(int)
            y_pred_binary = (y_pred.astype(int) >= 2).astype(int)
            binary_accuracy = float(accuracy_score(y_test_binary, y_pred_binary))
            binary_macro_f1 = float(f1_score(y_test_binary, y_pred_binary, average="macro", zero_division=0))
            metric_rows.append(
                {
                    "split_name": "fixed4",
                    "model": model_name,
                    "fold_id": int(fold_id),
                    "test_participant_ids": json.dumps(list(test_participants)),
                    "macro_f1": macro_f1,
                    "weighted_f1": weighted_f1,
                    "accuracy": accuracy,
                    "mae": mae,
                    "within_1_accuracy": within_1_accuracy,
                    "binary_accuracy": binary_accuracy,
                    "binary_macro_f1": binary_macro_f1,
                    "num_samples": int(len(y_test)),
                    "train_label_mean": float(np.mean(y_train)),
                    "train_label_mode": int(_mode_label(y_train)),
                }
            )

            y_proba = _one_hot_proba(y_pred)
            fold_meta = metadata.loc[test_mask].reset_index(drop=True)
            for idx, row in fold_meta.iterrows():
                out = {
                    "split_name": "fixed4",
                    "model": model_name,
                    "fold_id": int(fold_id),
                    "test_participant_ids": json.dumps(list(test_participants)),
                    "window_id": str(row["window_id"]),
                    "session_key": str(row["session_key"]),
                    "participant_id": int(row["participant_id"]),
                    "event_id": str(row["event_id"]),
                    "video_uid": str(row["video_uid"]),
                    "y_true": ID_TO_LABEL[int(y_test[idx])],
                    "y_pred": ID_TO_LABEL[int(y_pred[idx])],
                }
                for class_idx, label in enumerate(LABEL_ORDER):
                    out[f"proba_{label}"] = float(y_proba[idx, class_idx])
                prediction_rows.append(out)

    metrics_df = pd.DataFrame(metric_rows)
    predictions_df = pd.DataFrame(prediction_rows)
    for column in ("macro_f1", "weighted_f1", "accuracy", "mae", "within_1_accuracy", "binary_accuracy", "binary_macro_f1"):
        if column in metrics_df:
            metrics_df[column] = metrics_df[column].round(4)
    metrics_path = args.output_dir / "metrics_per_fold.csv"
    predictions_path = args.output_dir / "predictions.csv"
    manifest_path = args.output_dir / "manifest.json"
    metrics_df.to_csv(metrics_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)
    with manifest_path.open("w", encoding="utf-8") as fp:
        json.dump(
            {
                "models": ["mean", "mode", "random_distribution"],
                "split_mode": "fixed4",
                "fixed4_folds": [list(fold) for fold in FIXED_4_FOLDS],
                "num_windows": int(len(y)),
                "participants": sorted({int(v) for v in pid_values.tolist()}),
                "label_order": list(LABEL_ORDER),
                "binary_rebinning": {"low": [1, 2], "high": [3, 4, 5]},
                "notes": "Mean and mode are fold-wise constants from training labels; random samples from the training label distribution.",
            },
            fp,
            indent=2,
            sort_keys=True,
        )

    print("\nSimple comparison baseline run complete:")
    print(json.dumps({"metrics_per_fold": str(metrics_path), "predictions": str(predictions_path), "manifest": str(manifest_path)}, indent=2))
    if not metrics_df.empty:
        print(metrics_df.groupby("model")[["mae", "accuracy", "within_1_accuracy", "binary_accuracy", "binary_macro_f1"]].mean().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
