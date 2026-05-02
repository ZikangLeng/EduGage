from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import pickle
import sys
from typing import Any
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, RobustScaler
from sklearn.svm import SVR

REPO_ROOT = Path(__file__).resolve().parents[4]
BASELINE_ROOT = Path(__file__).resolve().parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from engagement.architecture_data import (  # noqa: E402
    CHANNELS,
    _metadata_progress,
    _normalize_columns,
    _read_csv,
    _session_paths,
)
from engagement.train_eval import ID_TO_LABEL, LABEL_ORDER, LABEL_TO_ID  # noqa: E402

LOGGER = logging.getLogger(__name__)

FIXED_4_FOLDS: tuple[tuple[int, ...], ...] = (
    (1, 2, 3, 10),
    (4, 5, 7, 9),
    (6, 8, 11, 12),
    (13, 14, 15, 16),
)

FEATURE_STAT_NAMES = (
    "mean",
    "std",
    "kurtosis",
    "min",
    "max",
    "iqr",
    "diff_std",
    "slope",
)

FEATURE_CLIP_VALUE = 1_000_000.0

NATIVE_CHANNEL_SPECS: tuple[tuple[str, str, str], ...] = (
    ("beam:gaze_por_x", "beam", "gaze_por_x"),
    ("beam:gaze_por_y", "beam", "gaze_por_y"),
    ("beam:head_pos_x_m", "beam", "head_pos_x_m"),
    ("beam:head_pos_y_m", "beam", "head_pos_y_m"),
    ("beam:head_pos_z_m", "beam", "head_pos_z_m"),
    ("bandeda:Resistance_kOhms", "bandeda", "Resistance_kOhms"),
    ("bandhr:HeartRate_bpm", "bandhr", "HeartRate_bpm"),
    ("museppg:ppg1", "museppg", "ppg1"),
    ("ringppg:green", "ring", "green"),
    ("ringtemp:temp_0", "ring", "temp_0"),
    ("ringtemp:temp_1", "ring", "temp_1"),
    ("ringtemp:temp_2", "ring", "temp_2"),
    ("esenseimu:Accel_X_g", "esense", "Accel_X_g"),
    ("esenseimu:Accel_Y_g", "esense", "Accel_Y_g"),
    ("esenseimu:Accel_Z_g", "esense", "Accel_Z_g"),
    ("esenseimu:Gyro_X_deg_per_s", "esense", "Gyro_X_deg_per_s"),
    ("esenseimu:Gyro_Y_deg_per_s", "esense", "Gyro_Y_deg_per_s"),
    ("esenseimu:Gyro_Z_deg_per_s", "esense", "Gyro_Z_deg_per_s"),
    ("ecg:ecg_val", "ecg", "ecg_val"),
    ("museimu:acc_x", "acc", "x"),
    ("museimu:acc_y", "acc", "y"),
    ("museimu:acc_z", "acc", "z"),
    ("museimu:gyro_x", "gyro", "x"),
    ("museimu:gyro_y", "gyro", "y"),
    ("museimu:gyro_z", "gyro", "z"),
    ("museeeg:af7", "eeg", "af7"),
    ("museeeg:af8", "eeg", "af8"),
)

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train tabular all-modality ML baselines.")
    parser.add_argument("--preprocessed-root", type=Path, default=REPO_ROOT / "preprocessed_data")
    parser.add_argument("--output-dir", type=Path, default=BASELINE_ROOT / "results" / "ml_baselines_fixed4")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["random_forest", "linear_regression", "lgbm", "svm"],
        choices=["random_forest", "linear_regression", "lgbm", "svm"],
    )
    parser.add_argument(
        "--split-mode",
        default="fixed4",
        choices=["loso", "fixed4", "both"],
        help="Evaluation split policy. fixed4 is the requested four participant folds.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-windows", type=int, default=0, help="Optional smoke-test cap after loading.")
    parser.add_argument("--rf-trees", type=int, default=500)
    parser.add_argument("--lgbm-trees", type=int, default=500)
    parser.add_argument("--linear-alpha", type=float, default=1.0, help="L2 stabilization for the linear-regression baseline.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def _safe_slope(values: np.ndarray) -> np.ndarray:
    if values.shape[2] < 2:
        return np.zeros((values.shape[0], values.shape[1]), dtype=np.float32)
    first = values[:, :, 0]
    last = values[:, :, -1]
    denom = max(1, values.shape[2] - 1)
    return ((last - first) / denom).astype(np.float32)


def _safe_kurtosis(values: np.ndarray) -> np.ndarray:
    mean = np.mean(values, axis=2, keepdims=True)
    centered = values - mean
    var = np.mean(centered * centered, axis=2)
    fourth = np.mean(centered**4, axis=2)
    denom = var * var
    kurtosis = np.divide(fourth, denom, out=np.zeros_like(fourth), where=denom > 1e-12)
    return kurtosis - 3.0


def extract_tabular_features(x: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Summarize [N, C, T] all-modality windows into tabular channel stats."""
    if x.ndim != 3:
        raise ValueError("x must have shape [N, C, T].")
    x = np.asarray(x, dtype=np.float32)
    diff = np.diff(x, axis=2) if x.shape[2] > 1 else np.zeros_like(x)

    stats = [
        np.mean(x, axis=2),
        np.std(x, axis=2),
        _safe_kurtosis(x),
        np.min(x, axis=2),
        np.max(x, axis=2),
        np.percentile(x, 75, axis=2) - np.percentile(x, 25, axis=2),
        np.std(diff, axis=2),
        _safe_slope(x),
    ]
    features = np.concatenate([np.asarray(v, dtype=np.float64) for v in stats], axis=1)
    features = np.nan_to_num(features, nan=0.0, posinf=FEATURE_CLIP_VALUE, neginf=-FEATURE_CLIP_VALUE)
    features = np.clip(features, -FEATURE_CLIP_VALUE, FEATURE_CLIP_VALUE)

    names: list[str] = []
    for stat_name in FEATURE_STAT_NAMES:
        for channel in CHANNELS:
            safe_channel = channel.replace(":", "__")
            names.append(f"{safe_channel}__{stat_name}")
    return features.astype(np.float64), names


def _empty_channel_stats() -> np.ndarray:
    return np.zeros((len(FEATURE_STAT_NAMES),), dtype=np.float64)


def _native_channel_stats(df: pd.DataFrame, column: str, *, start_sec: float, end_sec: float) -> np.ndarray:
    if df.empty or "t_sec" not in df.columns or column not in df.columns:
        return _empty_channel_stats()

    t = pd.to_numeric(df["t_sec"], errors="coerce").to_numpy(dtype=float)
    v = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(t) & np.isfinite(v) & (t >= start_sec) & (t < end_sec)
    if not np.any(valid):
        return _empty_channel_stats()

    t = t[valid]
    v = v[valid]
    order = np.argsort(t, kind="mergesort")
    t = t[order]
    v = v[order]
    if v.size == 0:
        return _empty_channel_stats()

    diff = np.diff(v) if v.size > 1 else np.asarray([], dtype=float)
    mean = float(np.mean(v))
    centered = v - mean
    var = float(np.mean(centered * centered))
    fourth = float(np.mean(centered**4))
    kurtosis = (fourth / (var * var) - 3.0) if var > 1e-12 else 0.0
    p25 = float(np.percentile(v, 25))
    p75 = float(np.percentile(v, 75))
    duration = float(t[-1] - t[0]) if t.size > 1 else 0.0
    slope = float((v[-1] - v[0]) / duration) if duration > 1e-9 else 0.0

    stats = np.asarray(
        [
            mean,
            float(np.std(v)),
            kurtosis,
            float(np.min(v)),
            float(np.max(v)),
            p75 - p25,
            float(np.std(diff)) if diff.size else 0.0,
            slope,
        ],
        dtype=np.float64,
    )
    stats = np.nan_to_num(stats, nan=0.0, posinf=FEATURE_CLIP_VALUE, neginf=-FEATURE_CLIP_VALUE)
    return np.clip(stats, -FEATURE_CLIP_VALUE, FEATURE_CLIP_VALUE)


def build_native_tabular_dataset_from_preprocessed(preprocessed_root: Path) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, list[str]]:
    feature_rows: list[np.ndarray] = []
    y_rows: list[int] = []
    meta_rows: list[dict[str, Any]] = []
    feature_names: list[str] = []
    for channel_name, _stream_name, _column_name in NATIVE_CHANNEL_SPECS:
        safe_channel = channel_name.replace(":", "__")
        feature_names.extend([f"{safe_channel}__{stat_name}" for stat_name in FEATURE_STAT_NAMES])
    feature_names.append("metadata__video_progress_rounded")

    for windows_path in sorted(preprocessed_root.glob("P*/*_supervised_windows.csv")):
        session_windows = pd.read_csv(windows_path)
        if session_windows.empty:
            continue
        paths = _session_paths(windows_path)
        streams = {name: _normalize_columns(_read_csv(path)) for name, path in paths.items()}

        for row in session_windows.to_dict(orient="records"):
            label_value = pd.to_numeric(row.get("label_5class", row.get("label_3class")), errors="coerce")
            start = pd.to_numeric(row.get("t_start_sec"), errors="coerce")
            end = pd.to_numeric(row.get("t_end_sec"), errors="coerce")
            if pd.isna(label_value) or pd.isna(start) or pd.isna(end) or float(end) <= float(start):
                continue
            label_name = str(int(label_value))
            if label_name not in LABEL_TO_ID:
                continue

            channel_stats = [
                _native_channel_stats(streams[stream_name], column_name, start_sec=float(start), end_sec=float(end))
                for _channel_name, stream_name, column_name in NATIVE_CHANNEL_SPECS
            ]
            progress = np.asarray([_metadata_progress(row, session_windows=session_windows)], dtype=np.float64)
            feature_rows.append(np.concatenate([*channel_stats, progress], axis=0))
            y_rows.append(LABEL_TO_ID[label_name])
            meta_rows.append(
                {
                    "window_id": str(row.get("window_id", "")),
                    "session_key": str(row.get("session_key", "")),
                    "participant_id": int(pd.to_numeric(row.get("participant_id"), errors="coerce")),
                    "event_id": str(row.get("event_id", "")),
                    "video_uid": str(row.get("video_uid", "")),
                    "label_5class": label_name,
                    "metadata_progress": float(progress[0]),
                }
            )

    if not feature_rows:
        raise ValueError(f"No native tabular windows found under {preprocessed_root}.")

    features = np.vstack(feature_rows).astype(np.float64)
    features = np.nan_to_num(features, nan=0.0, posinf=FEATURE_CLIP_VALUE, neginf=-FEATURE_CLIP_VALUE)
    features = np.clip(features, -FEATURE_CLIP_VALUE, FEATURE_CLIP_VALUE)
    if features.shape[1] != len(feature_names):
        raise ValueError(f"Feature width mismatch: got {features.shape[1]}, expected {len(feature_names)}.")
    return pd.DataFrame(features, columns=feature_names), np.asarray(y_rows, dtype=np.int64), pd.DataFrame(meta_rows), feature_names


def _build_split_specs(participants: list[int], split_mode: str) -> list[dict[str, Any]]:
    present = set(int(v) for v in participants)
    specs: list[dict[str, Any]] = []
    if split_mode in {"loso", "both"}:
        for fold_id, test_pid in enumerate(participants):
            specs.append({"split_name": "loso", "fold_id": int(fold_id), "test_participants": (int(test_pid),)})
    if split_mode in {"fixed4", "both"}:
        for fold_id, fold_participants in enumerate(FIXED_4_FOLDS):
            available = tuple(pid for pid in fold_participants if pid in present)
            if available:
                specs.append({"split_name": "fixed4", "fold_id": int(fold_id), "test_participants": available})
    return specs


def _clip_scaled_features(x: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(x, dtype=float), -10.0, 10.0)


def _build_model(name: str, *, seed: int, rf_trees: int, lgbm_trees: int, linear_alpha: float) -> Any:
    if name == "random_forest":
        return RandomForestRegressor(
            n_estimators=int(rf_trees),
            random_state=int(seed),
            n_jobs=-1,
            min_samples_leaf=2,
        )
    if name == "linear_regression":
        return make_pipeline(
            RobustScaler(),
            FunctionTransformer(_clip_scaled_features, validate=False),
            Ridge(alpha=float(linear_alpha), random_state=int(seed), solver="lsqr"),
        )
    if name == "svm":
        return make_pipeline(
            RobustScaler(),
            FunctionTransformer(_clip_scaled_features, validate=False),
            SVR(
                C=1.0,
                kernel="rbf",
                gamma="scale",
            ),
        )
    if name == "lgbm":
        try:
            from lightgbm import LGBMRegressor
        except ImportError as exc:
            raise ImportError("Install LightGBM first: pip install lightgbm") from exc
        return LGBMRegressor(
            objective="regression",
            n_estimators=int(lgbm_trees),
            learning_rate=0.03,
            num_leaves=15,
            min_child_samples=10,
            subsample=0.9,
            colsample_bytree=0.8,
            random_state=int(seed),
            n_jobs=-1,
            verbosity=-1,
        )
    raise ValueError(f"Unsupported ML baseline model: {name}")


def _predict_raw(model: Any, x_test: Any) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", message="X does not have valid feature names.*", category=UserWarning)
        raw = model.predict(x_test)
    return np.nan_to_num(
        np.asarray(raw, dtype=float).reshape(-1),
        nan=2.0,
        posinf=float(len(LABEL_ORDER) - 1),
        neginf=0.0,
    )


def _round_regression_predictions(raw: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(raw), 0, len(LABEL_ORDER) - 1).astype(int)


def _within_one_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true.astype(int) - y_pred.astype(int)) <= 1))


def _to_low_high(y: np.ndarray) -> np.ndarray:
    """Map ordinal ids 0,1 to low and 2,3,4 to high."""
    return (np.asarray(y, dtype=int) >= 2).astype(int)


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "models").mkdir(parents=True, exist_ok=True)

    feature_table, y, metadata, feature_names = build_native_tabular_dataset_from_preprocessed(args.preprocessed_root)
    if int(args.max_windows) > 0:
        feature_table = feature_table.iloc[: int(args.max_windows)].reset_index(drop=True)
        y = y[: int(args.max_windows)]
        metadata = metadata.iloc[: int(args.max_windows)].reset_index(drop=True)
    participants = sorted({int(v) for v in metadata["participant_id"].tolist()})
    split_specs = _build_split_specs(participants, str(args.split_mode))
    if not split_specs:
        raise ValueError(f"No evaluation folds are available for split mode {args.split_mode!r}.")

    LOGGER.info(
        "Loaded %s windows into tabular all-modality feature matrix %s.",
        len(feature_table),
        feature_table.shape,
    )

    pid_values = metadata["participant_id"].astype(int).to_numpy()
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []

    for model_name in args.models:
        for split_spec in split_specs:
            split_name = str(split_spec["split_name"])
            fold_id = int(split_spec["fold_id"])
            test_participants = tuple(int(v) for v in split_spec["test_participants"])
            test_mask = np.isin(pid_values, test_participants)
            train_mask = ~test_mask
            if not np.any(train_mask) or not np.any(test_mask):
                continue

            model = _build_model(
                model_name,
                seed=int(args.seed) + fold_id,
                rf_trees=int(args.rf_trees),
                lgbm_trees=int(args.lgbm_trees),
                linear_alpha=float(args.linear_alpha),
            )
            model.fit(feature_table.loc[train_mask], y[train_mask])
            y_pred_raw = _predict_raw(model, feature_table.loc[test_mask])
            y_pred = _round_regression_predictions(y_pred_raw)
            y_test = y[test_mask]

            accuracy = float(accuracy_score(y_test, y_pred))
            mae = float(mean_absolute_error(y_test, y_pred))
            raw_mae = float(mean_absolute_error(y_test, y_pred_raw))
            within_1_accuracy = _within_one_accuracy(y_test, y_pred)
            y_test_binary = _to_low_high(y_test)
            y_pred_binary = _to_low_high(y_pred)
            binary_accuracy = float(accuracy_score(y_test_binary, y_pred_binary))
            binary_macro_f1 = float(f1_score(y_test_binary, y_pred_binary, average="macro", zero_division=0))
            test_participant_tag = "-".join(str(v) for v in test_participants)

            metric_rows.append(
                {
                    "split_name": split_name,
                    "model": model_name,
                    "fold_id": fold_id,
                    "test_participant_ids": json.dumps(list(test_participants)),
                    "accuracy": accuracy,
                    "within_1_accuracy": within_1_accuracy,
                    "binary_accuracy": binary_accuracy,
                    "binary_macro_f1": binary_macro_f1,
                    "mae": mae,
                    "raw_mae": raw_mae,
                    "num_samples": int(len(y_test)),
                }
            )

            model_path = args.output_dir / "models" / f"{model_name}__{split_name}_fold_{fold_id}__participants_{test_participant_tag}.pkl"
            with model_path.open("wb") as fp:
                pickle.dump({"model": model, "feature_names": feature_names, "channels": list(CHANNELS)}, fp)

            fold_meta = metadata.loc[test_mask].reset_index(drop=True)
            for idx, row in fold_meta.iterrows():
                out = {
                    "split_name": split_name,
                    "model": model_name,
                    "fold_id": fold_id,
                    "test_participant_ids": json.dumps(list(test_participants)),
                    "window_id": str(row["window_id"]),
                    "session_key": str(row["session_key"]),
                    "participant_id": int(row["participant_id"]),
                    "event_id": str(row["event_id"]),
                    "video_uid": str(row["video_uid"]),
                    "y_true": ID_TO_LABEL[int(y_test[idx])],
                    "y_pred": ID_TO_LABEL[int(y_pred[idx])],
                    "y_pred_raw": float(y_pred_raw[idx] + 1.0),
                    "y_true_binary": "high" if int(y_test_binary[idx]) else "low",
                    "y_pred_binary": "high" if int(y_pred_binary[idx]) else "low",
                }
                prediction_rows.append(out)

    metrics_df = pd.DataFrame(metric_rows)
    predictions_df = pd.DataFrame(prediction_rows)
    for column in ("accuracy", "within_1_accuracy", "binary_accuracy", "binary_macro_f1", "mae", "raw_mae"):
        if column in metrics_df:
            metrics_df[column] = metrics_df[column].round(4)
    if "y_pred_raw" in predictions_df:
        predictions_df["y_pred_raw"] = predictions_df["y_pred_raw"].round(4)
    metrics_path = args.output_dir / "metrics_per_fold.csv"
    predictions_path = args.output_dir / "predictions.csv"
    manifest_path = args.output_dir / "manifest.json"
    metrics_df.to_csv(metrics_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)
    with manifest_path.open("w", encoding="utf-8") as fp:
        json.dump(
            {
                "models": list(args.models),
                "split_mode": str(args.split_mode),
                "fixed4_folds": [list(fold) for fold in FIXED_4_FOLDS],
                "input_representation": "native_sampling_rate_statistical_features",
                "feature_source": "Native sampling rate preprocessed sensor samples inside each labeled window.",
                "channels": [channel_name for channel_name, _stream_name, _column_name in NATIVE_CHANNEL_SPECS]
                + ["metadata:video_progress_rounded"],
                "feature_stats": list(FEATURE_STAT_NAMES),
                "num_features": int(feature_table.shape[1]),
                "feature_clip_value": float(FEATURE_CLIP_VALUE),
                "linear_regression_impl": "sklearn.linear_model.Ridge",
                "linear_alpha": float(args.linear_alpha),
                "target_type": "regression_0_to_4_rounded_clipped_to_labels_1_to_5",
                "binary_rebinning": {"low": [1, 2], "high": [3, 4, 5]},
                "metrics": ["mae", "raw_mae", "accuracy", "within_1_accuracy", "binary_accuracy", "binary_macro_f1"],
                "num_windows": int(len(feature_table)),
                "participants": participants,
            },
            fp,
            indent=2,
            sort_keys=True,
        )

    print("\nML baseline run complete:")
    print(json.dumps({"metrics_per_fold": str(metrics_path), "predictions": str(predictions_path), "manifest": str(manifest_path)}, indent=2))
    if not metrics_df.empty:
        summary_columns = ["mae", "raw_mae", "accuracy", "within_1_accuracy", "binary_accuracy", "binary_macro_f1"]
        print(metrics_df.groupby(["split_name", "model"])[summary_columns].mean().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
