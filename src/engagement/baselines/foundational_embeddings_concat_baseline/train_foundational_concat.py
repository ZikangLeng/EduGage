#!/usr/bin/env python3
"""Strict-intersection concatenated-embedding linear baseline.

This baseline aligns all 10 embedding modalities by window_id, keeps only
windows present in every modality, concatenates the embeddings, and trains a
single linear classifier under the fixed 4-fold participant split.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[4]
BASELINE_ROOT = Path(__file__).resolve().parent


FIXED_FOLDS = (
    (1, 2, 3, 10),
    (4, 5, 7, 9),
    (6, 8, 11, 12),
    (13, 14, 15, 16),
)

MODALITIES = (
    ("ecg", "output/foundational_embeddings/ECGFounder/ecgfounder_embeddings/ecgfounder_embeddings.npz"),
    ("eye", "output/foundational_embeddings/Moment/moment_beameye_embeddings/moment_beameye_embeddings.npz"),
    ("eda", "output/foundational_embeddings/Moment/moment_msbandeda_embeddings/moment_msbandeda_embeddings.npz"),
    ("hr", "output/foundational_embeddings/Moment/moment_msbandhr_embeddings/moment_msbandhr_embeddings.npz"),
    (
        "ringtemp",
        "output/foundational_embeddings/Moment/moment_ringtemp_raw_embeddings/moment_ringtemp_raw_embeddings.npz",
    ),
    ("eeg", "output/foundational_embeddings/NeuroLM/neurolm_embeddings/neurolm_embeddings.npz"),
    (
        "esense_imu",
        "output/foundational_embeddings/NormWear/normwear_esenseimu_embeddings/normwear_esenseimu_embeddings.npz",
    ),
    (
        "muse_imu",
        "output/foundational_embeddings/NormWear/normwear_museimu_embeddings/normwear_museimu_embeddings.npz",
    ),
    (
        "muse_ppg",
        "output/foundational_embeddings/pulseppg/pulseppg_muse_embeddings/pulseppg_embeddings.npz",
    ),
    (
        "ring_ppg",
        "output/foundational_embeddings/pulseppg/pulseppg_ring_green_embeddings/pulseppg_ring_green_embeddings.npz",
    ),
)

TARGET_CLASSES = {
    "label_2class_12_low": ("low", "high"),
    "label_2class_123_low": ("low", "high"),
    "label_3class": ("1", "2", "3"),
    "label_5class": ("1", "2", "3", "4", "5"),
}

METADATA_FEATURES = ("video_progress_rounded",)
DEFAULT_METADATA_REPEAT_DIM = 512


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 42
    max_epochs: int = 500
    patience: int = 50
    batch_size: int = 64
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-2
    val_fraction: float = 0.2


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_labels(preprocessed_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(preprocessed_dir.glob("P*/*_supervised_windows.csv")):
        frame = pd.read_csv(path)
        keep = [
            "window_id",
            "participant_id",
            "session_key",
            "video_uid",
            "event_id",
            "label_3class",
            "label_5class",
            "t_start_video_sec",
            "t_end_video_sec",
        ]
        frames.append(frame[[c for c in keep if c in frame.columns]].copy())
    if not frames:
        raise FileNotFoundError(f"No supervised windows found under {preprocessed_dir}")
    labels = pd.concat(frames, ignore_index=True)
    labels["window_id"] = labels["window_id"].astype(str)
    labels["participant_id"] = pd.to_numeric(labels["participant_id"], errors="coerce")
    labels = labels.dropna(subset=["participant_id"]).copy()
    labels["participant_id"] = labels["participant_id"].astype(int)
    return labels.drop_duplicates(subset=["window_id"], keep="first")


def load_modality(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    if "window_id" not in data.files or "embeddings" not in data.files:
        raise ValueError(f"{path} must contain window_id and embeddings arrays")
    window_ids = data["window_id"].astype(str)
    embeddings = data["embeddings"].astype(np.float32)
    embeddings = np.nan_to_num(embeddings, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    embeddings = np.clip(embeddings, -1.0e4, 1.0e4)
    return window_ids, embeddings


def load_all_embeddings(root: Path) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, int], dict[str, int]]:
    embeddings: dict[str, dict[str, np.ndarray]] = {}
    dims: dict[str, int] = {}
    counts: dict[str, int] = {}
    for modality, rel_path in MODALITIES:
        path = root / rel_path
        window_ids, values = load_modality(path)
        embeddings[modality] = {wid: values[i] for i, wid in enumerate(window_ids)}
        dims[modality] = int(values.shape[1])
        counts[modality] = int(values.shape[0])
    return embeddings, dims, counts


def labels_for_target(labels: pd.DataFrame, target: str) -> pd.DataFrame:
    out = labels.copy()
    if target == "label_2class_12_low":
        source = pd.to_numeric(out["label_5class"], errors="coerce")
        out[target] = pd.Series(pd.NA, index=out.index, dtype="string")
        out.loc[source.isin([1, 2]), target] = "low"
        out.loc[source.isin([3, 4, 5]), target] = "high"
    elif target == "label_2class_123_low":
        source = pd.to_numeric(out["label_5class"], errors="coerce")
        out[target] = pd.Series(pd.NA, index=out.index, dtype="string")
        out.loc[source.isin([1, 2, 3]), target] = "low"
        out.loc[source.isin([4, 5]), target] = "high"
    else:
        out[target] = pd.to_numeric(out[target], errors="coerce")
        out = out.dropna(subset=[target]).copy()
        out[target] = out[target].round().astype(int).astype(str)

    out = out.dropna(subset=[target]).copy()
    out[target] = out[target].astype(str)
    out = out[out[target].isin(TARGET_CLASSES[target])].copy()
    out["y"] = pd.Categorical(out[target], categories=TARGET_CLASSES[target]).codes.astype(int)
    return out[out["y"] >= 0].copy()


def build_video_progress_context(labels: pd.DataFrame, *, repeat_dim: int = 1) -> np.ndarray:
    if repeat_dim <= 0:
        raise ValueError("repeat_dim must be positive")

    scalar_values = np.zeros((len(labels), 1), dtype=np.float32)
    if labels.empty or "video_uid" not in labels.columns:
        return np.repeat(scalar_values, repeat_dim, axis=1)

    grouped = labels.groupby(["session_key", "video_uid"], sort=False)
    for _, group in grouped:
        end_candidates = pd.to_numeric(group["t_end_video_sec"], errors="coerce")
        duration = float(end_candidates.max()) if not end_candidates.dropna().empty else np.nan
        if not np.isfinite(duration) or duration <= 0:
            continue

        for row_index, row in group.iterrows():
            start_v = pd.to_numeric(row.get("t_start_video_sec"), errors="coerce")
            end_v = pd.to_numeric(row.get("t_end_video_sec"), errors="coerce")
            if pd.isna(start_v) or pd.isna(end_v):
                continue
            center = 0.5 * (float(start_v) + float(end_v))
            progress = float(np.clip(center / duration, 0.0, 1.0))
            scalar_values[int(labels.index.get_loc(row_index)), 0] = float(round(progress, 1))
    return np.repeat(scalar_values, repeat_dim, axis=1)


def build_strict_dataset(
    labels: pd.DataFrame,
    embeddings: dict[str, dict[str, np.ndarray]],
    modalities: tuple[str, ...],
    *,
    metadata_repeat_dim: int = DEFAULT_METADATA_REPEAT_DIM,
) -> tuple[pd.DataFrame, np.ndarray]:
    common = set(labels["window_id"].astype(str))
    for modality in modalities:
        common &= set(embeddings[modality])
    rows = labels[labels["window_id"].isin(common)].copy()
    rows = rows.sort_values(["participant_id", "session_key", "video_uid", "event_id", "window_id"]).reset_index(drop=True)

    x_rows: list[np.ndarray] = []
    for window_id in rows["window_id"].astype(str):
        x_rows.append(np.concatenate([embeddings[modality][window_id] for modality in modalities]).astype(np.float32))
    if not x_rows:
        raise ValueError("Strict intersection produced zero rows")
    context = build_video_progress_context(rows, repeat_dim=metadata_repeat_dim)
    x = np.concatenate([np.vstack(x_rows), context], axis=1).astype(np.float32)
    return rows, x


def make_train_val_split(y_train: np.ndarray, seed: int, val_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(len(y_train))
    class_counts = np.bincount(y_train)
    min_count = int(class_counts[class_counts > 0].min()) if np.any(class_counts > 0) else 0
    if min_count < 2 or len(np.unique(y_train)) < 2:
        return indices, indices
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    train_idx, val_idx = next(splitter.split(np.zeros_like(y_train), y_train))
    return train_idx, val_idx


def class_weights(y: np.ndarray, n_classes: int) -> torch.Tensor:
    counts = np.bincount(y, minlength=n_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (n_classes * counts)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def train_linear(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    config: TrainConfig,
    n_classes: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    set_seed(config.seed)
    train_idx, val_idx = make_train_val_split(y_train, config.seed, config.val_fraction)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train).astype(np.float32)
    x_test_scaled = scaler.transform(x_test).astype(np.float32)

    x_sub = torch.tensor(x_train_scaled[train_idx], dtype=torch.float32)
    y_sub = torch.tensor(y_train[train_idx], dtype=torch.long)
    x_val = torch.tensor(x_train_scaled[val_idx], dtype=torch.float32)
    y_val = torch.tensor(y_train[val_idx], dtype=torch.long)
    x_eval = torch.tensor(x_test_scaled, dtype=torch.float32)

    loader = DataLoader(
        TensorDataset(x_sub, y_sub),
        batch_size=min(config.batch_size, len(y_sub)),
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed),
    )
    model = nn.Linear(x_train.shape[1], n_classes)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(y_train, n_classes))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_val = float("inf")
    best_epoch = 0
    bad_epochs = 0

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(x_val), y_val).item())
        if val_loss < best_val - 1.0e-5:
            best_val = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= config.patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(x_eval).numpy()
    pred = logits.argmax(axis=1)
    return pred, {"best_epoch": best_epoch, "best_val_loss": best_val, "epochs_run": epoch}


def evaluate_fold(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> dict[str, float]:
    labels = list(range(n_classes))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)),
    }


def run_target(
    target: str,
    rows: pd.DataFrame,
    x: np.ndarray,
    out_dir: Path,
    config: TrainConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    n_classes = len(TARGET_CLASSES[target])
    fold_metrics: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []

    for fold_id, test_participants in enumerate(FIXED_FOLDS, start=1):
        test_mask = rows["participant_id"].isin(test_participants).to_numpy()
        train_mask = ~test_mask
        y = rows["y"].to_numpy(dtype=int)
        x_train, y_train = x[train_mask], y[train_mask]
        x_test, y_test = x[test_mask], y[test_mask]
        if len(y_test) == 0:
            raise ValueError(f"{target} fold {fold_id} has no strict-intersection test rows")

        started = time.time()
        y_pred, train_info = train_linear(x_train, y_train, x_test, config, n_classes)
        metrics = evaluate_fold(y_test, y_pred, n_classes)
        metrics.update(
            {
                "target": target,
                "fold": fold_id,
                "test_participants": ",".join(map(str, test_participants)),
                "n_train": int(len(y_train)),
                "n_test": int(len(y_test)),
                "train_class_counts": json.dumps(np.bincount(y_train, minlength=n_classes).astype(int).tolist()),
                "test_class_counts": json.dumps(np.bincount(y_test, minlength=n_classes).astype(int).tolist()),
                "elapsed_sec": float(time.time() - started),
                **train_info,
            }
        )
        fold_metrics.append(metrics)

        test_rows = rows[test_mask].reset_index(drop=True)
        for row, true_code, pred_code in zip(test_rows.itertuples(index=False), y_test, y_pred, strict=True):
            predictions.append(
                {
                    "target": target,
                    "fold": fold_id,
                    "window_id": row.window_id,
                    "participant_id": int(row.participant_id),
                    "true_code": int(true_code),
                    "pred_code": int(pred_code),
                    "true_label": TARGET_CLASSES[target][int(true_code)],
                    "pred_label": TARGET_CLASSES[target][int(pred_code)],
                }
            )

    return fold_metrics, predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--out-dir", type=Path, default=BASELINE_ROOT / "results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--metadata-repeat-dim", type=int, default=DEFAULT_METADATA_REPEAT_DIM)
    args = parser.parse_args()

    root = args.root.resolve()
    out_dir = (root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    config = TrainConfig(seed=args.seed, max_epochs=args.max_epochs, patience=args.patience)

    labels = load_labels(root / "preprocessed_data")
    embeddings, dims, counts = load_all_embeddings(root)
    modalities = tuple(name for name, _ in MODALITIES)
    base_rows, x = build_strict_dataset(
        labels,
        embeddings,
        modalities,
        metadata_repeat_dim=args.metadata_repeat_dim,
    )
    input_dim = int(x.shape[1])

    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "root": str(root),
        "strict_intersection_rows_before_target_filter": int(len(base_rows)),
        "modalities": modalities,
        "embedding_counts": counts,
        "embedding_dims": dims,
        "metadata_features": METADATA_FEATURES,
        "metadata_dim": int(args.metadata_repeat_dim),
        "metadata_repeat_dim": int(args.metadata_repeat_dim),
        "metadata_source": "rounded normalized timestamp: round(window_center_video_sec / max_video_end_sec, 1)",
        "metadata_representation": "constant metadata channel flattened into the concatenated feature vector",
        "input_dim": input_dim,
        "folds": [list(fold) for fold in FIXED_FOLDS],
        "config": asdict(config),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    all_metrics: list[dict[str, Any]] = []
    all_predictions: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    print(f"strict_intersection_rows={len(base_rows)} input_dim={input_dim}")
    print("dims=" + json.dumps(dims, sort_keys=True))

    for target in TARGET_CLASSES:
        target_rows = labels_for_target(base_rows, target)
        target_x = x[target_rows.index.to_numpy()]
        target_rows = target_rows.reset_index(drop=True)
        print(f"target={target} n={len(target_rows)} classes={TARGET_CLASSES[target]}")
        metrics, predictions = run_target(target, target_rows, target_x, out_dir, config)
        all_metrics.extend(metrics)
        all_predictions.extend(predictions)

        mdf = pd.DataFrame(metrics)
        summary = {
            "target": target,
            "n_samples": int(len(target_rows)),
            "input_dim": input_dim,
            "accuracy": float(mdf["accuracy"].mean()),
            "balanced_accuracy": float(mdf["balanced_accuracy"].mean()),
            "macro_f1": float(mdf["macro_f1"].mean()),
            "weighted_f1": float(mdf["weighted_f1"].mean()),
            "mean_best_epoch": float(mdf["best_epoch"].mean()),
        }
        summary_rows.append(summary)
        print(
            f"  macro_f1={summary['macro_f1']:.4f} "
            f"bal_acc={summary['balanced_accuracy']:.4f} "
            f"acc={summary['accuracy']:.4f}"
        )

    pd.DataFrame(all_metrics).to_csv(out_dir / "metrics_per_fold.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(out_dir / "metrics_summary.csv", index=False)
    pd.DataFrame(all_predictions).to_csv(out_dir / "predictions.csv", index=False)
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
