#!/usr/bin/env python3
"""Strict-intersection gated-fusion baseline for foundation embeddings.

This baseline uses the same rows, folds, labels, embedding streams, and repeated
metadata channel as the simple concatenation runner. Each sensor/metadata input
is projected into a shared latent space, gated, and fused with a normalized
weighted average.
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

try:
    from train_foundational_concat import (
        BASELINE_ROOT,
        REPO_ROOT,
        DEFAULT_METADATA_REPEAT_DIM,
        FIXED_FOLDS,
        METADATA_FEATURES,
        MODALITIES,
        TARGET_CLASSES,
        build_video_progress_context,
        labels_for_target,
        load_all_embeddings,
        load_labels,
    )
except ModuleNotFoundError:
    from .train_foundational_concat import (
        BASELINE_ROOT,
        REPO_ROOT,
        DEFAULT_METADATA_REPEAT_DIM,
        FIXED_FOLDS,
        METADATA_FEATURES,
        MODALITIES,
        TARGET_CLASSES,
        build_video_progress_context,
        labels_for_target,
        load_all_embeddings,
        load_labels,
    )


METADATA_MODALITY = "metadata"


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 42
    max_epochs: int = 500
    patience: int = 50
    batch_size: int = 64
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-2
    val_fraction: float = 0.2
    embedding_dim: int = 128
    fusion_hidden_dim: int = 256
    dropout: float = 0.2


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_strict_dataset(
    labels: pd.DataFrame,
    embeddings: dict[str, dict[str, np.ndarray]],
    modalities: tuple[str, ...],
    *,
    metadata_repeat_dim: int = DEFAULT_METADATA_REPEAT_DIM,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], np.ndarray]:
    common = set(labels["window_id"].astype(str))
    for modality in modalities:
        common &= set(embeddings[modality])
    rows = labels[labels["window_id"].isin(common)].copy()
    rows = rows.sort_values(["participant_id", "session_key", "video_uid", "event_id", "window_id"]).reset_index(drop=True)

    if rows.empty:
        raise ValueError("Strict intersection produced zero rows")

    modality_arrays: dict[str, np.ndarray] = {}
    for modality in modalities:
        modality_arrays[modality] = np.vstack(
            [embeddings[modality][window_id] for window_id in rows["window_id"].astype(str)]
        ).astype(np.float32)
    modality_arrays[METADATA_MODALITY] = build_video_progress_context(
        rows,
        repeat_dim=metadata_repeat_dim,
    )
    modality_mask = np.ones((len(rows), len(modalities) + 1), dtype=np.float32)
    return rows, modality_arrays, modality_mask


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


class FoundationEmbeddingGatedFusionModel(nn.Module):
    def __init__(
        self,
        *,
        modality_dims: dict[str, int],
        modality_order: tuple[str, ...],
        embedding_dim: int,
        fusion_hidden_dim: int,
        n_classes: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.modality_order = modality_order
        self.embedding_dim = int(embedding_dim)

        self.projectors = nn.ModuleDict()
        self.gates = nn.ModuleDict()
        for modality in modality_order:
            input_dim = int(modality_dims[modality])
            self.projectors[modality] = nn.Sequential(
                nn.Linear(input_dim, self.embedding_dim),
                nn.LayerNorm(self.embedding_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.gates[modality] = nn.Sequential(
                nn.Linear(self.embedding_dim, fusion_hidden_dim),
                nn.GELU(),
                nn.Linear(fusion_hidden_dim, 1),
            )

        self.classifier = nn.Sequential(
            nn.Linear(self.embedding_dim, fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, n_classes),
        )

    def forward(
        self,
        modality_inputs: dict[str, torch.Tensor],
        modality_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        encoded_modalities: list[torch.Tensor] = []
        gate_values: list[torch.Tensor] = []

        for idx, modality in enumerate(self.modality_order):
            z_m = self.projectors[modality](modality_inputs[modality])
            gate_logits = self.gates[modality](z_m)
            gate = torch.sigmoid(gate_logits) * modality_mask[:, idx : idx + 1]
            encoded_modalities.append(z_m)
            gate_values.append(gate)

        stacked_embeddings = torch.stack(encoded_modalities, dim=1)
        stacked_gates = torch.stack(gate_values, dim=1)
        denom = torch.clamp(stacked_gates.sum(dim=1), min=1.0e-6)
        fused = (stacked_embeddings * stacked_gates).sum(dim=1) / denom
        logits = self.classifier(fused)
        return {
            "fused_embedding": fused,
            "gates": stacked_gates.squeeze(-1),
            "logits": logits,
        }


def _scale_by_modality(
    train_inputs: dict[str, np.ndarray],
    test_inputs: dict[str, np.ndarray],
    train_indices: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    scaled_train: dict[str, np.ndarray] = {}
    scaled_test: dict[str, np.ndarray] = {}
    for modality, x_train in train_inputs.items():
        scaler = StandardScaler()
        scaler.fit(x_train[train_indices])
        scaled_train[modality] = scaler.transform(x_train).astype(np.float32)
        scaled_test[modality] = scaler.transform(test_inputs[modality]).astype(np.float32)
    return scaled_train, scaled_test


def train_gated(
    x_train_by_modality: dict[str, np.ndarray],
    train_mask: np.ndarray,
    y_train: np.ndarray,
    x_test_by_modality: dict[str, np.ndarray],
    test_mask: np.ndarray,
    config: TrainConfig,
    n_classes: int,
    modality_dims: dict[str, int],
    modalities: tuple[str, ...],
) -> tuple[np.ndarray, dict[str, Any], np.ndarray]:
    set_seed(config.seed)
    train_idx, val_idx = make_train_val_split(y_train, config.seed, config.val_fraction)
    scaled_train, scaled_test = _scale_by_modality(x_train_by_modality, x_test_by_modality, train_idx)

    train_tensors = [torch.tensor(scaled_train[m][train_idx], dtype=torch.float32) for m in modalities]
    val_tensors = [torch.tensor(scaled_train[m][val_idx], dtype=torch.float32) for m in modalities]
    test_tensors = {m: torch.tensor(scaled_test[m], dtype=torch.float32) for m in modalities}
    sub_mask = torch.tensor(train_mask[train_idx], dtype=torch.float32)
    val_mask = torch.tensor(train_mask[val_idx], dtype=torch.float32)
    eval_mask = torch.tensor(test_mask, dtype=torch.float32)
    y_sub = torch.tensor(y_train[train_idx], dtype=torch.long)
    y_val = torch.tensor(y_train[val_idx], dtype=torch.long)

    loader = DataLoader(
        TensorDataset(*train_tensors, sub_mask, y_sub),
        batch_size=min(config.batch_size, len(y_sub)),
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed),
    )
    model = FoundationEmbeddingGatedFusionModel(
        modality_dims=modality_dims,
        modality_order=modalities,
        embedding_dim=config.embedding_dim,
        fusion_hidden_dim=config.fusion_hidden_dim,
        n_classes=n_classes,
        dropout=config.dropout,
    )
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(y_train, n_classes))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_val = float("inf")
    best_epoch = 0
    bad_epochs = 0

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        for batch in loader:
            xs = {m: batch[i] for i, m in enumerate(modalities)}
            mask = batch[len(modalities)]
            yb = batch[len(modalities) + 1]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xs, mask)["logits"], yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_inputs = {m: val_tensors[i] for i, m in enumerate(modalities)}
            val_loss = float(loss_fn(model(val_inputs, val_mask)["logits"], y_val).item())
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
        outputs = model(test_tensors, eval_mask)
        logits = outputs["logits"].numpy()
        gates = outputs["gates"].numpy()
    pred = logits.argmax(axis=1)
    return pred, {"best_epoch": best_epoch, "best_val_loss": best_val, "epochs_run": epoch}, gates


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
    x_by_modality: dict[str, np.ndarray],
    modality_mask: np.ndarray,
    config: TrainConfig,
    modality_dims: dict[str, int],
    modalities: tuple[str, ...],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    n_classes = len(TARGET_CLASSES[target])
    fold_metrics: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []

    for fold_id, test_participants in enumerate(FIXED_FOLDS, start=1):
        test_row_mask = rows["participant_id"].isin(test_participants).to_numpy()
        train_row_mask = ~test_row_mask
        y = rows["y"].to_numpy(dtype=int)
        y_train = y[train_row_mask]
        y_test = y[test_row_mask]
        if len(y_test) == 0:
            raise ValueError(f"{target} fold {fold_id} has no strict-intersection test rows")

        x_train_by_modality = {m: x_by_modality[m][train_row_mask] for m in modalities}
        x_test_by_modality = {m: x_by_modality[m][test_row_mask] for m in modalities}
        started = time.time()
        y_pred, train_info, gates = train_gated(
            x_train_by_modality=x_train_by_modality,
            train_mask=modality_mask[train_row_mask],
            y_train=y_train,
            x_test_by_modality=x_test_by_modality,
            test_mask=modality_mask[test_row_mask],
            config=config,
            n_classes=n_classes,
            modality_dims=modality_dims,
            modalities=modalities,
        )
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

        test_rows = rows[test_row_mask].reset_index(drop=True)
        for row_idx, (row, true_code, pred_code) in enumerate(zip(test_rows.itertuples(index=False), y_test, y_pred, strict=True)):
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
            gate_record = {
                "target": target,
                "fold": fold_id,
                "window_id": row.window_id,
                "participant_id": int(row.participant_id),
            }
            gate_record.update({f"gate_{m}": float(gates[row_idx, i]) for i, m in enumerate(modalities)})
            gate_rows.append(gate_record)

    return fold_metrics, predictions, gate_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--out-dir", type=Path, default=BASELINE_ROOT / "gated_results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--fusion-hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--metadata-repeat-dim", type=int, default=DEFAULT_METADATA_REPEAT_DIM)
    args = parser.parse_args()

    root = args.root.resolve()
    out_dir = (root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    config = TrainConfig(
        seed=args.seed,
        max_epochs=args.max_epochs,
        patience=args.patience,
        embedding_dim=args.embedding_dim,
        fusion_hidden_dim=args.fusion_hidden_dim,
        dropout=args.dropout,
    )

    labels = load_labels(root / "preprocessed_data")
    embeddings, dims, counts = load_all_embeddings(root)
    sensor_modalities = tuple(name for name, _ in MODALITIES)
    base_rows, x_by_modality, modality_mask = build_strict_dataset(
        labels,
        embeddings,
        sensor_modalities,
        metadata_repeat_dim=args.metadata_repeat_dim,
    )
    modalities = sensor_modalities + (METADATA_MODALITY,)
    dims = {**dims, METADATA_MODALITY: int(args.metadata_repeat_dim)}
    counts = {**counts, METADATA_MODALITY: int(len(base_rows))}

    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "root": str(root),
        "model": "foundation_embedding_gated_fusion_with_metadata_modality",
        "strict_intersection_rows_before_target_filter": int(len(base_rows)),
        "modalities": modalities,
        "sensor_modalities": sensor_modalities,
        "embedding_counts": counts,
        "embedding_dims": dims,
        "shared_embedding_dim": int(config.embedding_dim),
        "metadata_modality": {
            "name": METADATA_MODALITY,
            "dim": int(args.metadata_repeat_dim),
            "features": list(METADATA_FEATURES),
            "source": "rounded normalized timestamp: round(window_center_video_sec / max_video_end_sec, 1)",
            "usage": "constant metadata channel repeated per window, projected and gated like a sensor modality",
        },
        "folds": [list(fold) for fold in FIXED_FOLDS],
        "config": asdict(config),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    all_metrics: list[dict[str, Any]] = []
    all_predictions: list[dict[str, Any]] = []
    all_gates: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    print(f"strict_intersection_rows={len(base_rows)}")
    print("dims=" + json.dumps(dims, sort_keys=True))
    print("metadata_modality=" + json.dumps(manifest["metadata_modality"]))

    for target in TARGET_CLASSES:
        target_rows = labels_for_target(base_rows, target)
        target_idx = target_rows.index.to_numpy()
        target_x_by_modality = {m: x_by_modality[m][target_idx] for m in modalities}
        target_mask = modality_mask[target_idx]
        target_rows = target_rows.reset_index(drop=True)
        print(f"target={target} n={len(target_rows)} classes={TARGET_CLASSES[target]}")
        metrics, predictions, gates = run_target(
            target,
            target_rows,
            target_x_by_modality,
            target_mask,
            config,
            dims,
            modalities,
        )
        all_metrics.extend(metrics)
        all_predictions.extend(predictions)
        all_gates.extend(gates)

        mdf = pd.DataFrame(metrics)
        summary = {
            "target": target,
            "n_samples": int(len(target_rows)),
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
    pd.DataFrame(all_gates).to_csv(out_dir / "gates.csv", index=False)
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
