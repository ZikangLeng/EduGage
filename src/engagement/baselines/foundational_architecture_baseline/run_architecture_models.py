from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[4]
BASELINE_ROOT = Path(__file__).resolve().parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from engagement.architecture_data import (  # noqa: E402
    CHANNELS,
    TARGET_HZ,
    TARGET_POINTS,
    build_architecture_dataset_from_preprocessed,
)
from engagement.architecture_models import (  # noqa: E402
    ArchitectureName,
    OFFICIAL_ARCHITECTURE_SOURCES,
    build_architecture_model,
)
from engagement.train_eval import ID_TO_LABEL, LABEL_ORDER  # noqa: E402

LOGGER = logging.getLogger(__name__)

FIXED_4_FOLDS: tuple[tuple[int, ...], ...] = (
    (1, 2, 3, 10),
    (4, 5, 7, 9),
    (6, 8, 11, 12),
    (13, 14, 15, 16),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train raw-window architecture models as scalar regressors.")
    parser.add_argument("--preprocessed-root", type=Path, default=REPO_ROOT / "preprocessed_data")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASELINE_ROOT / "results" / "architecture_models",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["deepconvlstm", "deepconvlstm_attention", "tinyhar"],
        choices=["deepconvlstm", "deepconvlstm_attention", "tinyhar"],
    )
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument(
        "--split-mode",
        default="fixed4",
        choices=["loso", "fixed4", "both"],
        help="Evaluation split policy. fixed4 is the requested four participant folds.",
    )
    parser.add_argument("--max-windows", type=int, default=0, help="Optional smoke-test cap after loading.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def _choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _standardize(train_x: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=(0, 2), keepdims=True)
    std = train_x.std(axis=(0, 2), keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return ((x - mean) / std).astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def _make_loader(x: np.ndarray, y: np.ndarray, *, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(x.astype(np.float32)), torch.from_numpy(y.astype(np.float32)))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def _train_validation_indices(y: np.ndarray, seed: int, validation_fraction: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []

    for class_id in sorted(set(int(v) for v in y.tolist())):
        idx = np.where(y == class_id)[0]
        rng.shuffle(idx)
        if idx.size >= 5:
            n_val = max(1, int(round(idx.size * validation_fraction)))
            val_parts.append(idx[:n_val])
            train_parts.append(idx[n_val:])
        else:
            train_parts.append(idx)

    train_idx = np.concatenate(train_parts) if train_parts else np.arange(len(y))
    val_idx = np.concatenate(val_parts) if val_parts else train_idx.copy()
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx.astype(int), val_idx.astype(int)


def _build_split_specs(participants: list[int], split_mode: str) -> list[dict[str, Any]]:
    present = set(int(v) for v in participants)
    specs: list[dict[str, Any]] = []

    if split_mode in {"loso", "both"}:
        for fold_id, test_pid in enumerate(participants):
            specs.append(
                {
                    "split_name": "loso",
                    "fold_id": int(fold_id),
                    "test_participants": (int(test_pid),),
                }
            )

    if split_mode in {"fixed4", "both"}:
        for fold_id, fold_participants in enumerate(FIXED_4_FOLDS):
            available = tuple(pid for pid in fold_participants if pid in present)
            if available:
                specs.append(
                    {
                        "split_name": "fixed4",
                        "fold_id": int(fold_id),
                        "test_participants": available,
                    }
                )
    return specs


def _round_regression_predictions(raw: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(raw), 0, len(LABEL_ORDER) - 1).astype(np.int64)


def _to_low_high(y: np.ndarray) -> np.ndarray:
    return (np.asarray(y, dtype=int) >= 2).astype(int)


def _predict(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    pred_rows: list[np.ndarray] = []
    raw_rows: list[np.ndarray] = []
    with torch.inference_mode():
        for xb, _yb in loader:
            raw = model(xb.to(device=device)).detach().cpu().numpy().reshape(-1)
            raw = np.nan_to_num(raw, nan=2.0, posinf=float(len(LABEL_ORDER) - 1), neginf=0.0)
            raw_rows.append(raw.astype(float))
            pred_rows.append(_round_regression_predictions(raw))
    return np.concatenate(pred_rows, axis=0), np.concatenate(raw_rows, axis=0)


def _train_one_fold(
    *,
    model_name: ArchitectureName,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    checkpoint_path: Path,
) -> dict[str, Any]:
    model = build_architecture_model(
        model_name,
        in_channels=x_train.shape[1],
        n_classes=2,
        input_points=x_train.shape[2],
        dropout=float(args.dropout),
    ).to(device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    train_loader = _make_loader(x_train, y_train, batch_size=int(args.batch_size), shuffle=True)
    val_loader = _make_loader(x_val, y_val, batch_size=int(args.batch_size), shuffle=False)
    eval_loader = _make_loader(x_test, y_test, batch_size=int(args.batch_size), shuffle=False)

    best_state = None
    best_loss = float("inf")
    stale_epochs = 0
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        losses: list[float] = []
        for xb, yb in train_loader:
            xb = xb.to(device=device)
            yb = yb.to(device=device)
            optimizer.zero_grad(set_to_none=True)
            raw = model(xb).reshape(-1)
            loss = torch.nn.functional.l1_loss(raw, yb.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        model.eval()
        val_losses: list[float] = []
        with torch.inference_mode():
            for xb, yb in val_loader:
                raw = model(xb.to(device=device)).reshape(-1)
                loss = torch.nn.functional.l1_loss(raw, yb.to(device=device).float())
                val_losses.append(float(loss.detach().cpu()))
        val_loss = float(np.mean(val_losses)) if val_losses else float(np.mean(losses))
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= int(args.patience):
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": model_name,
            "state_dict": model.state_dict(),
            "channels": list(CHANNELS),
            "target_hz": TARGET_HZ,
            "target_points": TARGET_POINTS,
            "label_order": list(LABEL_ORDER),
            "target_type": "regression_0_to_4_rounded_clipped_to_labels_1_to_5",
            "official_source": OFFICIAL_ARCHITECTURE_SOURCES.get(model_name, {}),
        },
        checkpoint_path,
    )

    y_pred, y_pred_raw = _predict(model, eval_loader, device)
    return {
        "best_loss": best_loss,
        "epochs_trained": epoch,
        "y_pred": y_pred,
        "y_pred_raw": y_pred_raw,
    }


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    device = _choose_device(str(args.device))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_architecture_dataset_from_preprocessed(args.preprocessed_root)
    x = dataset.x
    y = dataset.y
    metadata = dataset.metadata.copy()
    if int(args.max_windows) > 0:
        x = x[: int(args.max_windows)]
        y = y[: int(args.max_windows)]
        metadata = metadata.iloc[: int(args.max_windows)].reset_index(drop=True)

    if x.shape[1:] != (len(CHANNELS), TARGET_POINTS):
        raise ValueError(f"Expected input [N, {len(CHANNELS)}, {TARGET_POINTS}], got {x.shape}.")

    participants = sorted({int(v) for v in metadata["participant_id"].tolist()})
    if len(participants) < 2:
        raise ValueError("LOSO training requires at least two participants.")

    LOGGER.info("Loaded raw architecture tensor %s on %s.", x.shape, device)
    split_specs = _build_split_specs(participants, str(args.split_mode))
    if not split_specs:
        raise ValueError(f"No evaluation folds are available for split mode {args.split_mode!r}.")

    prediction_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []

    for model_name in args.models:
        for split_spec in split_specs:
            split_name = str(split_spec["split_name"])
            fold_id = int(split_spec["fold_id"])
            test_participants = tuple(int(v) for v in split_spec["test_participants"])
            test_mask = metadata["participant_id"].astype(int).isin(test_participants).to_numpy()
            train_mask = ~test_mask
            if not np.any(train_mask) or not np.any(test_mask):
                continue

            x_train, mean, std = _standardize(x[train_mask], x[train_mask])
            x_test = ((x[test_mask] - mean) / std).astype(np.float32)
            y_train = y[train_mask]
            y_test = y[test_mask]
            fit_idx, val_idx = _train_validation_indices(
                y_train,
                seed=int(args.seed) + int(fold_id),
            )
            test_participant_tag = "-".join(str(v) for v in test_participants)
            checkpoint_path = (
                args.output_dir
                / "checkpoints"
                / split_name
                / f"{model_name}__{split_name}_fold_{fold_id}__participants_{test_participant_tag}.pt"
            )

            stats = _train_one_fold(
                model_name=model_name,
                x_train=x_train[fit_idx],
                y_train=y_train[fit_idx],
                x_val=x_train[val_idx],
                y_val=y_train[val_idx],
                x_test=x_test,
                y_test=y_test,
                args=args,
                device=device,
                checkpoint_path=checkpoint_path,
            )
            y_pred = stats["y_pred"]
            y_pred_raw = stats["y_pred_raw"]
            macro_f1 = float(f1_score(y_test, y_pred, average="macro", labels=list(range(len(LABEL_ORDER))), zero_division=0))
            weighted_f1 = float(f1_score(y_test, y_pred, average="weighted", labels=list(range(len(LABEL_ORDER))), zero_division=0))
            accuracy = float(accuracy_score(y_test, y_pred))
            mae = float(mean_absolute_error(y_test, y_pred))
            raw_mae = float(mean_absolute_error(y_test, y_pred_raw))
            within_1_accuracy = float(np.mean(np.abs(y_test.astype(int) - y_pred.astype(int)) <= 1))
            y_test_binary = _to_low_high(y_test)
            y_pred_binary = _to_low_high(y_pred)
            binary_accuracy = float(accuracy_score(y_test_binary, y_pred_binary))
            binary_macro_f1 = float(f1_score(y_test_binary, y_pred_binary, average="macro", zero_division=0))

            metric_rows.append(
                {
                    "split_name": split_name,
                    "model": model_name,
                    "fold_id": int(fold_id),
                    "test_participant_ids": json.dumps(list(test_participants)),
                    "macro_f1": macro_f1,
                    "weighted_f1": weighted_f1,
                    "accuracy": accuracy,
                    "within_1_accuracy": within_1_accuracy,
                    "binary_accuracy": binary_accuracy,
                    "binary_macro_f1": binary_macro_f1,
                    "mae": mae,
                    "raw_mae": raw_mae,
                    "num_samples": int(len(y_test)),
                    "train_best_loss": float(stats["best_loss"]),
                    "train_epochs": int(stats["epochs_trained"]),
                }
            )

            fold_meta = metadata.loc[test_mask].reset_index(drop=True)
            for idx, row in fold_meta.iterrows():
                out = {
                    "split_name": split_name,
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
                    "y_pred_raw": float(y_pred_raw[idx] + 1.0),
                    "y_true_binary": "high" if int(y_test_binary[idx]) else "low",
                    "y_pred_binary": "high" if int(y_pred_binary[idx]) else "low",
                }
                prediction_rows.append(out)

    predictions_df = pd.DataFrame(prediction_rows)
    metrics_df = pd.DataFrame(metric_rows)
    for column in (
        "macro_f1",
        "weighted_f1",
        "accuracy",
        "within_1_accuracy",
        "binary_accuracy",
        "binary_macro_f1",
        "mae",
        "raw_mae",
    ):
        if column in metrics_df:
            metrics_df[column] = metrics_df[column].round(4)
    if "y_pred_raw" in predictions_df:
        predictions_df["y_pred_raw"] = predictions_df["y_pred_raw"].round(4)
    predictions_path = args.output_dir / "predictions.csv"
    metrics_path = args.output_dir / "metrics_per_fold.csv"
    manifest_path = args.output_dir / "manifest.json"
    predictions_df.to_csv(predictions_path, index=False)
    metrics_df.to_csv(metrics_path, index=False)
    with manifest_path.open("w", encoding="utf-8") as fp:
        json.dump(
            {
                "input_shape": [len(CHANNELS), TARGET_POINTS],
                "target_hz": TARGET_HZ,
                "channels": list(CHANNELS),
                "models": list(args.models),
                "num_windows": int(len(x)),
                "participants": participants,
                "split_mode": str(args.split_mode),
                "fixed4_folds": [list(fold) for fold in FIXED_4_FOLDS],
                "loss": "unweighted_mae_l1",
                "target_type": "regression_0_to_4_rounded_clipped_to_labels_1_to_5",
                "reported_mae": "computed_after_rounding_and_clipping_regression_output",
                "binary_rebinning": {"low": [1, 2], "high": [3, 4, 5]},
                "official_sources": {
                    name: OFFICIAL_ARCHITECTURE_SOURCES.get(name, {})
                    for name in args.models
                },
            },
            fp,
            indent=2,
            sort_keys=True,
        )

    print("\nArchitecture model run complete:")
    print(json.dumps({"predictions": str(predictions_path), "metrics_per_fold": str(metrics_path), "manifest": str(manifest_path)}, indent=2))
    if not metrics_df.empty:
        print(
            metrics_df.groupby(["split_name", "model"])[
                ["mae", "raw_mae", "accuracy", "within_1_accuracy", "binary_accuracy", "binary_macro_f1"]
            ]
            .mean()
            .to_string()
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
