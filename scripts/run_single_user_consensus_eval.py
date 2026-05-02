from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from engagement.baselines.consensus_zero_shot import (  # noqa: E402
    _round_1_to_5,
    _safe_r2,
    _resolve_windows,
    run_phase_z1_baseline_consensus,
)
from engagement.config import (  # noqa: E402
    build_default_config,
    validate_baseline_config,
    validate_core_config,
)
from engagement.env_utils import load_env_files  # noqa: E402
from engagement.io_utils import read_table_with_fallback  # noqa: E402


LOGGER = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run ConSensus zero-shot baseline on a single participant and report both "
            "5-class and binary (1/2/3 low vs 4/5 high) metrics."
        )
    )
    parser.add_argument("--participant-id", type=int, required=True, help="Participant ID to evaluate.")
    parser.add_argument(
        "--max-windows",
        type=int,
        default=80,
        help="Optional cap on number of windows for quick testing (<=0 means all windows).",
    )
    parser.add_argument("--run-id", default=None, help="Optional run id. Default uses config timestamp.")
    parser.add_argument(
        "--provider",
        default=None,
        help="Optional provider override (e.g. openai, together, ollama, heuristic).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional model override (e.g. openai:gpt-5-mini).",
    )
    parser.add_argument(
        "--openai-base-url",
        default=None,
        help="Optional OpenAI-compatible base URL (e.g. http://localhost:8000/v1 for LiteLLM).",
    )
    parser.add_argument(
        "--openai-api-key",
        default=None,
        help="Optional API key for OpenAI-compatible endpoints.",
    )
    parser.add_argument(
        "--litellm-base-url",
        default=None,
        help="Optional LiteLLM base URL (used for local-* OpenAI-compatible models).",
    )
    parser.add_argument(
        "--litellm-api-key",
        default=None,
        help="Optional LiteLLM API key for local models.",
    )
    parser.add_argument(
        "--preprocessed-root",
        default=None,
        help="Optional path to preprocessed_data root. Default: <repo>/preprocessed_data",
    )
    parser.add_argument(
        "--window-generation-mode",
        choices=["interval_sliding", "event_trailing"],
        default=None,
        help="Window construction policy. event_trailing yields one trailing window per supervised event.",
    )
    parser.add_argument(
        "--window-size-sec",
        type=float,
        default=None,
        help="Optional override for config.window_size_sec (prompt metadata + any upstream logic).",
    )
    parser.add_argument(
        "--stride-sec",
        type=float,
        default=None,
        help="Optional override for config.stride_sec.",
    )
    parser.add_argument(
        "--examples-per-class",
        type=int,
        default=None,
        help="Optional override for in-context examples per class. Use 0 for strict zero-shot.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def _compute_five_class_metrics(pred_df: pd.DataFrame) -> tuple[dict[str, float], np.ndarray]:
    y_true = pd.to_numeric(pred_df["y_true"], errors="coerce").to_numpy(dtype=float)
    y_pred_cont = pd.to_numeric(pred_df["y_pred_continuous"], errors="coerce").to_numpy(dtype=float)
    y_pred = pd.to_numeric(pred_df["y_pred"], errors="coerce").to_numpy(dtype=int)

    y_pred_clipped = np.asarray([_round_1_to_5(float(v)) for v in y_pred_cont], dtype=int)

    mae = float(np.mean(np.abs(y_true - y_pred_cont))) if len(pred_df) else 0.0
    rmse = float(np.sqrt(np.mean((y_true - y_pred_cont) ** 2))) if len(pred_df) else 0.0
    r2 = float(_safe_r2(y_true, y_pred_cont))
    acc = float(accuracy_score(y_true.astype(int), y_pred_clipped)) if len(pred_df) else 0.0
    macro_f1 = float(
        f1_score(y_true.astype(int), y_pred_clipped, average="macro", labels=np.arange(1, 6), zero_division=0)
    ) if len(pred_df) else 0.0
    cm = confusion_matrix(y_true.astype(int), y_pred.astype(int), labels=np.arange(1, 6))

    metrics = {
        "task": "five_class",
        "num_samples": int(len(pred_df)),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "rounded_accuracy": acc,
        "rounded_macro_f1": macro_f1,
    }
    return metrics, cm


def _compute_binary_metrics(pred_df: pd.DataFrame) -> tuple[dict[str, float], np.ndarray]:
    y_true = pd.to_numeric(pred_df["y_true"], errors="coerce").astype(int)
    y_pred = pd.to_numeric(pred_df["y_pred"], errors="coerce").astype(int)

    # Requested mapping: 1/2/3 -> low(0), 4/5 -> high(1)
    y_true_bin = (y_true >= 4).astype(int)
    y_pred_bin = (y_pred >= 4).astype(int)

    metrics = {
        "task": "binary_low_123_vs_high_45",
        "num_samples": int(len(pred_df)),
        "accuracy": float(accuracy_score(y_true_bin, y_pred_bin)),
        "precision_high": float(precision_score(y_true_bin, y_pred_bin, zero_division=0)),
        "recall_high": float(recall_score(y_true_bin, y_pred_bin, zero_division=0)),
        "f1_high": float(f1_score(y_true_bin, y_pred_bin, zero_division=0)),
        "f1_macro": float(f1_score(y_true_bin, y_pred_bin, average="macro", zero_division=0)),
    }
    cm = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])
    return metrics, cm


def _print_json(title: str, payload: dict[str, Any]) -> None:
    print(f"\n{title}")
    print(json.dumps(payload, indent=2, sort_keys=True))


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    loaded_env_files = load_env_files(REPO_ROOT)
    if loaded_env_files:
        LOGGER.info("Loaded environment from: %s", ", ".join(str(p) for p in loaded_env_files))

    config = build_default_config(repo_root=REPO_ROOT)
    if args.run_id:
        config.run_id = args.run_id
    if args.provider:
        config.baseline.consensus_zero_shot.provider = str(args.provider)
    if args.model:
        config.baseline.consensus_zero_shot.model = str(args.model)
    if args.openai_base_url:
        os.environ["OPENAI_BASE_URL"] = str(args.openai_base_url)
    if args.openai_api_key:
        os.environ["OPENAI_API_KEY"] = str(args.openai_api_key)
    if args.litellm_base_url:
        os.environ["LITELLM_BASE_URL"] = str(args.litellm_base_url)
    if args.litellm_api_key:
        os.environ["LITELLM_API_KEY"] = str(args.litellm_api_key)
    if args.window_size_sec is not None:
        config.window_size_sec = float(args.window_size_sec)
    if args.window_generation_mode is not None:
        config.window_generation_mode = str(args.window_generation_mode)
    if args.stride_sec is not None:
        config.stride_sec = float(args.stride_sec)
    if args.examples_per_class is not None:
        config.baseline.consensus_zero_shot.examples_per_class = int(args.examples_per_class)
        config.baseline.consensus_zero_shot.zero_shot = bool(int(args.examples_per_class) <= 0)

    validate_core_config(config)
    validate_baseline_config(config)

    preprocessed_root = (
        Path(args.preprocessed_root).resolve()
        if args.preprocessed_root
        else (config.repo_root / "preprocessed_data").resolve()
    )

    all_windows = _resolve_windows(config, window_index=None)
    if all_windows.empty:
        raise RuntimeError("No windows found. Run Phase C first or provide a valid run/windows artifact.")

    user_windows = all_windows[all_windows["participant_id"] == int(args.participant_id)].copy()
    if user_windows.empty:
        raise RuntimeError(f"No windows found for participant_id={args.participant_id}")

    user_windows = user_windows.sort_values(["session_key", "t_start_sec"], kind="mergesort")
    if int(args.max_windows) > 0 and len(user_windows) > int(args.max_windows):
        user_windows = user_windows.head(int(args.max_windows)).copy()

    _print_json(
        "Run Setup",
        {
            "run_id": config.run_id,
            "participant_id": int(args.participant_id),
            "num_windows_selected": int(len(user_windows)),
            "provider": config.baseline.consensus_zero_shot.provider,
            "model": config.baseline.consensus_zero_shot.model,
            "examples_per_class": int(config.baseline.consensus_zero_shot.examples_per_class),
            "window_size_sec": float(config.window_size_sec),
            "stride_sec": float(config.stride_sec),
            "window_generation_mode": str(config.window_generation_mode),
            "preprocessed_root": str(preprocessed_root),
        },
    )

    outputs, stats = run_phase_z1_baseline_consensus(
        config,
        window_index=user_windows,
        preprocessed_root=preprocessed_root,
    )

    predictions_path = Path(outputs["predictions"]).resolve()
    pred_df = read_table_with_fallback(predictions_path)
    if pred_df.empty:
        raise RuntimeError("No predictions produced for selected user/windows.")

    metrics_5, cm_5 = _compute_five_class_metrics(pred_df)
    metrics_bin, cm_bin = _compute_binary_metrics(pred_df)

    out_dir = predictions_path.parent
    pred_out = pred_df.copy()
    pred_out["y_true_binary"] = (pd.to_numeric(pred_out["y_true"], errors="coerce") >= 4).astype(int)
    pred_out["y_pred_binary"] = (pd.to_numeric(pred_out["y_pred"], errors="coerce") >= 4).astype(int)
    pred_out_path = out_dir / f"single_user_p{int(args.participant_id)}_predictions_with_binary.csv"
    pred_out.to_csv(pred_out_path, index=False)

    cm5_path = out_dir / f"single_user_p{int(args.participant_id)}_confusion_5class.csv"
    cm_bin_path = out_dir / f"single_user_p{int(args.participant_id)}_confusion_binary.csv"
    pd.DataFrame(cm_5, index=[1, 2, 3, 4, 5], columns=[1, 2, 3, 4, 5]).to_csv(cm5_path)
    pd.DataFrame(cm_bin, index=["low_123", "high_45"], columns=["pred_low_123", "pred_high_45"]).to_csv(cm_bin_path)

    summary = {
        "participant_id": int(args.participant_id),
        "num_windows_evaluated": int(len(pred_df)),
        "mapping_binary": {"low": [1, 2, 3], "high": [4, 5]},
        "five_class_metrics": metrics_5,
        "binary_metrics": metrics_bin,
        "outputs": {
            "baseline_predictions": str(predictions_path),
            "predictions_with_binary": str(pred_out_path),
            "confusion_5class": str(cm5_path),
            "confusion_binary": str(cm_bin_path),
            "baseline_metrics_overall": str(Path(outputs["metrics_overall"]).resolve()),
            "baseline_summary": str(Path(outputs["z1_summary"]).resolve()),
        },
        "baseline_stats": stats,
    }

    summary_path = out_dir / f"single_user_p{int(args.participant_id)}_eval_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    _print_json("5-Class Metrics", metrics_5)
    print("\n5-class confusion matrix (rows=true, cols=pred) [1..5]:")
    print(cm_5)

    _print_json("Binary Metrics (1/2/3 low vs 4/5 high)", metrics_bin)
    print("\nBinary confusion matrix (rows=true, cols=pred) [low_123, high_45]:")
    print(cm_bin)

    _print_json("Artifact Paths", {"summary": str(summary_path), **summary["outputs"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
