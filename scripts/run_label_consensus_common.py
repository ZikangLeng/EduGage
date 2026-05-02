from __future__ import annotations

import argparse
from datetime import UTC, datetime
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
    _resolve_windows,
    _round_1_to_5,
    _safe_r2,
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


def _build_parser(default_label: int | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run ConSensus on a sampled subset of windows for one participant and one "
            "ground-truth engagement label."
        )
    )
    parser.add_argument("--participant-id", type=int, required=True, help="Participant ID to evaluate.")
    parser.add_argument(
        "--target-label",
        type=int,
        default=default_label,
        choices=[1, 2, 3, 4, 5],
        help="Ground-truth engagement label to filter for.",
    )
    parser.add_argument(
        "--windows-per-label",
        type=int,
        required=True,
        help="Number of windows to sample for the requested label.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when sampling windows.",
    )
    parser.add_argument(
        "--sample-mode",
        choices=["random", "first"],
        default="random",
        help="How to choose windows within the target label subset.",
    )
    parser.add_argument("--run-id", default=None, help="Optional explicit run id.")
    parser.add_argument("--tag", default="", help="Optional free-form tag added to artifacts.")
    parser.add_argument("--provider", default=None, help="Optional provider override.")
    parser.add_argument("--model", default=None, help="Optional model override.")
    parser.add_argument("--openai-base-url", default=None, help="Optional OpenAI-compatible base URL.")
    parser.add_argument("--openai-api-key", default=None, help="Optional OpenAI-compatible API key.")
    parser.add_argument("--litellm-base-url", default=None, help="Optional LiteLLM base URL.")
    parser.add_argument("--litellm-api-key", default=None, help="Optional LiteLLM API key.")
    parser.add_argument(
        "--preprocessed-root",
        default=None,
        help="Optional path to preprocessed_data root. Default: <repo>/preprocessed_data",
    )
    parser.add_argument(
        "--window-generation-mode",
        choices=["interval_sliding", "event_trailing"],
        default=None,
        help="Optional window construction override.",
    )
    parser.add_argument("--window-size-sec", type=float, default=None, help="Optional window size override.")
    parser.add_argument("--stride-sec", type=float, default=None, help="Optional stride override.")
    parser.add_argument(
        "--examples-per-class",
        type=int,
        default=0,
        help="In-context examples per class. Default: 0 for strict zero-shot prompt sweeps.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser


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
        f1_score(
            y_true.astype(int),
            y_pred_clipped,
            average="macro",
            labels=np.arange(1, 6),
            zero_division=0,
        )
    ) if len(pred_df) else 0.0
    cm = confusion_matrix(y_true.astype(int), y_pred.astype(int), labels=np.arange(1, 6))
    return {
        "task": "five_class",
        "num_samples": int(len(pred_df)),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "rounded_accuracy": acc,
        "rounded_macro_f1": macro_f1,
    }, cm


def _compute_binary_metrics(pred_df: pd.DataFrame) -> tuple[dict[str, float], np.ndarray]:
    y_true = pd.to_numeric(pred_df["y_true"], errors="coerce").astype(int)
    y_pred = pd.to_numeric(pred_df["y_pred"], errors="coerce").astype(int)
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


def _default_run_id(participant_id: int, target_label: int, tag: str) -> str:
    stamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%S%fZ")
    suffix = f"__{tag.strip()}" if tag.strip() else ""
    return f"{stamp}__prompt_iter_p{participant_id}_y{target_label}{suffix}"


def _sample_windows(
    windows: pd.DataFrame,
    *,
    participant_id: int,
    target_label: int,
    windows_per_label: int,
    seed: int,
    sample_mode: str,
) -> pd.DataFrame:
    subset = windows[
        (windows["participant_id"] == int(participant_id))
        & (pd.to_numeric(windows["label_5class"], errors="coerce") == int(target_label))
    ].copy()
    if subset.empty:
        return subset
    subset = subset.sort_values(["session_key", "t_start_sec", "window_id"], kind="mergesort")
    n = min(int(windows_per_label), len(subset))
    if sample_mode == "first":
        return subset.head(n).copy()
    return subset.sample(n=n, random_state=int(seed)).sort_values(
        ["session_key", "t_start_sec", "window_id"],
        kind="mergesort",
    )


def _print_json(title: str, payload: dict[str, Any]) -> None:
    print(f"\n{title}")
    print(json.dumps(payload, indent=2, sort_keys=True))


def _write_skip_summary(
    *,
    config,
    participant_id: int,
    target_label: int,
    windows_requested: int,
    sample_mode: str,
    seed: int,
    reason: str,
) -> Path:
    phase_dir = config.run_dir / "baselines" / "consensus_zs"
    phase_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "skipped",
        "participant_id": int(participant_id),
        "target_label": int(target_label),
        "windows_requested": int(windows_requested),
        "windows_selected": 0,
        "sample_mode": str(sample_mode),
        "seed": int(seed),
        "reason": str(reason),
    }
    out_path = phase_dir / f"label_{int(target_label)}_summary.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return out_path


def run_for_fixed_label(default_label: int | None = None) -> int:
    parser = _build_parser(default_label=default_label)
    args = parser.parse_args()
    if args.target_label is None:
        parser.error("--target-label is required.")

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    loaded_env_files = load_env_files(REPO_ROOT)
    if loaded_env_files:
        LOGGER.info("Loaded environment from: %s", ", ".join(str(p) for p in loaded_env_files))

    config = build_default_config(repo_root=REPO_ROOT)
    config.run_id = args.run_id or _default_run_id(
        int(args.participant_id),
        int(args.target_label),
        str(args.tag),
    )
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
    if args.window_generation_mode is not None:
        config.window_generation_mode = str(args.window_generation_mode)
    if args.window_size_sec is not None:
        config.window_size_sec = float(args.window_size_sec)
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
        raise RuntimeError("No windows found. Run Phase C first or provide a valid windows artifact.")

    selected_windows = _sample_windows(
        all_windows,
        participant_id=int(args.participant_id),
        target_label=int(args.target_label),
        windows_per_label=int(args.windows_per_label),
        seed=int(args.seed),
        sample_mode=str(args.sample_mode),
    )
    if selected_windows.empty:
        summary_path = _write_skip_summary(
            config=config,
            participant_id=int(args.participant_id),
            target_label=int(args.target_label),
            windows_requested=int(args.windows_per_label),
            sample_mode=str(args.sample_mode),
            seed=int(args.seed),
            reason=(
                f"No windows found for participant_id={args.participant_id} "
                f"and label={args.target_label}."
            ),
        )
        _print_json(
            "Skipped",
            {
                "participant_id": int(args.participant_id),
                "target_label": int(args.target_label),
                "reason": (
                    f"No windows found for participant_id={args.participant_id} "
                    f"and label={args.target_label}."
                ),
                "label_summary": str(summary_path),
            },
        )
        return 0

    phase_dir = config.run_dir / "baselines" / "consensus_zs"
    phase_dir.mkdir(parents=True, exist_ok=True)
    selected_windows_csv = phase_dir / f"selected_windows_label_{int(args.target_label)}.csv"
    selected_windows.to_csv(selected_windows_csv, index=False)

    _print_json(
        "Run Setup",
        {
            "run_id": config.run_id,
            "participant_id": int(args.participant_id),
            "target_label": int(args.target_label),
            "windows_requested": int(args.windows_per_label),
            "windows_selected": int(len(selected_windows)),
            "sample_mode": str(args.sample_mode),
            "provider": config.baseline.consensus_zero_shot.provider,
            "model": config.baseline.consensus_zero_shot.model,
            "examples_per_class": int(config.baseline.consensus_zero_shot.examples_per_class),
            "window_size_sec": float(config.window_size_sec),
            "window_generation_mode": str(config.window_generation_mode),
            "selected_windows_csv": str(selected_windows_csv),
        },
    )

    outputs, stats = run_phase_z1_baseline_consensus(
        config,
        window_index=selected_windows,
        example_window_index=all_windows,
        preprocessed_root=preprocessed_root,
    )

    predictions_path = Path(outputs["predictions"]).resolve()
    pred_df = read_table_with_fallback(predictions_path)
    if pred_df.empty:
        raise RuntimeError("No predictions produced for selected label subset.")

    metrics_5, cm_5 = _compute_five_class_metrics(pred_df)
    metrics_bin, cm_bin = _compute_binary_metrics(pred_df)
    prediction_distribution = (
        pd.to_numeric(pred_df["y_pred"], errors="coerce")
        .value_counts(dropna=False)
        .sort_index()
        .to_dict()
    )

    label_summary = {
        "participant_id": int(args.participant_id),
        "target_label": int(args.target_label),
        "windows_requested": int(args.windows_per_label),
        "windows_selected": int(len(selected_windows)),
        "sample_mode": str(args.sample_mode),
        "seed": int(args.seed),
        "prediction_distribution": {str(k): int(v) for k, v in prediction_distribution.items()},
        "five_class_metrics": metrics_5,
        "binary_metrics": metrics_bin,
        "outputs": {
            "selected_windows_csv": str(selected_windows_csv),
            "predictions": str(predictions_path),
            "baseline_metrics_overall": str(Path(outputs["metrics_overall"]).resolve()),
            "baseline_summary": str(Path(outputs["z1_summary"]).resolve()),
        },
        "baseline_stats": stats,
    }

    label_summary_path = phase_dir / f"label_{int(args.target_label)}_summary.json"
    label_summary_path.write_text(
        json.dumps(label_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    cm5_path = phase_dir / f"label_{int(args.target_label)}_confusion_5class.csv"
    cmb_path = phase_dir / f"label_{int(args.target_label)}_confusion_binary.csv"
    pd.DataFrame(cm_5, index=[1, 2, 3, 4, 5], columns=[1, 2, 3, 4, 5]).to_csv(cm5_path)
    pd.DataFrame(cm_bin, index=["low_123", "high_45"], columns=["pred_low_123", "pred_high_45"]).to_csv(cmb_path)

    _print_json("5-Class Metrics", metrics_5)
    _print_json("Binary Metrics", metrics_bin)
    _print_json(
        "Artifacts",
        {
            "label_summary": str(label_summary_path),
            "confusion_5class": str(cm5_path),
            "confusion_binary": str(cmb_path),
            **label_summary["outputs"],
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run_for_fixed_label())
