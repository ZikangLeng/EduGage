from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from engagement.io_utils import read_table_with_fallback


DEFAULT_METRIC_TOLERANCE = 1e-6
LEGACY_MODEL_IMPLS = {"legacy_manual_softmax_v1"}
SPLIT_FILE_CANDIDATES = (
    "evaluation_splits.parquet",
    "fixed_group_splits.parquet",
    "grouped_kfold_splits.parquet",
    "loso_splits.parquet",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_split_table(phase_e_dir: Path) -> pd.DataFrame:
    for file_name in SPLIT_FILE_CANDIDATES:
        candidate = phase_e_dir / file_name
        if candidate.exists() or candidate.with_suffix(".csv").exists():
            return read_table_with_fallback(candidate)
    return pd.DataFrame()


def _has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (list, tuple, dict)):
        return True
    try:
        return not bool(pd.isna(value))
    except (TypeError, ValueError):
        return True


def _parse_train_participants(value: Any) -> list[int]:
    if isinstance(value, list):
        return sorted({int(v) for v in value})
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return sorted({int(v) for v in parsed})
    raise ValueError(f"Invalid train_participants payload: {value!r}")


def _required_columns_for(label: str) -> set[str]:
    mapping: dict[str, set[str]] = {
        "windows": {
            "window_id",
            "session_key",
            "participant_id",
            "event_id",
            "video_uid",
            "label_3class",
        },
        "embeddings": {
            "window_id",
            "session_key",
            "participant_id",
            "modality",
            "source_id",
            "embedding",
        },
        "predictions": {
            "experiment",
            "fold_id",
            "test_participant_id",
            "window_id",
            "participant_id",
            "event_id",
            "y_true",
            "y_pred",
            "proba_low",
            "proba_neutral",
            "proba_high",
        },
        "evaluation_splits": {
            "experiment",
            "fold_id",
            "train_participants",
            "num_train_windows",
            "num_test_windows",
        },
        "metrics_overall": {
            "experiment",
            "macro_f1",
            "weighted_f1",
            "accuracy",
            "num_samples",
        },
    }
    return mapping.get(label, set())


def _compare_row_counts(baseline_dir: Path, candidate_dir: Path) -> list[str]:
    checks = [
        ("windows", "window_index.parquet", baseline_dir.parent / "windows", candidate_dir.parent / "windows"),
        ("embeddings", "embedding_table.parquet", baseline_dir.parent / "embeddings", candidate_dir.parent / "embeddings"),
        ("predictions", "predictions.parquet", baseline_dir, candidate_dir),
        ("evaluation_splits", None, baseline_dir, candidate_dir),
    ]

    issues: list[str] = []
    for label, file_name, base_root, cand_root in checks:
        if file_name is None:
            base_df = _read_split_table(base_root)
            cand_df = _read_split_table(cand_root)
        else:
            base_df = read_table_with_fallback(base_root / file_name)
            cand_df = read_table_with_fallback(cand_root / file_name)
        if len(base_df) != len(cand_df):
            issues.append(
                f"Row count drift for {label}: baseline={len(base_df)}, candidate={len(cand_df)}"
            )

    return issues


def _compare_schema_columns(baseline_dir: Path, candidate_dir: Path) -> list[str]:
    checks = [
        ("windows", "window_index.parquet", baseline_dir.parent / "windows", candidate_dir.parent / "windows"),
        ("embeddings", "embedding_table.parquet", baseline_dir.parent / "embeddings", candidate_dir.parent / "embeddings"),
        ("predictions", "predictions.parquet", baseline_dir, candidate_dir),
        ("evaluation_splits", None, baseline_dir, candidate_dir),
        ("metrics_overall", "metrics_overall.parquet", baseline_dir, candidate_dir),
    ]

    issues: list[str] = []
    for label, file_name, base_root, cand_root in checks:
        if file_name is None:
            base_df = _read_split_table(base_root)
            cand_df = _read_split_table(cand_root)
        else:
            base_df = read_table_with_fallback(base_root / file_name)
            cand_df = read_table_with_fallback(cand_root / file_name)

        required_columns = _required_columns_for(label)
        missing_required = sorted(required_columns - set(cand_df.columns))
        if missing_required:
            issues.append(f"Candidate {label} missing required columns: {missing_required}")

        base_cols = set(base_df.columns)
        cand_cols = set(cand_df.columns)
        if base_cols != cand_cols:
            missing_from_candidate = sorted(base_cols - cand_cols)
            extra_in_candidate = sorted(cand_cols - base_cols)
            issues.append(
                f"Schema drift for {label}: missing_from_candidate={missing_from_candidate}, "
                f"extra_in_candidate={extra_in_candidate}"
            )

    return issues


def _metric_tolerance_for(experiment: str, metric: str, allowlist: dict[str, Any]) -> float:
    exp_cfg = allowlist.get("experiments", {}).get(experiment, {})
    metric_cfg = exp_cfg.get("metrics", {})
    return float(metric_cfg.get(metric, allowlist.get("default_metric_tolerance", DEFAULT_METRIC_TOLERANCE)))


def _compare_metrics(baseline_dir: Path, candidate_dir: Path, allowlist: dict[str, Any]) -> list[str]:
    baseline_metrics = read_table_with_fallback(baseline_dir / "metrics_overall.parquet")
    candidate_metrics = read_table_with_fallback(candidate_dir / "metrics_overall.parquet")

    issues: list[str] = []

    merge = baseline_metrics.merge(
        candidate_metrics,
        on="experiment",
        suffixes=("_baseline", "_candidate"),
        how="outer",
        indicator=True,
    )

    missing = merge[merge["_merge"] != "both"]
    if not missing.empty:
        issues.append("Experiment set changed between baseline and candidate metrics_overall artifacts.")
        return issues

    for row in merge.to_dict(orient="records"):
        experiment = str(row["experiment"])
        for metric in ("macro_f1", "weighted_f1", "accuracy"):
            b = float(row[f"{metric}_baseline"])
            c = float(row[f"{metric}_candidate"])
            tol = _metric_tolerance_for(experiment, metric, allowlist)
            if abs(c - b) > tol:
                issues.append(
                    f"Metric drift {experiment}.{metric}: baseline={b:.6f}, candidate={c:.6f}, tolerance={tol:.6f}"
                )

    return issues


def _check_split_integrity(split_df: pd.DataFrame, label: str) -> list[str]:
    issues: list[str] = []
    required = _required_columns_for("evaluation_splits")
    missing = sorted(required - set(split_df.columns))
    if missing:
        issues.append(f"{label} evaluation_splits missing required columns: {missing}")
        return issues

    key_cols = ["experiment", "fold_id"]
    if "test_participant_id" in split_df.columns:
        key_cols.append("test_participant_id")
    elif "test_group" in split_df.columns:
        key_cols.append("test_group")
    duplicate_rows = split_df.duplicated(subset=key_cols, keep=False)
    if duplicate_rows.any():
        issues.append(f"{label} evaluation_splits has duplicate fold rows.")

    for row in split_df.to_dict(orient="records"):
        try:
            train_participants = _parse_train_participants(row.get("train_participants"))
        except ValueError as exc:
            issues.append(f"{label} split parse error: {exc}")
            continue

        raw_test_participants = row.get("test_participants")
        if _has_value(raw_test_participants):
            test_participants = _parse_train_participants(raw_test_participants)
        elif _has_value(row.get("test_participant_id")):
            test_participants = [int(row["test_participant_id"])]
        else:
            test_participants = []

        leaked = sorted(set(test_participants) & set(train_participants))
        if leaked:
            issues.append(
                f"{label} split leakage: test participants {leaked} appear in train_participants."
            )

        if int(row["num_train_windows"]) <= 0:
            issues.append(
                f"{label} invalid num_train_windows for experiment={row['experiment']} fold={row['fold_id']}."
            )
        if int(row["num_test_windows"]) <= 0:
            issues.append(
                f"{label} invalid num_test_windows for experiment={row['experiment']} fold={row['fold_id']}."
            )

    return issues


def _compare_split_parity(baseline_dir: Path, candidate_dir: Path) -> list[str]:
    baseline_df = _read_split_table(baseline_dir)
    candidate_df = _read_split_table(candidate_dir)

    issues: list[str] = []
    issues.extend(_check_split_integrity(baseline_df, "baseline"))
    issues.extend(_check_split_integrity(candidate_df, "candidate"))

    key_cols = ["experiment", "fold_id"]
    if "test_group" in baseline_df.columns and "test_group" in candidate_df.columns:
        key_cols.append("test_group")
    elif "test_participant_id" in baseline_df.columns and "test_participant_id" in candidate_df.columns:
        key_cols.append("test_participant_id")
    base_cmp = baseline_df[key_cols + ["train_participants", "num_train_windows", "num_test_windows"]].copy()
    cand_cmp = candidate_df[key_cols + ["train_participants", "num_train_windows", "num_test_windows"]].copy()

    base_cmp["train_participants_norm"] = base_cmp["train_participants"].map(_parse_train_participants)
    cand_cmp["train_participants_norm"] = cand_cmp["train_participants"].map(_parse_train_participants)

    base_cmp = base_cmp.drop(columns=["train_participants"]).sort_values(key_cols).reset_index(drop=True)
    cand_cmp = cand_cmp.drop(columns=["train_participants"]).sort_values(key_cols).reset_index(drop=True)

    if not base_cmp.equals(cand_cmp):
        issues.append("Evaluation split parity drift between baseline and candidate artifacts.")

    return issues


def _check_checkpoint_schema(phase_e_dir: Path) -> list[str]:
    issues: list[str] = []
    checkpoints_dir = phase_e_dir / "checkpoints"
    for meta_path in checkpoints_dir.glob("*.json"):
        payload = _read_json(meta_path)

        version = payload.get("artifact_schema_version")
        if version is None:
            issues.append(f"Missing artifact_schema_version in {meta_path}")
        else:
            try:
                version_int = int(version)
            except (TypeError, ValueError):
                issues.append(f"Non-integer artifact_schema_version in {meta_path}: {version!r}")
            else:
                if version_int < 2:
                    issues.append(f"Unsupported checkpoint schema version in {meta_path}: {version_int}")

        model_impl = str(payload.get("model_impl", "")).strip()
        if not model_impl:
            issues.append(f"Missing model_impl in {meta_path}")
        elif model_impl in LEGACY_MODEL_IMPLS:
            issues.append(f"Legacy model_impl is no longer supported: {meta_path}")

    return issues


def run_regression_gate(
    baseline_phase_e_dir: Path,
    candidate_phase_e_dir: Path,
    allowlist_path: Path | None,
) -> list[str]:
    allowlist: dict[str, Any] = {}
    if allowlist_path is not None and allowlist_path.exists():
        allowlist = _read_json(allowlist_path)

    issues: list[str] = []
    issues.extend(_compare_row_counts(baseline_phase_e_dir, candidate_phase_e_dir))
    issues.extend(_compare_schema_columns(baseline_phase_e_dir, candidate_phase_e_dir))
    issues.extend(_compare_split_parity(baseline_phase_e_dir, candidate_phase_e_dir))
    issues.extend(_compare_metrics(baseline_phase_e_dir, candidate_phase_e_dir, allowlist))
    issues.extend(_check_checkpoint_schema(candidate_phase_e_dir))
    return issues


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare baseline vs candidate Phase E artifacts.")
    parser.add_argument("--baseline-phase-e-dir", required=True, help="Path to baseline phase_e directory")
    parser.add_argument("--candidate-phase-e-dir", required=True, help="Path to candidate phase_e directory")
    parser.add_argument(
        "--allowlist",
        default=None,
        help="Optional JSON allowlist for accepted metric drift tolerances.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    baseline_dir = Path(args.baseline_phase_e_dir).resolve()
    candidate_dir = Path(args.candidate_phase_e_dir).resolve()
    allowlist = Path(args.allowlist).resolve() if args.allowlist else None

    issues = run_regression_gate(baseline_dir, candidate_dir, allowlist)
    if issues:
        print("Regression gate failed:")
        for issue in issues:
            print(f"- {issue}")
        return 1

    print("Regression gate passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
