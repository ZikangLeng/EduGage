from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from engagement.baselines.consensus_zero_shot import (  # noqa: E402
    run_phase_z1_baseline_consensus,
)
from engagement.config import (  # noqa: E402
    build_default_config,
    model_availability,
    persist_run_metadata,
    validate_baseline_config,
    validate_core_config,
)
from engagement.data_pipeline import run_phase_b  # noqa: E402
from engagement.env_utils import load_env_files  # noqa: E402
from engagement.features import run_phase_d  # noqa: E402
from engagement.labels_windows import run_phase_c  # noqa: E402
from engagement.modality_ablation import (  # noqa: E402
    apply_complete_only_regression_preset,
    run_modality_ablation_greedy,
)
from engagement.optuna_tune import (  # noqa: E402
    run_optuna_multimodal_parallel,
    run_optuna_multimodal_study,
)
from engagement.train_eval import run_phase_e  # noqa: E402
from engagement.preprocessed_data import run_phase_preprocessed_data  # noqa: E402
from engagement.trial_matrix import run_trial_matrix, run_trial_matrix_optuna  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the V1 engagement pipeline.")
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
        help="Optional override for config.window_size_sec.",
    )
    parser.add_argument(
        "--stride-sec",
        type=float,
        default=None,
        help="Optional override for config.stride_sec.",
    )
    parser.add_argument(
        "--stage",
        choices=[
            "config",
            "data",
            "labels",
            "features",
            "preprocessed_data",
            "train_eval",
            "ablate_modalities",
            "trial_matrix",
            "trial_matrix_optuna",
            "tune_optuna",
            "tune_optuna_parallel",
            "baseline_consensus_zs",
            "all",
        ],
        default="all",
        help="Pipeline stage to execute.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional explicit run ID. Defaults to UTC timestamp.",
    )
    parser.add_argument(
        "--allow-relative-time-fallback",
        action="store_true",
        help="Allow generating windows without absolute timestamp alignment.",
    )
    parser.add_argument(
        "--include-p2-absolute",
        action="store_true",
        help="Ensure P2 is included for absolute-time slicing.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    parser.add_argument(
        "--force-recompute-embeddings",
        action="store_true",
        help="Ignore embedding cache and recompute all embeddings for this config/preprocess hash.",
    )
    parser.add_argument(
        "--provider",
        default=None,
        help="Optional zero-shot baseline provider override (e.g. openai, together, ollama, heuristic).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional zero-shot baseline model override (e.g. openai:gpt-5-mini or openai:local-qwen).",
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
        "--multimodal-target-points",
        type=int,
        default=None,
        help="Optional override for multimodal_train.target_points when native-frequency mode is disabled.",
    )
    parser.add_argument(
        "--multimodal-use-native-frequency",
        action="store_true",
        help="Use native per-modality sample frequency with padding for batching.",
    )
    parser.add_argument(
        "--multimodal-disable-multi-gpu",
        action="store_true",
        help="Disable automatic multi-GPU DataParallel when multiple CUDA GPUs are available.",
    )
    parser.add_argument(
        "--multimodal-use-resampled-grid",
        action="store_true",
        help="Use fixed-length resampled modality windows instead of native-frequency inputs.",
    )
    parser.add_argument(
        "--multimodal-task-mode",
        choices=["multitask", "binary", "ordinal"],
        default=None,
        help="Train the multimodal backend in multitask, binary-only, or ordinal-only mode.",
    )
    parser.add_argument(
        "--multimodal-split-mode",
        choices=["fixed_groups", "grouped_kfold", "loso"],
        default=None,
        help="Participant-level evaluation split mode for the multimodal backend.",
    )
    parser.add_argument(
        "--multimodal-num-folds",
        type=int,
        default=None,
        help="Number of participant-group folds when using grouped_kfold splitting.",
    )
    parser.add_argument(
        "--multimodal-epochs",
        type=int,
        default=None,
        help="Optional override for multimodal_train.epochs.",
    )
    parser.add_argument(
        "--multimodal-batch-size",
        type=int,
        default=None,
        help="Optional override for multimodal_train.batch_size.",
    )
    parser.add_argument(
        "--multimodal-modality-dropout-prob",
        type=float,
        default=None,
        help="Optional override for multimodal_train.modality_dropout_prob. Applied during training only.",
    )
    parser.add_argument(
        "--multimodal-complete-windows-only",
        action="store_true",
        help="Restrict multimodal Phase E training and evaluation to windows with all modeled modalities present.",
    )
    parser.add_argument(
        "--multimodal-complete-windows-reference",
        choices=["selected", "all"],
        default=None,
        help="Which modality set defines completeness when --multimodal-complete-windows-only is enabled.",
    )
    parser.add_argument(
        "--multimodal-disable-context",
        action="store_true",
        help="Disable relative-position context features for multimodal training/evaluation.",
    )
    parser.add_argument(
        "--multimodal-learning-rate",
        type=float,
        default=None,
        help="Optional override for multimodal_train.learning_rate.",
    )
    parser.add_argument(
        "--multimodal-weight-decay",
        type=float,
        default=None,
        help="Optional override for multimodal_train.weight_decay.",
    )
    parser.add_argument(
        "--multimodal-lambda-binary",
        type=float,
        default=None,
        help="Optional override for multimodal_train.lambda_binary.",
    )
    parser.add_argument(
        "--multimodal-lambda-ordinal",
        type=float,
        default=None,
        help="Optional override for multimodal_train.lambda_ordinal.",
    )
    parser.add_argument(
        "--multimodal-lambda-regression",
        type=float,
        default=None,
        help="Optional override for multimodal_train.lambda_regression.",
    )
    parser.add_argument(
        "--multimodal-embedding-dim",
        type=int,
        default=None,
        help="Optional override for multimodal_train.embedding_dim.",
    )
    parser.add_argument(
        "--multimodal-fusion-hidden-dim",
        type=int,
        default=None,
        help="Optional override for multimodal_train.fusion_hidden_dim.",
    )
    parser.add_argument(
        "--multimodal-device",
        choices=["auto", "cpu", "cuda", "mps"],
        default=None,
        help="Optional override for multimodal_train.device. Defaults to auto (GPU first).",
    )
    parser.add_argument(
        "--multimodal-cuda-device-index",
        type=int,
        default=None,
        help="Optional explicit CUDA device index for single-GPU runs. Disables multi-GPU wrapping for that process.",
    )
    parser.add_argument(
        "--multimodal-selected-modalities",
        default=None,
        help="Optional comma-separated modeled modalities to keep active during multimodal training/evaluation.",
    )
    parser.add_argument(
        "--optuna-trials",
        type=int,
        default=20,
        help="Number of Optuna trials to run for the multimodal tuning stage.",
    )
    parser.add_argument(
        "--optuna-timeout-minutes",
        type=float,
        default=None,
        help="Optional Optuna timeout in minutes. Omit to let all requested trials finish.",
    )
    parser.add_argument(
        "--optuna-study-name",
        default=None,
        help="Optional Optuna study name. Reuse this to resume a previous study.",
    )
    parser.add_argument(
        "--optuna-storage-url",
        default=None,
        help="Optional Optuna storage URL. Defaults to a SQLite study DB under artifacts/optuna/.",
    )
    parser.add_argument(
        "--optuna-workers-per-gpu",
        type=int,
        default=1,
        help="Number of parallel Optuna worker processes to launch per GPU in tune_optuna_parallel mode.",
    )
    parser.add_argument(
        "--optuna-gpus",
        default=None,
        help="Comma-separated GPU indices to use for tune_optuna_parallel. Defaults to all visible CUDA GPUs.",
    )
    parser.add_argument(
        "--ablation-strategy",
        choices=["forward", "backward"],
        default="backward",
        help="Greedy modality-ablation strategy to use in ablate_modalities mode.",
    )
    parser.add_argument(
        "--ablation-group-scheme",
        choices=["device", "modality"],
        default="device",
        help="Whether ablate_modalities should group candidates by device or by individual modeled modality.",
    )
    parser.add_argument(
        "--ablation-study-name",
        default=None,
        help="Optional explicit study name for ablate_modalities outputs.",
    )
    parser.add_argument(
        "--ablation-min-groups",
        type=int,
        default=1,
        help="Minimum number of groups to keep in greedy ablate_modalities search.",
    )
    parser.add_argument(
        "--ablation-max-groups",
        type=int,
        default=None,
        help="Maximum number of groups to keep in greedy ablate_modalities search.",
    )
    parser.add_argument(
        "--ablation-workers-per-gpu",
        type=int,
        default=1,
        help="Number of concurrent modality-ablation runs to launch per GPU at each greedy step.",
    )
    parser.add_argument(
        "--ablation-gpus",
        default=None,
        help="Comma-separated GPU indices to use for ablate_modalities. Defaults to all visible CUDA GPUs when available.",
    )
    parser.add_argument(
        "--trial-matrix-config",
        default=None,
        help="Path to a JSON config describing a list of multimodal trials to run.",
    )
    parser.add_argument(
        "--trial-matrix-study-name",
        default=None,
        help="Optional explicit study name for trial_matrix outputs.",
    )
    parser.add_argument(
        "--trial-matrix-workers-per-gpu",
        type=int,
        default=None,
        help="Number of concurrent trial_matrix runs to launch per GPU.",
    )
    parser.add_argument(
        "--trial-matrix-gpus",
        default=None,
        help="Comma-separated GPU indices to use for trial_matrix. Defaults to all visible CUDA GPUs when available.",
    )
    parser.add_argument(
        "--trial-matrix-optuna-trials",
        type=int,
        default=None,
        help="Default number of Optuna trials to run per subset when using trial_matrix_optuna.",
    )
    parser.add_argument(
        "--trial-matrix-optuna-timeout-minutes",
        type=float,
        default=None,
        help="Optional default Optuna timeout in minutes per subset when using trial_matrix_optuna.",
    )
    return parser.parse_args()


def _print_config_and_models(config) -> None:
    baseline_cfg = config.baseline.consensus_zero_shot
    payload = {
        "run_id": config.run_id,
        "data_root": str(config.data_root),
        "artifact_root": str(config.artifact_root),
        "window_size_sec": config.window_size_sec,
        "stride_sec": config.stride_sec,
        "window_generation_mode": config.window_generation_mode,
        "exclude_video_uid_patterns": list(config.exclude_video_uid_patterns),
        "short_interval_policy": config.short_interval_policy,
        "absolute_time_mode": config.absolute_time_mode,
        "exclude_p2_in_absolute_mode": config.exclude_p2_in_absolute_mode,
        "config_hash": config.config_hash(),
        "multimodal_train": {
            "use_multi_gpu": config.multimodal_train.use_multi_gpu,
            "cuda_device_index": config.multimodal_train.cuda_device_index,
            "use_native_frequency": config.multimodal_train.use_native_frequency,
            "task_mode": config.multimodal_train.task_mode,
            "split_mode": config.multimodal_train.split_mode,
            "num_folds": config.multimodal_train.num_folds,
            "participant_folds": [list(fold) for fold in config.multimodal_train.participant_folds],
            "target_points": config.multimodal_train.target_points,
            "epochs": config.multimodal_train.epochs,
            "batch_size": config.multimodal_train.batch_size,
            "modality_dropout_prob": config.multimodal_train.modality_dropout_prob,
            "complete_windows_only": config.multimodal_train.complete_windows_only,
            "complete_windows_reference": config.multimodal_train.complete_windows_reference,
            "use_context": config.multimodal_train.use_context,
            "selected_modeled_modalities": list(config.multimodal_train.selected_modeled_modalities),
            "learning_rate": config.multimodal_train.learning_rate,
            "weight_decay": config.multimodal_train.weight_decay,
            "lambda_binary": config.multimodal_train.lambda_binary,
            "lambda_ordinal": config.multimodal_train.lambda_ordinal,
            "lambda_regression": config.multimodal_train.lambda_regression,
            "embedding_dim": config.multimodal_train.embedding_dim,
            "fusion_hidden_dim": config.multimodal_train.fusion_hidden_dim,
            "device": config.multimodal_train.device,
        },
        "baseline_consensus_zero_shot": {
            "enabled": baseline_cfg.enabled,
            "model": baseline_cfg.model,
            "provider": baseline_cfg.provider,
            "temperature": baseline_cfg.temperature,
            "num_ctx": baseline_cfg.num_ctx,
            "max_concurrent_samples": baseline_cfg.max_concurrent_samples,
            "max_concurrent_model_calls": baseline_cfg.max_concurrent_model_calls,
            "modalities": dict(sorted(baseline_cfg.modalities.items())),
            "log_raw_prompts": baseline_cfg.log_raw_prompts,
            "zero_shot": baseline_cfg.zero_shot,
            "examples_per_class": baseline_cfg.examples_per_class,
            "consensus_repo_path": baseline_cfg.consensus_repo_path,
            "consensus_commit": baseline_cfg.consensus_commit,
            "baseline_config_hash": config.baseline_config_hash(),
        },
    }
    print("Validated config:")
    print(json.dumps(payload, indent=2, sort_keys=True))

    print("\nModel availability:")
    print(json.dumps(model_availability(config), indent=2, sort_keys=True))


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    loaded_env_files = load_env_files(REPO_ROOT)
    if loaded_env_files:
        logging.getLogger(__name__).info(
            "Loaded environment from: %s",
            ", ".join(str(p) for p in loaded_env_files),
        )

    config = build_default_config(repo_root=REPO_ROOT)
    if args.run_id:
        config.run_id = args.run_id
    if args.allow_relative_time_fallback:
        config.absolute_time_mode = False
    if args.include_p2_absolute:
        config.exclude_p2_in_absolute_mode = False
    if args.provider:
        config.baseline.consensus_zero_shot.provider = str(args.provider)
    if args.model:
        config.baseline.consensus_zero_shot.model = str(args.model)
    if args.window_generation_mode:
        config.window_generation_mode = str(args.window_generation_mode)
    if args.window_size_sec is not None:
        config.window_size_sec = float(args.window_size_sec)
    if args.stride_sec is not None:
        config.stride_sec = float(args.stride_sec)
    if args.openai_base_url:
        os.environ["OPENAI_BASE_URL"] = str(args.openai_base_url)
    if args.openai_api_key:
        os.environ["OPENAI_API_KEY"] = str(args.openai_api_key)
    if args.litellm_base_url:
        os.environ["LITELLM_BASE_URL"] = str(args.litellm_base_url)
    if args.litellm_api_key:
        os.environ["LITELLM_API_KEY"] = str(args.litellm_api_key)
    if args.multimodal_target_points is not None:
        config.multimodal_train.target_points = int(args.multimodal_target_points)
    if args.multimodal_use_native_frequency:
        config.multimodal_train.use_native_frequency = True
    if args.multimodal_disable_multi_gpu:
        config.multimodal_train.use_multi_gpu = False
    if args.multimodal_use_resampled_grid:
        config.multimodal_train.use_native_frequency = False
    if args.multimodal_task_mode:
        config.multimodal_train.task_mode = str(args.multimodal_task_mode)
    if args.multimodal_split_mode:
        config.multimodal_train.split_mode = str(args.multimodal_split_mode)
    if args.multimodal_num_folds is not None:
        config.multimodal_train.num_folds = int(args.multimodal_num_folds)
    if args.multimodal_epochs is not None:
        config.multimodal_train.epochs = int(args.multimodal_epochs)
    if args.multimodal_batch_size is not None:
        config.multimodal_train.batch_size = int(args.multimodal_batch_size)
    if args.multimodal_modality_dropout_prob is not None:
        config.multimodal_train.modality_dropout_prob = float(args.multimodal_modality_dropout_prob)
    if args.multimodal_complete_windows_only:
        config.multimodal_train.complete_windows_only = True
    if args.multimodal_complete_windows_reference:
        config.multimodal_train.complete_windows_reference = str(
            args.multimodal_complete_windows_reference
        )
    if args.multimodal_disable_context:
        config.multimodal_train.use_context = False
    if args.multimodal_selected_modalities:
        config.multimodal_train.selected_modeled_modalities = tuple(
            token.strip()
            for token in str(args.multimodal_selected_modalities).split(",")
            if token.strip()
        )
    if args.multimodal_learning_rate is not None:
        config.multimodal_train.learning_rate = float(args.multimodal_learning_rate)
    if args.multimodal_weight_decay is not None:
        config.multimodal_train.weight_decay = float(args.multimodal_weight_decay)
    if args.multimodal_lambda_binary is not None:
        config.multimodal_train.lambda_binary = float(args.multimodal_lambda_binary)
    if args.multimodal_lambda_ordinal is not None:
        config.multimodal_train.lambda_ordinal = float(args.multimodal_lambda_ordinal)
    if args.multimodal_lambda_regression is not None:
        config.multimodal_train.lambda_regression = float(args.multimodal_lambda_regression)
    if args.multimodal_embedding_dim is not None:
        config.multimodal_train.embedding_dim = int(args.multimodal_embedding_dim)
    if args.multimodal_fusion_hidden_dim is not None:
        config.multimodal_train.fusion_hidden_dim = int(args.multimodal_fusion_hidden_dim)
    if args.multimodal_device:
        config.multimodal_train.device = str(args.multimodal_device)
    if args.multimodal_cuda_device_index is not None:
        config.multimodal_train.cuda_device_index = int(args.multimodal_cuda_device_index)
        config.multimodal_train.use_multi_gpu = False
    if args.stage == "ablate_modalities":
        apply_complete_only_regression_preset(config)
        config.multimodal_train.use_multi_gpu = False
    if args.stage == "trial_matrix":
        config.multimodal_train.use_multi_gpu = False
    if args.stage == "trial_matrix_optuna":
        config.multimodal_train.use_multi_gpu = False

    validate_core_config(config)
    if args.stage == "baseline_consensus_zs":
        validate_baseline_config(config)
        logging.getLogger(__name__).info("Applied core + baseline validation rules.")
    else:
        logging.getLogger(__name__).info("Applied core validation rules.")
    config.ensure_directories()
    persist_run_metadata(config)

    _print_config_and_models(config)

    manifest_df: pd.DataFrame | None = None
    windows_df: pd.DataFrame | None = None
    embedding_df: pd.DataFrame | None = None

    if args.stage in {"data", "all"}:
        manifest_df, alignment_df, outputs_b = run_phase_b(config)
        print("\nPhase B complete:")
        print(f"- sessions discovered: {len(manifest_df)}")
        print(f"- alignment rows: {len(alignment_df)}")
        print(json.dumps({k: str(v) for k, v in outputs_b.items()}, indent=2, sort_keys=True))

    if args.stage in {"labels", "all"}:
        report_events_df, windows_df, outputs_c = run_phase_c(config, session_manifest=manifest_df)
        print("\nPhase C complete:")
        print(f"- report events: {len(report_events_df)}")
        print(f"- windows: {len(windows_df)}")
        print(json.dumps({k: str(v) for k, v in outputs_c.items()}, indent=2, sort_keys=True))

    if args.stage in {"features", "all"}:
        embedding_df, outputs_d, stats_d = run_phase_d(
            config,
            session_manifest=manifest_df,
            window_index=windows_df,
            force_recompute=args.force_recompute_embeddings,
        )
        print("\nPhase D complete:")
        print(f"- embeddings: {len(embedding_df)}")
        print(json.dumps({k: str(v) for k, v in outputs_d.items()}, indent=2, sort_keys=True))
        print(json.dumps(stats_d, indent=2, sort_keys=True))

    if args.stage in {"preprocessed_data", "all"}:
        preprocessed_df, outputs_p = run_phase_preprocessed_data(
            config,
            session_manifest=manifest_df,
            window_index=windows_df,
        )
        print("\nPhase P complete:")
        print(f"- exported modality slices: {len(preprocessed_df)}")
        print(json.dumps({k: str(v) for k, v in outputs_p.items()}, indent=2, sort_keys=True))

    if args.stage in {"train_eval", "all"}:
        predictions_df, outputs_e, stats_e = run_phase_e(
            config,
            window_index=windows_df,
            embedding_table=embedding_df,
        )
        print("\nPhase E complete:")
        print(f"- prediction rows: {len(predictions_df)}")
        print(json.dumps({k: str(v) for k, v in outputs_e.items()}, indent=2, sort_keys=True))
        print(json.dumps(stats_e, indent=2, sort_keys=True))

    if args.stage == "ablate_modalities":
        ablation_config = config
        outputs_a, stats_a = run_modality_ablation_greedy(
            ablation_config,
            strategy=args.ablation_strategy,
            study_name=args.ablation_study_name,
            group_scheme=args.ablation_group_scheme,
            min_groups=int(args.ablation_min_groups),
            max_groups=args.ablation_max_groups,
            workers_per_gpu=int(args.ablation_workers_per_gpu),
            gpu_list=args.ablation_gpus,
        )
        print("\nModality ablation complete:")
        print(json.dumps({k: str(v) for k, v in outputs_a.items()}, indent=2, sort_keys=True))
        print(json.dumps(stats_a, indent=2, sort_keys=True))

    if args.stage == "trial_matrix":
        if not args.trial_matrix_config:
            raise ValueError("--trial-matrix-config is required when --stage trial_matrix is used.")
        outputs_tm, stats_tm = run_trial_matrix(
            config,
            config_path=args.trial_matrix_config,
            study_name=args.trial_matrix_study_name,
            workers_per_gpu=args.trial_matrix_workers_per_gpu,
            gpu_list=args.trial_matrix_gpus,
        )
        print("\nTrial matrix complete:")
        print(json.dumps({k: str(v) for k, v in outputs_tm.items()}, indent=2, sort_keys=True))
        print(json.dumps(stats_tm, indent=2, sort_keys=True))

    if args.stage == "trial_matrix_optuna":
        if not args.trial_matrix_config:
            raise ValueError("--trial-matrix-config is required when --stage trial_matrix_optuna is used.")
        outputs_tmo, stats_tmo = run_trial_matrix_optuna(
            config,
            config_path=args.trial_matrix_config,
            study_name=args.trial_matrix_study_name,
            workers_per_gpu=args.trial_matrix_workers_per_gpu,
            gpu_list=args.trial_matrix_gpus,
            n_trials=args.trial_matrix_optuna_trials,
            timeout_minutes=args.trial_matrix_optuna_timeout_minutes,
        )
        print("\nTrial matrix Optuna complete:")
        print(json.dumps({k: str(v) for k, v in outputs_tmo.items()}, indent=2, sort_keys=True))
        print(json.dumps(stats_tmo, indent=2, sort_keys=True))

    if args.stage == "tune_optuna":
        outputs_o, stats_o = run_optuna_multimodal_study(
            config,
            n_trials=int(args.optuna_trials),
            timeout_minutes=args.optuna_timeout_minutes,
            study_name=args.optuna_study_name,
            storage_url=args.optuna_storage_url,
        )
        print("\nOptuna tuning complete:")
        print(json.dumps({k: str(v) for k, v in outputs_o.items()}, indent=2, sort_keys=True))
        print(json.dumps(stats_o, indent=2, sort_keys=True))

    if args.stage == "tune_optuna_parallel":
        outputs_op, stats_op = run_optuna_multimodal_parallel(
            config,
            n_trials=int(args.optuna_trials),
            workers_per_gpu=int(args.optuna_workers_per_gpu),
            gpu_list=args.optuna_gpus,
            timeout_minutes=args.optuna_timeout_minutes,
            study_name=args.optuna_study_name,
            storage_url=args.optuna_storage_url,
        )
        print("\nOptuna parallel tuning complete:")
        print(json.dumps({k: str(v) for k, v in outputs_op.items()}, indent=2, sort_keys=True))
        print(json.dumps(stats_op, indent=2, sort_keys=True))

    if args.stage == "baseline_consensus_zs":
        outputs_z1, stats_z1 = run_phase_z1_baseline_consensus(config)
        print("\nPhase Z1 complete:")
        print(json.dumps({k: str(v) for k, v in outputs_z1.items()}, indent=2, sort_keys=True))
        print(json.dumps(stats_z1, indent=2, sort_keys=True))

    if args.stage == "config":
        print("\nConfig and model availability generated successfully.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


