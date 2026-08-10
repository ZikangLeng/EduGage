from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Literal

ShortIntervalPolicy = Literal["skip", "pad"]
IntervalBoundaryMode = Literal["half_open"]
WindowGenerationMode = Literal["interval_sliding", "event_trailing"]
DevicePreference = Literal["auto", "cpu", "cuda", "mps"]
TaskMode = Literal["multitask", "binary", "ordinal"]
SplitMode = Literal["loso", "grouped_kfold", "fixed_groups"]
CompleteWindowsReference = Literal["selected", "all"]


@dataclass(frozen=True)
class ExternalModelSpec:
    """Pinned metadata for external encoders used in V1."""

    name: str
    repo_url: str
    commit: str
    checkpoint_path: str
    expected_input: str


@dataclass
class ConsensusZeroShotConfig:
    """Configuration for the ConSensus engagement baseline."""

    enabled: bool = False
    model: str = "openai:gpt-5-mini"
    provider: str = "openai"
    temperature: float = 0.0
    num_ctx: int = 15000
    max_concurrent_samples: int = 4
    max_concurrent_model_calls: int = 4
    modalities: dict[str, bool] = field(
        default_factory=lambda: {
            "eeg": True,
            "acc": True,
            "gyro": True,
            "ppg": True,
            "ring": True,
            "ecg": True,
            "eda": True,
            "hr": False,
            "eye": True,
            "esense": True,
        }
    )
    devices: dict[str, bool] = field(
        default_factory=lambda: {
            "muse_headband": True,
            "esense_left_earbud": True,
            "polar_ecg_cheststrap": True,
            "msband_right_wrist": True,
            "ring_right_finger": True,
            "beam_eye_tracker": True,
            "unknown_device": True,
        }
    )
    log_raw_prompts: bool = False
    zero_shot: bool = True
    examples_per_class: int = 0
    consensus_repo_path: str = ""
    consensus_commit: str = "unknown"


@dataclass
class BaselineConfig:
    """Container for baseline-specific configs."""

    consensus_zero_shot: ConsensusZeroShotConfig = field(
        default_factory=ConsensusZeroShotConfig
    )


@dataclass
class MultimodalTrainConfig:
    """Configuration for the raw-window multimodal Phase E backend."""

    use_native_frequency: bool = True
    use_multi_gpu: bool = True
    cuda_device_index: int | None = None
    task_mode: TaskMode = "ordinal"
    split_mode: SplitMode = "fixed_groups"
    num_folds: int = 4
    participant_folds: tuple[tuple[int, ...], ...] = (
        (1, 2, 3, 10),
        (4, 5, 7, 9),
        (6, 8, 11, 12),
        (13, 14, 15, 16),
    )
    target_points: int = 192
    epochs: int = 20
    validation_fraction: float = 0.2
    early_stopping_patience: int = 8
    batch_size: int = 16
    modality_dropout_prob: float = 0.15
    complete_windows_only: bool = False
    complete_windows_reference: CompleteWindowsReference = "selected"
    use_context: bool = True
    selected_modeled_modalities: tuple[str, ...] = ()
    learning_rate: float = 6.079002903362803e-4
    weight_decay: float = 8.026036272969121e-5
    lambda_binary: float = 1.0
    lambda_ordinal: float = 1.0
    lambda_regression: float = 1.0
    embedding_dim: int = 32
    fusion_hidden_dim: int = 64
    device: DevicePreference = "auto"


@dataclass
class RunConfig:
    """Central run configuration for the lean V1 pipeline."""

    repo_root: Path
    data_root: Path
    artifact_root: Path
    external_models_root: Path
    run_id: str

    seed: int = 42
    window_size_sec: float = 44.0
    stride_sec: float = 44.0
    window_generation_mode: WindowGenerationMode = "event_trailing"

    exclude_video_uid_patterns: tuple[str, ...] = ("training",)
    short_interval_policy: ShortIntervalPolicy = "skip"
    interval_boundary_mode: IntervalBoundaryMode = "half_open"
    require_single_label_per_window: bool = True

    absolute_time_mode: bool = True
    exclude_p2_in_absolute_mode: bool = False

    enabled_modalities: dict[str, bool] = field(
        default_factory=lambda: {
            "eeg": True,
            "acc": True,
            "gyro": True,
            "ppg": True,
            "ring": True,
            "ecg": True,
            "eda": True,
            "hr": True,
            "eye": True,
            "markers": True,
            "esense": True,
        }
    )
    model_registry: dict[str, ExternalModelSpec] = field(default_factory=dict)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    multimodal_train: MultimodalTrainConfig = field(default_factory=MultimodalTrainConfig)

    @property
    def run_dir(self) -> Path:
        return self.artifact_root / "runs" / self.run_id

    @property
    def manifests_dir(self) -> Path:
        return self.artifact_root / "manifests"

    @property
    def windows_dir(self) -> Path:
        return self.artifact_root / "windows"

    @property
    def embeddings_dir(self) -> Path:
        return self.artifact_root / "embeddings"

    def ensure_directories(self) -> None:
        for path in (
            self.artifact_root,
            self.run_dir,
            self.manifests_dir,
            self.windows_dir,
            self.embeddings_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["repo_root"] = str(self.repo_root)
        payload["data_root"] = str(self.data_root)
        payload["artifact_root"] = str(self.artifact_root)
        payload["external_models_root"] = str(self.external_models_root)
        payload["model_registry"] = {
            name: asdict(spec) for name, spec in self.model_registry.items()
        }
        return payload

    def config_hash(self) -> str:
        serialized = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:12]

    def baseline_config_hash(self) -> str:
        baseline_payload = asdict(self.baseline)
        serialized = json.dumps(
            baseline_payload, sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:12]


def _default_model_registry() -> dict[str, ExternalModelSpec]:
    return {
        "eeg": ExternalModelSpec(
            name="EEGPT",
            repo_url="https://github.com/BINE022/EEGPT",
            commit="a0e0a8fad729e2ecf4eedb3a81548a6e6d48a705",
            checkpoint_path="eegpt/model.ckpt",
            expected_input="[channels, time] preprocessed EEG segments",
        ),
        "ecg": ExternalModelSpec(
            name="ECGFounder",
            repo_url="https://github.com/PKUDigitalHealth/ECGFounder",
            commit="04edac702b61c91face519774ddcc0cd712fef23",
            checkpoint_path="ecgfounder/model.ckpt",
            expected_input="1D ECG waveform segment",
        ),
        "ppg": ExternalModelSpec(
            name="PulsePPG",
            repo_url="https://github.com/maxxu05/pulseppg",
            commit="716eaf9cf966e8f76436f2263872ef38b1f90166",
            checkpoint_path="pulseppg/model.ckpt",
            expected_input="PPG segment aligned to window",
        ),
        "eda_imu": ExternalModelSpec(
            name="NormWear",
            repo_url="https://github.com/Mobile-Sensing-and-UbiComp-Laboratory/NormWear",
            commit="f3bc7439efcf30bf9df36566f87b35a277f9cc5e",
            checkpoint_path="normwear/model.ckpt",
            expected_input="EDA + IMU synchronized segment",
        ),
        "eye": ExternalModelSpec(
            name="EyeFeaturesMLP",
            repo_url="local",
            commit="local",
            checkpoint_path="eye/projector.ckpt",
            expected_input="engineered gaze/head features",
        ),
    }


def _build_run_id() -> str:
    return datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")


def _normalize_commit(value: str) -> str:
    candidate = value.strip().split()[0] if value.strip() else ""
    if len(candidate) != 40:
        return ""
    valid_chars = set("0123456789abcdefABCDEF")
    if any(ch not in valid_chars for ch in candidate):
        return ""
    return candidate.lower()


def _resolve_git_commit(repo_path: Path) -> str:
    if not repo_path.exists():
        return "missing_repo"

    try:
        result = subprocess.run(
            ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, OSError):
        return "unknown"

    if result.returncode != 0:
        return "unknown"

    commit = _normalize_commit(result.stdout)
    return commit or "unknown"

def _default_baseline_config(repo_root: Path) -> BaselineConfig:
    consensus_repo = repo_root / "consensus"
    return BaselineConfig(
        consensus_zero_shot=ConsensusZeroShotConfig(
            consensus_repo_path=str(consensus_repo),
            consensus_commit=_resolve_git_commit(consensus_repo),
        )
    )


def build_default_config(repo_root: Path | None = None) -> RunConfig:
    root = (repo_root or Path.cwd()).resolve()
    config = RunConfig(
        repo_root=root,
        data_root=root / "data",
        artifact_root=root / "artifacts",
        external_models_root=root / "external_models",
        run_id=_build_run_id(),
        model_registry=_default_model_registry(),
        baseline=_default_baseline_config(root),
    )
    validate_core_config(config)
    return config


def validate_core_config(config: RunConfig) -> None:
    valid_modeled_modalities = {
        "eeg",
        "ecg",
        "ppg",
        "eda",
        "eye",
        "imu_muse",
        "imu_esense",
        "hr",
        "ring_ppg",
        "ring_temp",
        "ring_imu",
    }
    if config.window_generation_mode not in {"interval_sliding", "event_trailing"}:
        raise ValueError(
            f"Invalid window_generation_mode: {config.window_generation_mode}"
        )

    if config.short_interval_policy not in {"skip", "pad"}:
        raise ValueError(f"Invalid short_interval_policy: {config.short_interval_policy}")

    if config.interval_boundary_mode != "half_open":
        raise ValueError(
            "Only half-open interval boundaries are currently supported: [start, end)."
        )

    if config.window_size_sec <= 0:
        raise ValueError("window_size_sec must be positive.")

    if config.stride_sec <= 0:
        raise ValueError("stride_sec must be positive.")

    if config.multimodal_train.target_points <= 0:
        raise ValueError("multimodal_train.target_points must be positive.")
    if not isinstance(config.multimodal_train.use_multi_gpu, bool):
        raise ValueError("multimodal_train.use_multi_gpu must be a boolean.")
    cuda_device_index = config.multimodal_train.cuda_device_index
    if cuda_device_index is not None:
        if not isinstance(cuda_device_index, int):
            raise ValueError("multimodal_train.cuda_device_index must be an integer or None.")
        if cuda_device_index < 0:
            raise ValueError("multimodal_train.cuda_device_index must be >= 0.")
    if config.multimodal_train.task_mode not in {"multitask", "binary", "ordinal"}:
        raise ValueError(
            f"Invalid multimodal_train.task_mode: {config.multimodal_train.task_mode}"
        )
    if config.multimodal_train.split_mode not in {"loso", "grouped_kfold", "fixed_groups"}:
        raise ValueError(
            f"Invalid multimodal_train.split_mode: {config.multimodal_train.split_mode}"
        )
    if config.multimodal_train.num_folds <= 1:
        raise ValueError("multimodal_train.num_folds must be >= 2.")
    if config.multimodal_train.split_mode == "fixed_groups":
        participant_folds = tuple(
            tuple(int(participant_id) for participant_id in fold)
            for fold in config.multimodal_train.participant_folds
        )
        if len(participant_folds) < 2:
            raise ValueError("multimodal_train.participant_folds must contain at least 2 folds.")
        seen_participants: set[int] = set()
        for fold in participant_folds:
            if not fold:
                raise ValueError("multimodal_train.participant_folds cannot contain an empty fold.")
            for participant_id in fold:
                if participant_id in seen_participants:
                    raise ValueError(
                        "multimodal_train.participant_folds must be participant-disjoint."
                    )
                seen_participants.add(participant_id)
    if config.multimodal_train.epochs <= 0:
        raise ValueError("multimodal_train.epochs must be positive.")
    if not 0.0 < config.multimodal_train.validation_fraction < 1.0:
        raise ValueError("multimodal_train.validation_fraction must be in (0, 1).")
    if config.multimodal_train.early_stopping_patience <= 0:
        raise ValueError("multimodal_train.early_stopping_patience must be positive.")
    if config.multimodal_train.batch_size <= 0:
        raise ValueError("multimodal_train.batch_size must be positive.")
    if config.multimodal_train.learning_rate <= 0:
        raise ValueError("multimodal_train.learning_rate must be positive.")
    if config.multimodal_train.weight_decay < 0:
        raise ValueError("multimodal_train.weight_decay must be >= 0.")
    if config.multimodal_train.modality_dropout_prob < 0 or config.multimodal_train.modality_dropout_prob >= 1:
        raise ValueError("multimodal_train.modality_dropout_prob must be in [0, 1).")
    if config.multimodal_train.complete_windows_reference not in {"selected", "all"}:
        raise ValueError(
            "multimodal_train.complete_windows_reference must be 'selected' or 'all'."
        )
    if config.multimodal_train.lambda_binary < 0:
        raise ValueError("multimodal_train.lambda_binary must be >= 0.")
    if config.multimodal_train.lambda_ordinal < 0:
        raise ValueError("multimodal_train.lambda_ordinal must be >= 0.")
    if config.multimodal_train.lambda_regression < 0:
        raise ValueError("multimodal_train.lambda_regression must be >= 0.")
    if config.multimodal_train.embedding_dim <= 0:
        raise ValueError("multimodal_train.embedding_dim must be positive.")
    if config.multimodal_train.fusion_hidden_dim <= 0:
        raise ValueError("multimodal_train.fusion_hidden_dim must be positive.")
    if config.multimodal_train.device not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError(
            f"Invalid multimodal_train.device: {config.multimodal_train.device}"
        )
    selected_modeled_modalities = tuple(
        str(modality).strip()
        for modality in config.multimodal_train.selected_modeled_modalities
        if str(modality).strip()
    )
    if len(selected_modeled_modalities) != len(set(selected_modeled_modalities)):
        raise ValueError("multimodal_train.selected_modeled_modalities must not contain duplicates.")
    invalid_modeled_modalities = sorted(
        modality
        for modality in selected_modeled_modalities
        if modality not in valid_modeled_modalities
    )
    if invalid_modeled_modalities:
        raise ValueError(
            "multimodal_train.selected_modeled_modalities contains unsupported entries: "
            + ", ".join(invalid_modeled_modalities)
        )

    if not config.data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {config.data_root}")


def validate_baseline_config(config: RunConfig) -> None:
    baseline_cfg = config.baseline.consensus_zero_shot
    if baseline_cfg.examples_per_class < 0:
        raise ValueError(
            "baseline.consensus_zero_shot.examples_per_class must be >= 0."
        )
    expected_zero_shot = bool(int(baseline_cfg.examples_per_class) <= 0)
    if bool(baseline_cfg.zero_shot) != expected_zero_shot:
        raise ValueError(
            "baseline.consensus_zero_shot.zero_shot is inconsistent with examples_per_class. "
            "Use zero_shot=true when examples_per_class=0; otherwise set zero_shot=false."
        )
    if baseline_cfg.max_concurrent_samples <= 0:
        raise ValueError(
            "baseline.consensus_zero_shot.max_concurrent_samples must be positive."
        )
    if baseline_cfg.max_concurrent_model_calls <= 0:
        raise ValueError(
            "baseline.consensus_zero_shot.max_concurrent_model_calls must be positive."
        )
    if baseline_cfg.num_ctx <= 0:
        raise ValueError("baseline.consensus_zero_shot.num_ctx must be positive.")

    if not any(bool(v) for v in baseline_cfg.modalities.values()):
        raise ValueError("baseline.consensus_zero_shot.modalities must enable at least one modality.")

    if not any(bool(v) for v in baseline_cfg.devices.values()):
        raise ValueError("baseline.consensus_zero_shot.devices must enable at least one device.")


def model_availability(config: RunConfig) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for modality, spec in config.model_registry.items():
        checkpoint = config.external_models_root / spec.checkpoint_path
        rows.append(
            {
                "modality": modality,
                "model_name": spec.name,
                "repo_url": spec.repo_url,
                "commit": spec.commit,
                "checkpoint": str(checkpoint),
                "checkpoint_exists": checkpoint.exists(),
            }
        )
    return rows


def persist_run_metadata(config: RunConfig) -> None:
    config.ensure_directories()
    metadata_path = config.run_dir / "config.json"
    with metadata_path.open("w", encoding="utf-8") as fp:
        json.dump(config.to_dict(), fp, indent=2, sort_keys=True)

    availability_path = config.run_dir / "model_availability.json"
    with availability_path.open("w", encoding="utf-8") as fp:
        json.dump(model_availability(config), fp, indent=2, sort_keys=True)

    baseline_metadata = {
        "consensus_repo_path": config.baseline.consensus_zero_shot.consensus_repo_path,
        "consensus_commit": config.baseline.consensus_zero_shot.consensus_commit,
        "baseline_config_hash": config.baseline_config_hash(),
    }
    baseline_metadata_path = config.run_dir / "baseline_metadata.json"
    with baseline_metadata_path.open("w", encoding="utf-8") as fp:
        json.dump(baseline_metadata, fp, indent=2, sort_keys=True)





