from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


def coral_targets(labels_zero_based: torch.Tensor, num_classes: int) -> torch.Tensor:
    thresholds = torch.arange(num_classes - 1, device=labels_zero_based.device).view(1, -1)
    return (labels_zero_based.view(-1, 1) > thresholds).to(dtype=torch.float32)


def coral_class_probs(logits: torch.Tensor) -> torch.Tensor:
    threshold_probs = torch.sigmoid(logits)
    num_classes = int(logits.shape[1] + 1)

    probs: list[torch.Tensor] = []
    probs.append(1.0 - threshold_probs[:, 0])
    for idx in range(1, num_classes - 1):
        probs.append(threshold_probs[:, idx - 1] - threshold_probs[:, idx])
    probs.append(threshold_probs[:, -1])

    stacked = torch.stack(probs, dim=1)
    return torch.clamp(stacked, min=0.0, max=1.0)


def coral_predict(logits: torch.Tensor) -> torch.Tensor:
    threshold_passes = (logits >= 0.0).to(dtype=torch.int64)
    return torch.sum(threshold_passes, dim=1)


class CORALLoss(nn.Module):
    def __init__(self, pos_weight: torch.Tensor | None = None) -> None:
        super().__init__()
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight.to(dtype=torch.float32))
        else:
            self.pos_weight = None

    def forward(self, logits: torch.Tensor, labels_zero_based: torch.Tensor) -> torch.Tensor:
        targets = coral_targets(labels_zero_based, num_classes=int(logits.shape[1] + 1))
        return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.pos_weight)


class ResidualTemporalBlock(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        padding = dilation
        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=padding,
            dilation=dilation,
        )
        self.norm1 = nn.GroupNorm(1, channels)
        self.conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=padding,
            dilation=dilation,
        )
        self.norm2 = nn.GroupNorm(1, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = x + residual
        return F.gelu(x)


class TemporalSequenceEncoder(nn.Module):
    def __init__(
        self,
        input_channels: int,
        *,
        hidden_channels: int,
        num_blocks: int,
        output_dim: int,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, hidden_channels, kernel_size=7, padding=3),
            nn.GroupNorm(1, hidden_channels),
            nn.GELU(),
        )
        dilations = [2**idx for idx in range(num_blocks)]
        self.blocks = nn.Sequential(
            *[ResidualTemporalBlock(hidden_channels, dilation=d) for d in dilations]
        )
        self.proj = nn.Linear(hidden_channels, output_dim)

    def forward(self, x: torch.Tensor, time_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        if time_mask is None:
            x = x.mean(dim=-1)
        else:
            weights = time_mask.unsqueeze(1).to(dtype=x.dtype)
            denom = torch.clamp(weights.sum(dim=-1), min=1.0)
            x = (x * weights).sum(dim=-1) / denom
        return self.proj(x)


class ContextEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CORALOrdinalHead(nn.Module):
    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.score = nn.Linear(input_dim, 1)
        self.raw_bias = nn.Parameter(torch.zeros((self.num_classes - 1,), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        score = self.score(x)
        ordered_bias = torch.cumsum(F.softplus(self.raw_bias), dim=0)
        return score - ordered_bias.view(1, -1)


@dataclass(frozen=True)
class EncoderSpec:
    hidden_channels: int
    num_blocks: int

def _encoder_spec_for_modality(modality: str) -> EncoderSpec:
    if modality == "eeg":
        return EncoderSpec(hidden_channels=64, num_blocks=4)
    if modality in {"ecg", "ppg", "ring_ppg"}:
        return EncoderSpec(hidden_channels=64, num_blocks=4)
    if modality in {"imu_muse", "imu_esense", "ring_imu"}:
        return EncoderSpec(hidden_channels=64, num_blocks=4)
    if modality == "eye":
        return EncoderSpec(hidden_channels=48, num_blocks=3)
    if modality in {"eda", "hr", "ring_temp"}:
        return EncoderSpec(hidden_channels=32, num_blocks=3)
    return EncoderSpec(hidden_channels=48, num_blocks=3)


class MultimodalGatedFusionModel(nn.Module):
    def __init__(
        self,
        *,
        modality_input_dims: dict[str, int],
        modality_order: tuple[str, ...],
        context_dim: int,
        embedding_dim: int = 64,
        fusion_hidden_dim: int = 96,
        num_classes: int = 5,
    ) -> None:
        super().__init__()
        self.modality_order = modality_order
        self.embedding_dim = int(embedding_dim)
        self.num_classes = int(num_classes)

        self.encoders = nn.ModuleDict()
        self.gates = nn.ModuleDict()
        for modality in modality_order:
            input_dim = int(modality_input_dims[modality])
            spec = _encoder_spec_for_modality(modality)
            self.encoders[modality] = TemporalSequenceEncoder(
                input_channels=input_dim,
                hidden_channels=spec.hidden_channels,
                num_blocks=spec.num_blocks,
                output_dim=self.embedding_dim,
            )
            self.gates[modality] = nn.Sequential(
                nn.Linear(self.embedding_dim + context_dim, fusion_hidden_dim),
                nn.GELU(),
                nn.Linear(fusion_hidden_dim, 1),
            )

        self.context_encoder = ContextEncoder(
            input_dim=context_dim,
            hidden_dim=max(context_dim * 4, 16),
            output_dim=context_dim,
        )
        head_input_dim = self.embedding_dim + context_dim
        self.binary_head = nn.Linear(head_input_dim, 1)
        self.ordinal_head = CORALOrdinalHead(head_input_dim, num_classes=num_classes)
        self.regression_head = nn.Linear(head_input_dim, 1)

    def forward(
        self,
        modality_inputs: dict[str, torch.Tensor],
        modality_time_masks: dict[str, torch.Tensor],
        modality_mask: torch.Tensor,
        context_inputs: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        context_emb = self.context_encoder(context_inputs)

        encoded_modalities: list[torch.Tensor] = []
        gate_values: list[torch.Tensor] = []
        for idx, modality in enumerate(self.modality_order):
            encoder = self.encoders[modality]
            modality_input = modality_inputs[modality]
            time_mask = modality_time_masks.get(modality)
            if modality_input.shape[-1] == 0 or (
                time_mask is not None and float(time_mask.sum().detach().cpu()) <= 0.0
            ):
                batch_size = int(modality_input.shape[0])
                z_m = torch.zeros(
                    (batch_size, self.embedding_dim),
                    device=modality_input.device,
                    dtype=modality_input.dtype,
                )
            else:
                z_m = encoder(
                    modality_input,
                    time_mask,
                )
            gate_logits = self.gates[modality](torch.cat([z_m, context_emb], dim=1))
            gate = torch.sigmoid(gate_logits) * modality_mask[:, idx : idx + 1]
            encoded_modalities.append(z_m)
            gate_values.append(gate)

        stacked_embeddings = torch.stack(encoded_modalities, dim=1)
        stacked_gates = torch.stack(gate_values, dim=1)
        denom = torch.clamp(stacked_gates.sum(dim=1), min=1e-6)
        fused = (stacked_embeddings * stacked_gates).sum(dim=1) / denom

        head_input = torch.cat([fused, context_emb], dim=1)
        binary_logits = self.binary_head(head_input).squeeze(-1)
        ordinal_logits = self.ordinal_head(head_input)
        ordinal_proba = coral_class_probs(ordinal_logits)
        regression_score = self.regression_head(head_input).squeeze(-1)

        return {
            "context_embedding": context_emb,
            "fused_embedding": fused,
            "gates": stacked_gates.squeeze(-1),
            "binary_logits": binary_logits,
            "ordinal_logits": ordinal_logits,
            "ordinal_proba": ordinal_proba,
            "regression_score": regression_score,
        }
