"""Official-source architecture ports with ordinal heads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F


ArchitectureName = Literal["deepconvlstm", "deepconvlstm_attention", "tinyhar"]

OFFICIAL_ARCHITECTURE_SOURCES = {
    "deepconvlstm": {
        "repo": "https://github.com/STRCWearlab/DeepConvLSTM",
        "commit": "ea25e34da3a48b151a97bd3502802db547523b7f",
        "file": "DeepConvLSTM.ipynb",
        "notes": "Port of the notebook architecture: four valid 5x1 conv layers, two 128-unit LSTMs.",
    },
    "deepconvlstm_attention": {
        "repo": "https://github.com/isukrit/encodingHumanActivity",
        "commit": "4d32acebff6374d02bbbb7183fd970ff835e81e6",
        "file": "codes/model_proposed/model_with_self_attn.py and layers.py",
        "notes": "Port of CNN + LSTM + multi-hop self-attention with F=32, D=10.",
    },
    "tinyhar": {
        "repo": "https://github.com/teco-kit/ISWC22-HAR",
        "commit": "b84b89d09f6914fe93e82cde423e294042da2741",
        "file": "models/TinyHAR.py and notebooks/model/Train model.ipynb",
        "notes": "Port of TinyHAR_Model with notebook settings: attn, FC channel aggregation, LSTM, tnaive temporal aggregation.",
    },
}


def ordinal_targets(y: torch.Tensor, n_classes: int) -> torch.Tensor:
    if n_classes < 2:
        raise ValueError("n_classes must be >= 2.")
    y = y.to(dtype=torch.long).view(-1, 1)
    thresholds = torch.arange(n_classes - 1, device=y.device).view(1, -1)
    return (y > thresholds).to(dtype=torch.float32)


def ordinal_loss(
    logits: torch.Tensor,
    y: torch.Tensor,
    *,
    n_classes: int,
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    if logits.ndim != 2 or logits.shape[1] != n_classes - 1:
        raise ValueError(f"logits must have shape [batch, {n_classes - 1}].")
    targets = ordinal_targets(y, n_classes=n_classes)
    return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)


def ordinal_logits_to_proba(logits: torch.Tensor) -> torch.Tensor:
    gt = torch.sigmoid(logits)
    gt = torch.cummin(gt, dim=1).values
    p0 = 1.0 - gt[:, :1]
    middle = gt[:, :-1] - gt[:, 1:]
    plast = gt[:, -1:]
    proba = torch.cat([p0, middle, plast], dim=1)
    denom = proba.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return proba.clamp_min(0.0) / denom


class OfficialDeepConvLSTM(nn.Module):
    """PyTorch port of STRCWearlab/DeepConvLSTM notebook architecture."""

    def __init__(
        self,
        *,
        in_channels: int = 28,
        n_classes: int = 5,
        conv_filters: int = 64,
        filter_size: int = 5,
        lstm_units: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.conv_layers = nn.ModuleList()
        for idx in range(4):
            input_filters = 1 if idx == 0 else conv_filters
            self.conv_layers.append(nn.Conv2d(input_filters, conv_filters, kernel_size=(filter_size, 1)))
        self.lstm_layers = nn.ModuleList()
        for idx in range(lstm_layers):
            input_size = conv_filters * in_channels if idx == 0 else lstm_units
            self.lstm_layers.append(nn.LSTM(input_size, lstm_units, batch_first=True))
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(lstm_units, n_classes - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2).unsqueeze(1)
        # [B, 1, T, C], matching the official notebook's Conv2D 5x1 stack.
        for conv in self.conv_layers:
            x = F.relu(conv(x))

        bsz, filters, seq_len, channels = x.shape
        x = x.permute(0, 2, 1, 3).reshape(bsz, seq_len, filters * channels)
        for lstm in self.lstm_layers:
            x, _hidden = lstm(x)
        x = self.dropout(x[:, -1, :])
        return self.head(x)


class OfficialMultiHopSelfAttention(nn.Module):
    """Port of encodingHumanActivity SelfAttention(size=F, num_hops=D)."""

    def __init__(self, hidden_dim: int, size: int = 32, num_hops: int = 10) -> None:
        super().__init__()
        self.w1 = nn.Parameter(torch.empty(size, hidden_dim))
        self.w2 = nn.Parameter(torch.empty(num_hops, size))
        nn.init.xavier_uniform_(self.w1)
        nn.init.xavier_uniform_(self.w2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        attention_score = torch.tanh(torch.matmul(self.w1.unsqueeze(0), inputs.transpose(1, 2)))
        attention_weights = torch.softmax(torch.matmul(self.w2.unsqueeze(0), attention_score), dim=-1)
        embedding_matrix = torch.matmul(attention_weights, inputs)
        return torch.flatten(embedding_matrix, start_dim=1)


class OfficialDeepConvLSTMAttention(nn.Module):
    """PyTorch port of isukrit/encodingHumanActivity proposed model."""

    def __init__(
        self,
        *,
        in_channels: int = 28,
        n_classes: int = 5,
        conv_filters: int = 3,
        lstm_units: int = 32,
        attention_size: int = 32,
        attention_hops: int = 10,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, conv_filters, kernel_size=(1, in_channels), stride=(1, 1))
        self.lstm = nn.LSTM(conv_filters, lstm_units, batch_first=True)
        self.attention = OfficialMultiHopSelfAttention(
            hidden_dim=lstm_units,
            size=attention_size,
            num_hops=attention_hops,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(attention_hops * lstm_units, n_classes - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2).unsqueeze(1)
        x = self.conv(x).squeeze(-1).transpose(1, 2)
        x, _hidden = self.lstm(x)
        x = self.dropout(self.attention(x))
        return self.head(x)


class TinySelfAttentionInteraction(nn.Module):
    def __init__(self, _sensor_channel: int, n_channels: int) -> None:
        super().__init__()
        self.query = nn.Linear(n_channels, n_channels, bias=False)
        self.key = nn.Linear(n_channels, n_channels, bias=False)
        self.value = nn.Linear(n_channels, n_channels, bias=False)
        self.gamma = nn.Parameter(torch.tensor([0.0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f, g, h = self.query(x), self.key(x), self.value(x)
        beta = F.softmax(torch.bmm(f, g.permute(0, 2, 1).contiguous()), dim=1)
        o = self.gamma * torch.bmm(h.permute(0, 2, 1).contiguous(), beta) + x.permute(0, 2, 1).contiguous()
        return o.permute(0, 2, 1).contiguous()


class TinyFilterWeightedAggregation(nn.Module):
    def __init__(self, _sensor_channel: int, n_channels: int) -> None:
        super().__init__()
        self.value_projection = nn.Linear(n_channels, n_channels)
        self.weight_projection = nn.Linear(n_channels, n_channels)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.softmax(torch.tanh(self.weight_projection(x)))
        values = F.relu(self.value_projection(x))
        return torch.sum(values * weights, dim=1)


class TinyTemporalGRU(nn.Module):
    def __init__(self, _sensor_channel: int, filter_num: int) -> None:
        super().__init__()
        self.rnn = nn.GRU(filter_num, filter_num, 1, bidirectional=False, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs, _hidden = self.rnn(x)
        return outputs


class TinyTemporalLSTM(nn.Module):
    def __init__(self, _sensor_channel: int, filter_num: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(filter_num, filter_num, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs, _hidden = self.lstm(x)
        return outputs


class TinyTemporalWeightedAggregation(nn.Module):
    def __init__(self, _sensor_channel: int, hidden_dim: int) -> None:
        super().__init__()
        self.fc_1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_2 = nn.Linear(hidden_dim, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.gamma = nn.Parameter(torch.tensor([0.0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.tanh(self.fc_1(x))
        out = self.fc_2(out).squeeze(2)
        weights_att = self.softmax(out).unsqueeze(2)
        context = torch.sum(weights_att * x, 1)
        return x[:, -1, :] + self.gamma * context


class TinyFC(nn.Module):
    def __init__(self, channel_in: int, channel_out: int) -> None:
        super().__init__()
        self.fc = nn.Linear(channel_in, channel_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class OfficialTinyHAR(nn.Module):
    """PyTorch port of teco-kit/ISWC22-HAR TinyHAR_Model default configuration."""

    def __init__(
        self,
        *,
        in_channels: int = 28,
        input_points: int = 2200,
        n_classes: int = 5,
        filter_num: int = 20,
        nb_conv_layers: int = 4,
        filter_size: int = 5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        filter_num_list = [1]
        for _idx in range(nb_conv_layers - 1):
            filter_num_list.append(filter_num)
        filter_num_list.append(filter_num)

        self.layers_conv = nn.ModuleList()
        for idx in range(nb_conv_layers):
            stride = (2, 1) if idx % 2 == 1 else (1, 1)
            self.layers_conv.append(
                nn.Sequential(
                    nn.Conv2d(filter_num_list[idx], filter_num_list[idx + 1], (filter_size, 1), stride),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm2d(filter_num_list[idx + 1]),
                )
            )

        downsampling_length = self._downsampling_length(input_points)
        self.channel_interaction = TinySelfAttentionInteraction(in_channels, filter_num)
        self.channel_fusion = TinyFC(in_channels * filter_num, 2 * filter_num)
        self.temporal_interaction = TinyTemporalLSTM(in_channels, 2 * filter_num)
        self.dropout = nn.Dropout(dropout)
        self.temporal_fusion = TinyTemporalWeightedAggregation(in_channels, 2 * filter_num)
        self.prediction = nn.Linear(2 * filter_num, n_classes - 1)

    def _downsampling_length(self, input_points: int) -> int:
        with torch.no_grad():
            x = torch.rand(1, 1, input_points, self.in_channels)
            for layer in self.layers_conv:
                x = layer(x)
        return int(x.shape[2])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2).unsqueeze(1)
        # [B, 1, T, C], matching TinyHAR_Model's expected input_shape.
        for layer in self.layers_conv:
            x = layer(x)

        x = x.permute(0, 3, 2, 1)
        x = torch.cat(
            [self.channel_interaction(x[:, :, t, :]).unsqueeze(3) for t in range(x.shape[2])],
            dim=-1,
        )
        x = self.dropout(x)

        x = x.permute(0, 3, 1, 2)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = F.relu(self.channel_fusion(x))

        x = self.temporal_interaction(x)
        x = self.temporal_fusion(x)
        return self.prediction(x)


@dataclass(frozen=True)
class ArchitectureSpec:
    name: ArchitectureName
    input_channels: int = 28
    input_points: int = 2200
    n_classes: int = 5


def build_architecture_model(
    name: ArchitectureName,
    *,
    in_channels: int = 28,
    n_classes: int = 5,
    input_points: int = 2200,
    dropout: float | None = None,
) -> nn.Module:
    if name == "deepconvlstm":
        return OfficialDeepConvLSTM(
            in_channels=in_channels,
            n_classes=n_classes,
            dropout=0.5 if dropout is None else float(dropout),
        )
    if name == "deepconvlstm_attention":
        return OfficialDeepConvLSTMAttention(
            in_channels=in_channels,
            n_classes=n_classes,
            dropout=0.0 if dropout is None else float(dropout),
        )
    if name == "tinyhar":
        return OfficialTinyHAR(
            in_channels=in_channels,
            input_points=input_points,
            n_classes=n_classes,
            dropout=0.1 if dropout is None else float(dropout),
        )
    raise ValueError(f"Unsupported architecture model: {name}")
