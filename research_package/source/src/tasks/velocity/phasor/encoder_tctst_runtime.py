"""Runtime inference form of the repaired 100-frame TCTST phase encoder."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class RepairedTCTSTEncoder(nn.Module):
    """State-dict-compatible phase-only view of the repaired training model."""

    runtime_control_dt_seconds = 0.02

    def __init__(self, window: int = 100, dim: int = 384,
                 heads: int = 8, layers: int = 8, channels: int = 29) -> None:
        super().__init__()
        if window != 100 or channels != 29:
            raise ValueError("repaired TCTST runtime requires window=100 and 29 joints")
        self.window, self.dim, self.channels = window, dim, channels
        padded_window = ((window + 17) // 18) * 18
        self.conv = nn.Sequential(
            nn.ReplicationPad1d((padded_window - window, 0)),
            nn.Conv1d(1, 4, 3, stride=3), nn.ReLU(),
            nn.Conv1d(4, 4, 3, stride=3), nn.ReLU(),
            nn.MaxPool1d(2), nn.Flatten(),
        )
        self.embedding = nn.Linear(4 * (padded_window // 18), dim)
        self.channel_position = nn.Parameter(torch.randn(1, channels, dim) / math.sqrt(dim))
        self.mask_token = nn.Parameter(torch.zeros(dim))
        layer = nn.TransformerEncoderLayer(
            dim, heads, 4 * dim, dropout=.1, activation='relu', batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, layers, enable_nested_tensor=False,
        )
        self.head = nn.Sequential(nn.Linear(dim, 128), nn.ReLU(), nn.Linear(128, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1:] != (self.window, self.channels):
            raise ValueError(
                f"expected [batch,{self.window},{self.channels}], got {tuple(x.shape)}"
            )
        batch = x.shape[0]
        joint = x.transpose(1, 2)
        tokens = self.embedding(self.conv(joint.reshape(batch * self.channels, 1, self.window)))
        tokens = tokens.reshape(batch, self.channels, self.dim)
        hidden = tokens + self.channel_position
        layer_contexts = []
        for layer in self.transformer.layers:
            hidden = layer(hidden)
            layer_contexts.append(hidden.mean(dim=1))
        contextual = torch.stack(layer_contexts, dim=0).mean(dim=0)
        pooled = contextual + tokens.mean(dim=1) * math.sqrt(self.dim)
        raw = self.head(pooled).float()
        return F.normalize(raw[:, :2], dim=-1, eps=1e-8)
