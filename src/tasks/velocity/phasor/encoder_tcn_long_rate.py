"""Runtime form of the 100-frame TCN with its auxiliary frequency head."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .encoder_tcn_context import ContextResidualBlock


class LongRateTCNEncoder(nn.Module):
    """Return the phase portion of the trained phase-plus-rate checkpoint."""

    runtime_control_dt_seconds = 0.02

    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.Sequential(*[
            ContextResidualBlock(29 if i == 0 else 64, dilation)
            for i, dilation in enumerate((1, 2, 4, 8, 16, 32))
        ])
        self.head = nn.Linear(64, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != 100 or x.shape[2] != 29:
            raise ValueError(f"expected [batch,100,29], got {tuple(x.shape)}")
        raw = self.head(self.blocks(x.transpose(1, 2))[:, :, -1]).float()
        return F.normalize(raw[:, :2], dim=-1, eps=1e-8)
