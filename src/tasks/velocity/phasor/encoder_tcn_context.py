"""Runtime equivalent of the frozen five-block long-history TCN experiment."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .encoder_tcn import FastCausalConv1d, unit_phase


class ContextCausalConv1d(FastCausalConv1d):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.transpose(1, 2).contiguous()
        batch, steps, _ = h.shape
        result = self.bias.view(1, 1, -1).expand(batch, steps, -1)
        for tap in range(self.kernel_size):
            lag = (self.kernel_size - 1 - tap) * self.dilation
            if lag >= steps:
                continue
            projected = F.linear(h if lag == 0 else h[:, :-lag], self.weight[:, :, tap])
            if lag:
                projected = F.pad(projected, (0, 0, lag, 0))
            result = result + projected
        return result.transpose(1, 2)


class ContextResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.conv = ContextCausalConv1d(channels, 64, 3, dilation)
        self.skip = nn.Identity() if channels == 64 else nn.Linear(channels, 64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = x if isinstance(self.skip, nn.Identity) else self.skip(x.transpose(1, 2)).transpose(1, 2)
        return F.relu(self.conv(x) + skip)


class LongContextTCNEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.Sequential(*[ContextResidualBlock(29 if i == 0 else 64, d)
                                      for i, d in enumerate((1, 2, 4, 8, 16))])
        self.head = nn.Linear(64, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return unit_phase(self.head(self.blocks(x.transpose(1, 2))[:, :, -1]))
