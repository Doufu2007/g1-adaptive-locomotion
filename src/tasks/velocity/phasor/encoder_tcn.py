"""Causal TCN phase encoder, the best arm of the 2026-10-21 architecture sweep.

Copied unchanged from ``scripts/run_g1_contact_followup_comparison.py`` so that
a checkpoint trained there loads and evaluates bit-identically here.  Parameter
names must not be changed: the stored weights are a bare ``state_dict`` with
keys ``blocks.{0,1}.conv.{weight,bias}``, ``blocks.{0,1}.skip.{weight,bias}``
and ``head.{weight,bias}``.

Input and output conventions match the rest of the repository: the input is a
normalized ``[batch, window, joint_dim]`` joint-position history with
``window == 10``, and the output is a unit ``(sin(phi), cos(phi))`` vector.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

JOINT_DIM = 29
HIDDEN = 64


def unit_phase(raw: torch.Tensor) -> torch.Tensor:
    norm = torch.linalg.vector_norm(raw, dim=-1, keepdim=True).clamp_min(1e-8)
    return raw / norm


class FastCausalConv1d(nn.Module):
    """Conv1d-equivalent causal kernel evaluated as lagged channel projections.

    This preserves the exact kernel, dilation, parameter layout, and default
    Conv1d initialization while avoiding a very slow deterministic cuDNN path
    observed for short dilated sequences on the available Windows GPU.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels))
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5.0))
        fan_in = in_channels * kernel_size
        bound = 1.0 / np.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, channels, time]. Each kernel tap maps the matching past
        # position to the output position; this is cross-correlation exactly
        # as in Conv1d with left padding of dilation * (kernel_size - 1).
        time_major = x.transpose(1, 2).contiguous()
        batch, steps, _channels = time_major.shape
        output = self.bias.view(1, 1, -1)
        for tap in range(self.kernel_size):
            lag = (self.kernel_size - 1 - tap) * self.dilation
            if lag == 0:
                source = time_major
            else:
                source = time_major[:, : steps - lag]
            projected = F.linear(source, self.weight[:, :, tap])
            if lag:
                projected = F.pad(projected, (0, 0, lag, 0))
            output = output + projected
        return output.transpose(1, 2)


class CausalResidualBlock(nn.Module):
    def __init__(self, in_channels: int, dilation: int, hidden: int = HIDDEN) -> None:
        super().__init__()
        self.conv = FastCausalConv1d(in_channels, hidden, kernel_size=3, dilation=dilation)
        self.skip = nn.Identity() if in_channels == hidden else nn.Linear(in_channels, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        main = self.conv(x)
        skip = x if isinstance(self.skip, nn.Identity) else self.skip(x.transpose(1, 2)).transpose(1, 2)
        return F.relu(main + skip)


class TCNEncoder(nn.Module):
    """Two causal residual blocks, last-position readout, unit phase vector."""

    def __init__(self, joint_dim: int = JOINT_DIM, hidden: int = HIDDEN) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            CausalResidualBlock(joint_dim, 1, hidden),
            CausalResidualBlock(hidden, 2, hidden),
        )
        self.head = nn.Linear(hidden, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.blocks(x.transpose(1, 2))[:, :, -1]
        return unit_phase(self.head(h))
