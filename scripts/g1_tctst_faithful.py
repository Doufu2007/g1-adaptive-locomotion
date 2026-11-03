"""Faithful-adaptation TCTST with 29-channel masked pretraining.

The frozen adaptation lives in ``scripts/g1_external_models.py``.  That file is
hash-pinned in ``results/g1_external_methods_trials_20261022/run_v2/protocol.json``
and must not change, so the corrections required by
``results/g1_tctst_faithful_20261023/preregistration_20261023.md`` are made
here instead.

Two defects of the frozen masked arm are corrected:

1. Masked channels used to receive the all-zero window, which the *shared*
   convolution maps to one and the same token for every channel; the only
   channel signal was ``channel_position`` initialised at 0.02, which the
   first LayerNorm shrinks further.  Here the channel embedding is initialised
   at unit RMS and masked channels are replaced by a learned mask token.
2. The frozen pretraining tensor had 32 channels, the last three being phase
   sin/cos/valid, while inference supplies zeros there.  Here the estimator
   sees exactly the 29 joint channels in both phases.

The backbone hyper-parameters (window 100, dim 384, 8 heads, 8 layers) and the
down-sampling (3/3/2, left pad to 108) are inherited unchanged.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from scripts.g1_external_models import TCTSTPhase, phase_rate_output

JOINT_CHANNELS = 29
MASK_COUNT = 10


class ChannelIdentifiedTCTST(TCTSTPhase):
    """29-channel TCTST with unit-scale channel identity and a mask token."""

    def __init__(self, window: int = 100, dim: int = 384, heads: int = 8,
                 layers: int = 8, channels: int = JOINT_CHANNELS) -> None:
        super().__init__(window=window, dim=dim, heads=heads, layers=layers)
        self.channels = channels
        # Replace the 0.02-scale placeholder with a per-dimension embedding
        # whose per-channel RMS is 1, so channel identity survives LayerNorm.
        del self.channel_position
        self.channel_position = nn.Parameter(
            torch.randn(1, channels, dim) / math.sqrt(dim)
        )
        self.mask_token = nn.Parameter(torch.zeros(dim))

    def encode_tokens(self, joint: torch.Tensor,
                      mask: torch.Tensor | None = None) -> torch.Tensor:
        """joint ``[b, 29, window]`` -> channel tokens ``[b, 29, dim]``."""
        if joint.shape[1] != self.channels or joint.shape[2] != self.window:
            raise ValueError(
                f"expected [{self.channels},{self.window}] input, got "
                f"{tuple(joint.shape[1:])}"
            )
        b, c, t = joint.shape
        h = self.embedding(self.conv(joint.reshape(b * c, 1, t))).reshape(b, c, self.dim)
        if mask is not None:
            h = torch.where(mask[..., None], self.mask_token.to(h.dtype), h)
        return self.transformer(h + self.channel_position)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Unlike the frozen 32-channel version, nothing is appended here: the
        # estimator always sees exactly the 29 joint channels.
        return phase_rate_output(self.head(self.encode_tokens(x.transpose(1, 2)).mean(dim=1)))


class FixedRawState(ChannelIdentifiedTCTST):
    """Supervised head: raw sin/cos for the loss, unit vector at inference."""

    def raw_state(self, x: torch.Tensor) -> torch.Tensor:
        joint = x.transpose(1, 2)
        raw = self.head(self.encode_tokens(joint).mean(dim=1)).float()
        return torch.cat((raw[:, :2], F.softplus(raw[:, 2:3])), dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.raw_state(x)
        return torch.cat((F.normalize(raw[:, :2], dim=-1, eps=1e-8), raw[:, 2:3]), dim=-1)


class MaskedPretrainer(nn.Module):
    """Decoder over channel tokens; reconstructs the whole window of masked channels."""

    def __init__(self, model: ChannelIdentifiedTCTST) -> None:
        super().__init__()
        self.model = model
        layer = nn.TransformerEncoderLayer(model.dim, 8, 4 * model.dim, dropout=.1,
                                           activation='relu', batch_first=True)
        self.decoder = nn.TransformerEncoder(layer, 8, enable_nested_tensor=False)
        self.reconstruct = nn.Linear(model.dim, model.window)

    def forward(self, joint: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.reconstruct(self.decoder(self.model.encode_tokens(joint, mask)))


def masked_reconstruction_loss(prediction: torch.Tensor, target: torch.Tensor,
                               mask: torch.Tensor) -> torch.Tensor:
    """MSE over masked channels and all frames; every joint entry is valid."""
    selected = mask[..., None]
    return ((prediction - target).square() * selected).sum() / (
        mask.sum() * target.shape[-1]
    ).clamp_min(1)


def state_loss(raw: torch.Tensor, target: torch.Tensor, rate: torch.Tensor,
               valid: torch.Tensor) -> torch.Tensor:
    """Identical to the v2 amendment: raw half-MSE plus 0.02 rate MSE."""
    return .5 * (raw[:, :2] - target).square().sum(dim=-1).mean() + .02 * (
        (raw[:, 2] - rate).square() * valid
    ).sum() / valid.sum().clamp_min(1)
