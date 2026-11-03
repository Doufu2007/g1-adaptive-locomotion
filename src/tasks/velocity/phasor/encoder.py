"""Supervised circular phase encoder used by the G1 phase study.

The deployed input is always a ``10 x 29`` joint-position history.  The
encoder head uses the explicit ``(sin(phi), cos(phi))`` convention throughout
the repository.  ``PhaseDecoder`` remains as a small compatibility module for
old checkpoints, but reconstruction is not part of the phase objective.
"""

from collections.abc import Mapping, Sequence

import torch
import torch.nn as nn


def normalize_phase(raw: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize raw two-dimensional logits without producing NaNs.

    A zero vector stays zero when its norm is below ``eps``.  This is useful
    during initialization and keeps the result finite; trained outputs are
    expected to be non-zero and therefore unit norm.
    """
    if raw.shape[-1] != 2:
        raise ValueError("phase vectors must have final dimension 2")
    if eps <= 0.0:
        raise ValueError("eps must be positive")
    norm = torch.linalg.vector_norm(raw, dim=-1, keepdim=True)
    safe_norm = norm.clamp_min(eps)
    normalized = raw / safe_norm
    return torch.where(norm < eps, torch.zeros_like(normalized), normalized)


def target_vectors(phase: torch.Tensor) -> torch.Tensor:
    """Convert radians to the canonical ``(sin(phi), cos(phi))`` target."""
    values = torch.as_tensor(phase)
    return torch.stack((torch.sin(values), torch.cos(values)), dim=-1)


def circular_phase_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Circular directional loss ``mean(1 - z dot z*)``."""
    if pred.shape != target.shape or pred.shape[-1] != 2:
        raise ValueError("pred and target must have matching shape [..., 2]")
    if not torch.isfinite(pred).all() or not torch.isfinite(target).all():
        raise ValueError("pred and target must be finite")
    return (1.0 - (pred * target).sum(dim=-1)).mean()


def select_validation_candidate(
    validation_losses: Mapping[str, float] | Sequence[tuple[str, float]],
) -> str:
    """Select the lowest validation loss with deterministic tie handling.

    Only the supplied validation values are consulted.  The returned name is
    intended to be frozen before the test split is evaluated.
    """
    items = list(validation_losses.items()) if isinstance(validation_losses, Mapping) else list(validation_losses)
    if not items:
        raise ValueError("at least one validation candidate is required")
    for name, loss in items:
        if not isinstance(name, str) or not torch.isfinite(torch.tensor(float(loss))):
            raise ValueError("validation candidates must have finite numeric losses")
    return min(items, key=lambda item: (float(item[1]), item[0]))[0]


class PhaseEncoder(nn.Module):
    """1D-CNN over a window of joint positions → normalized 2D phase (sin, cos)."""

    def __init__(
        self,
        joint_dim: int = 29,
        window: int = 10,
        hidden: int = 128,
        *,
        preserve_time: bool = False,
    ):
        super().__init__()
        if joint_dim <= 0 or window <= 0 or hidden <= 0:
            raise ValueError("joint_dim, window, and hidden must be positive")
        self.joint_dim = joint_dim
        self.window = window
        self.hidden = hidden
        self.preserve_time = preserve_time
        self.conv = nn.Sequential(
            nn.Conv1d(joint_dim, hidden, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.ReLU(),
        )
        if not preserve_time:
            self.pool = nn.AdaptiveAvgPool1d(1)
            feature_dim = hidden
        else:
            self.pool = nn.Identity()
            feature_dim = hidden * window
        self.fc = nn.Linear(feature_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, window, joint_dim)
        if x.ndim != 3 or x.shape[1] != self.window or x.shape[2] != self.joint_dim:
            raise ValueError(
                f"expected input [batch, {self.window}, {self.joint_dim}], got {tuple(x.shape)}"
            )
        x = x.permute(0, 2, 1)                # → (B, joint_dim, window)
        x = self.pool(self.conv(x))
        x = x.flatten(start_dim=1)
        return normalize_phase(self.fc(x))


class PhaseDecoder(nn.Module):
    """2D phase → reconstructed joint trajectory window via MLP + reshape."""

    def __init__(self, joint_dim: int = 29, window: int = 10, hidden: int = 128):
        super().__init__()
        self.window = window
        self.joint_dim = joint_dim
        self.mlp = nn.Sequential(
            nn.Linear(2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, window * joint_dim),
        )

    def forward(self, phase: torch.Tensor) -> torch.Tensor:
        # phase: (B, 2)
        x = self.mlp(phase)                     # → (B, window * joint_dim)
        x = x.view(-1, self.window, self.joint_dim)  # → (B, window, joint_dim)
        return x


class PhaseAutoencoder(nn.Module):
    """Compatibility wrapper exposing the supervised encoder and decoder."""

    def __init__(
        self,
        joint_dim: int = 29,
        window: int = 10,
        *,
        hidden: int = 128,
        preserve_time: bool = False,
    ):
        super().__init__()
        self.encoder = PhaseEncoder(
            joint_dim, window, hidden, preserve_time=preserve_time
        )
        self.decoder = PhaseDecoder(joint_dim, window, hidden)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        phase = self.encoder(x)
        recon = self.decoder(phase)
        return phase, recon
