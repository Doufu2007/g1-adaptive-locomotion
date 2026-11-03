from __future__ import annotations

import torch
from torch import nn


class PhaseHistoryState:
    """Per-environment history with one encoder update per control step."""

    def __init__(
        self,
        num_envs: int,
        window: int,
        joint_dim: int,
        device: torch.device,
    ) -> None:
        self.buffer = torch.zeros(num_envs, window, joint_dim, device=device)
        self.phase = torch.zeros(num_envs, 2, device=device)
        self.last_control_step = torch.full(
            (num_envs,), -1, dtype=torch.long, device=device
        )

    def reset(
        self,
        env_ids: torch.Tensor | slice | None = None,
        *,
        fill_value: torch.Tensor | None = None,
    ) -> None:
        """Reset selected environments and optionally seed history with pose.

        ``fill_value`` is already normalized and has shape ``[envs, joints]``
        (or ``[joints]`` for a broadcastable value).  Seeding every history
        slot with the reset pose prevents an episode from inheriting frames
        from the previous episode while avoiding an artificial zero-motion
        prefix.
        """
        if env_ids is None:
            self.buffer.zero_()
            self.phase.zero_()
            self.last_control_step.fill_(-1)
            if fill_value is not None:
                values = torch.as_tensor(fill_value, device=self.buffer.device, dtype=self.buffer.dtype)
                if values.ndim == 1:
                    values = values.unsqueeze(0).expand(self.buffer.shape[0], -1)
                if values.shape != (self.buffer.shape[0], self.buffer.shape[2]):
                    raise ValueError("fill_value shape does not match phase history")
                self.buffer.copy_(values.unsqueeze(1).expand_as(self.buffer))
            return

        self.buffer[env_ids] = 0
        self.phase[env_ids] = 0
        self.last_control_step[env_ids] = -1
        if fill_value is not None:
            values = torch.as_tensor(fill_value, device=self.buffer.device, dtype=self.buffer.dtype)
            if values.ndim == 1:
                values = values.unsqueeze(0)
            selected_count = self.buffer[env_ids].shape[0]
            if values.shape == (self.buffer.shape[0], self.buffer.shape[2]):
                values = values[env_ids]
            if values.shape != (selected_count, self.buffer.shape[2]):
                raise ValueError("fill_value shape does not match selected environments")
            self.buffer[env_ids] = values.unsqueeze(1).expand(selected_count, self.buffer.shape[1], self.buffer.shape[2])

    @torch.no_grad()
    def update(
        self,
        joint_pos: torch.Tensor,
        control_step: int,
        encoder: nn.Module,
        mean: torch.Tensor,
        std: torch.Tensor,
    ) -> torch.Tensor:
        if joint_pos.shape[0] != self.buffer.shape[0]:
            raise ValueError("joint_pos batch size does not match phase history")
        if joint_pos.shape[1] != self.buffer.shape[2]:
            raise ValueError("joint_pos dimension does not match phase history")

        changed = self.last_control_step != control_step
        env_ids = changed.nonzero(as_tuple=False).squeeze(-1)
        if env_ids.numel() == 0:
            return self.phase

        normalized = (joint_pos[env_ids] - mean) / (std + 1e-8)
        history = torch.roll(self.buffer[env_ids], shifts=-1, dims=1)
        history[:, -1, :] = normalized
        self.buffer[env_ids] = history
        self.phase[env_ids] = encoder(history)
        self.last_control_step[env_ids] = control_step
        return self.phase


def phase_to_angle(phase: torch.Tensor) -> torch.Tensor:
    """Decode canonical ``(sin(phi), cos(phi))`` vectors to radians."""
    if phase.shape[-1] != 2:
        raise ValueError("phase must have final dimension 2")
    return torch.atan2(phase[..., 0], phase[..., 1])
