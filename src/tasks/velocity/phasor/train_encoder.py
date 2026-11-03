"""Train and select the frozen supervised G1 phase encoder.

The command line entry point is intentionally fail-closed: it refuses to
consume the repository's dance clip until a verified walking manifest and
contact audit are available.  The helpers are usable in an offline study once
that manifest is marked ``READY``.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.tasks.velocity.phasor.data import validate_provenance
from src.tasks.velocity.phasor.encoder import (
    PhaseEncoder,
    circular_phase_loss,
    select_validation_candidate,
    target_vectors,
)


@dataclass(frozen=True)
class SelectionResult:
    """Frozen candidate and validation-only selection record."""

    selected_architecture: str
    validation_losses: dict[str, float]
    test_loss: float | None
    checkpoint: dict[str, Any]


def contiguous_window_pairs(
    num_windows: int, device: torch.device | None = None
) -> torch.Tensor:
    """Return adjacent indices from one already ordered sequence."""
    if num_windows < 2:
        return torch.empty((0, 2), dtype=torch.long, device=device)
    start = torch.arange(num_windows - 1, dtype=torch.long, device=device)
    return torch.stack((start, start + 1), dim=1)


def resample_joint_positions(
    joint_pos: torch.Tensor, source_fps: float, target_fps: float
) -> torch.Tensor:
    """Linearly resample a time-major joint sequence."""
    if joint_pos.ndim != 2:
        raise ValueError("joint_pos must have shape (frames, joints)")
    if source_fps <= 0 or target_fps <= 0:
        raise ValueError("frame rates must be positive")
    if len(joint_pos) < 2 or source_fps == target_fps:
        return joint_pos.clone()

    duration = (len(joint_pos) - 1) / source_fps
    output_count = int(torch.floor(torch.tensor(duration * target_fps)).item()) + 1
    output_times = torch.arange(
        output_count, device=joint_pos.device, dtype=joint_pos.dtype
    ) / target_fps
    source_positions = output_times * source_fps
    lower = source_positions.floor().long().clamp(max=len(joint_pos) - 2)
    upper = lower + 1
    fraction = (source_positions - lower).unsqueeze(1)
    return joint_pos[lower] * (1 - fraction) + joint_pos[upper] * fraction


def sliding_windows(sequence: np.ndarray, window: int) -> np.ndarray:
    """Return all overlapping windows, including the final valid window."""
    values = np.asarray(sequence)
    if values.ndim != 2:
        raise ValueError("sequence must have shape (frames, joints)")
    if window <= 0 or len(values) < window:
        raise ValueError("sequence is shorter than the requested window")
    return np.stack(
        [values[index : index + window] for index in range(len(values) - window + 1)]
    )


def _fit_candidate(
    train_windows: torch.Tensor,
    train_targets: torch.Tensor,
    validation_windows: torch.Tensor,
    validation_targets: torch.Tensor,
    *,
    preserve_time: bool,
    hidden: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
) -> tuple[PhaseEncoder, float]:
    if train_windows.ndim != 3 or train_windows.shape[-1] != 29:
        raise ValueError("train_windows must have shape [N, 10, 29]")
    if train_windows.shape[1] != 10:
        raise ValueError("the phase study fixes a 10-frame history")
    if len(train_windows) == 0 or len(validation_windows) == 0:
        raise ValueError("train and validation splits must be non-empty")

    model = PhaseEncoder(
        joint_dim=train_windows.shape[-1],
        window=train_windows.shape[1],
        hidden=hidden,
        preserve_time=preserve_time,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    target_train = target_vectors(train_targets.to(device))
    dataset = TensorDataset(train_windows.to(device), target_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for _ in range(max(0, epochs)):
        model.train()
        for batch, target in loader:
            loss = circular_phase_loss(model(batch), target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        validation_pred = model(validation_windows.to(device))
        validation_loss = circular_phase_loss(
            validation_pred, target_vectors(validation_targets.to(device))
        )
    return model, float(validation_loss.cpu())


def train_and_select(
    train_windows: torch.Tensor | np.ndarray,
    train_phase: torch.Tensor | np.ndarray,
    validation_windows: torch.Tensor | np.ndarray,
    validation_phase: torch.Tensor | np.ndarray,
    *,
    test_windows: torch.Tensor | np.ndarray | None = None,
    test_phase: torch.Tensor | np.ndarray | None = None,
    hidden: int = 128,
    epochs: int = 10,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    device: torch.device | None = None,
) -> SelectionResult:
    """Fit both architectures, select by validation, then freeze and test."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_x = torch.as_tensor(train_windows, dtype=torch.float32)
    val_x = torch.as_tensor(validation_windows, dtype=torch.float32)
    test_x = None if test_windows is None else torch.as_tensor(test_windows, dtype=torch.float32)
    train_y = torch.as_tensor(train_phase, dtype=torch.float32).reshape(-1)
    val_y = torch.as_tensor(validation_phase, dtype=torch.float32).reshape(-1)
    test_y = None if test_phase is None else torch.as_tensor(test_phase, dtype=torch.float32).reshape(-1)
    if len(train_x) != len(train_y) or len(val_x) != len(val_y):
        raise ValueError("window and phase counts must match")

    candidates: dict[str, PhaseEncoder] = {}
    validation_losses: dict[str, float] = {}
    for name, preserve_time in (("pooled", False), ("preserve_time", True)):
        candidate, loss = _fit_candidate(
            train_x,
            train_y,
            val_x,
            val_y,
            preserve_time=preserve_time,
            hidden=hidden,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=device,
        )
        candidates[name] = candidate
        validation_losses[name] = loss

    selected_name = select_validation_candidate(validation_losses)
    selected = candidates[selected_name]
    for parameter in selected.parameters():
        parameter.requires_grad_(False)
    test_loss = None
    if test_x is not None or test_y is not None:
        if test_x is None or test_y is None or len(test_x) != len(test_y):
            raise ValueError("test windows and phases must be supplied together")
        with torch.no_grad():
            test_loss = float(
                circular_phase_loss(
                    selected(test_x.to(device)), target_vectors(test_y.to(device))
                ).cpu()
            )

    checkpoint = {
        "state_dict": selected.state_dict(),
        "joint_dim": int(train_x.shape[-1]),
        "window": int(train_x.shape[1]),
        "preserve_time": selected_name == "preserve_time",
        "validation_losses": validation_losses,
        "selected_architecture": selected_name,
    }
    return SelectionResult(selected_name, validation_losses, test_loss, checkpoint)


def main() -> int:
    manifest_path = _PROJECT_ROOT / "docs" / "data" / "g1_phase_data_provenance.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    try:
        validate_provenance(manifest)
    except Exception as error:
        print(f"DATA_NOT_READY: {error}")
        return 2
    raise RuntimeError(
        "Verified walking data is available; construct sequence-level splits and contact labels before training."
    )


if __name__ == "__main__":
    raise SystemExit(main())
