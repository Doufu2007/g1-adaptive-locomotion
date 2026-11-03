"""Dual-stream actor model for the G1 velocity study."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from math import prod
from typing import Any

import torch
from torch import nn
from tensordict import TensorDict

from rsl_rl.models import MLPModel


StreamSlice = tuple[int, int]
StreamSlices = tuple[StreamSlice, ...]


def _activation(name: str) -> nn.Module:
  activations: dict[str, type[nn.Module]] = {
    "elu": nn.ELU,
    "gelu": nn.GELU,
    "relu": nn.ReLU,
    "selu": nn.SELU,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
  }
  try:
    return activations[name.lower()]()
  except KeyError as exc:
    supported = ", ".join(sorted(activations))
    raise ValueError(
      f"Unsupported activation '{name}'. Expected one of: {supported}"
    ) from exc


def _as_slices(values: Iterable[Sequence[int]], *, stream_name: str) -> StreamSlices:
  slices: list[StreamSlice] = []
  for raw_slice in values:
    if len(raw_slice) != 2:
      raise ValueError(f"{stream_name} slices must be (start, end) pairs")
    start, end = int(raw_slice[0]), int(raw_slice[1])
    if start < 0 or end <= start:
      raise ValueError(f"Invalid {stream_name} slice ({start}, {end})")
    slices.append((start, end))
  if not slices:
    raise ValueError(f"{stream_name} must contain at least one slice")
  return tuple(slices)


def validate_stream_slices(
  *,
  input_dim: int,
  proprio_slices: StreamSlices,
  extero_slices: StreamSlices,
) -> None:
  """Require the two streams to form an exact, non-overlapping partition."""
  owners: list[str | None] = [None] * input_dim
  for stream_name, stream_slices in (
    ("proprio", proprio_slices),
    ("extero", extero_slices),
  ):
    for start, end in stream_slices:
      if end > input_dim:
        raise ValueError(
          f"{stream_name} slice ({start}, {end}) exceeds input dimension {input_dim}"
        )
      for index in range(start, end):
        if owners[index] is not None:
          raise ValueError(
            f"Observation index {index} is assigned to both "
            f"{owners[index]} and {stream_name}"
          )
        owners[index] = stream_name

  missing = [index for index, owner in enumerate(owners) if owner is None]
  if missing:
    preview = ", ".join(map(str, missing[:8]))
    raise ValueError(f"Stream slices do not cover observation indices: {preview}")


def _build_encoder(
  input_dim: int,
  hidden_dims: Sequence[int],
  activation: str,
) -> nn.Sequential:
  if not hidden_dims or any(dim <= 0 for dim in hidden_dims):
    raise ValueError("Encoder hidden dimensions must be positive and non-empty")
  layers: list[nn.Module] = []
  current_dim = input_dim
  for hidden_dim in hidden_dims:
    layers.extend((nn.Linear(current_dim, hidden_dim), _activation(activation)))
    current_dim = hidden_dim
  layers.append(nn.LayerNorm(current_dim))
  return nn.Sequential(*layers)


def _build_fusion(
  input_dim: int,
  output_dim: int,
  hidden_dims: Sequence[int],
  activation: str,
) -> nn.Sequential:
  if not hidden_dims or any(dim <= 0 for dim in hidden_dims):
    raise ValueError("Fusion hidden dimensions must be positive and non-empty")
  layers: list[nn.Module] = []
  current_dim = input_dim
  for hidden_dim in hidden_dims:
    layers.extend((nn.Linear(current_dim, hidden_dim), _activation(activation)))
    current_dim = hidden_dim
  layers.append(nn.Linear(current_dim, output_dim))
  return nn.Sequential(*layers)


class DualStreamMLP(nn.Module):
  """Encode proprioception and exteroception separately before fusion."""

  def __init__(
    self,
    input_dim: int,
    output_dim: int,
    *,
    proprio_slices: Iterable[Sequence[int]],
    extero_slices: Iterable[Sequence[int]],
    proprio_hidden_dims: Sequence[int] = (256, 128),
    extero_hidden_dims: Sequence[int] = (128, 64),
    fusion_hidden_dims: Sequence[int] = (256, 128),
    activation: str = "elu",
  ) -> None:
    super().__init__()
    self.input_dim = input_dim
    self.proprio_slices = _as_slices(proprio_slices, stream_name="proprio")
    self.extero_slices = _as_slices(extero_slices, stream_name="extero")
    validate_stream_slices(
      input_dim=input_dim,
      proprio_slices=self.proprio_slices,
      extero_slices=self.extero_slices,
    )

    self.proprio_dim = sum(end - start for start, end in self.proprio_slices)
    self.extero_dim = sum(end - start for start, end in self.extero_slices)
    self.proprio_encoder = _build_encoder(
      self.proprio_dim, proprio_hidden_dims, activation
    )
    self.extero_encoder = _build_encoder(
      self.extero_dim, extero_hidden_dims, activation
    )
    self.fusion_trunk = _build_fusion(
      proprio_hidden_dims[-1] + extero_hidden_dims[-1],
      output_dim,
      fusion_hidden_dims,
      activation,
    )

  @staticmethod
  def _collect(x: torch.Tensor, stream_slices: StreamSlices) -> torch.Tensor:
    return torch.cat([x[..., start:end] for start, end in stream_slices], dim=-1)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    if x.shape[-1] != self.input_dim:
      raise ValueError(
        f"Dual-stream actor expected {self.input_dim} observations, got {x.shape[-1]}"
      )
    proprio = self._collect(x, self.proprio_slices)
    extero = self._collect(x, self.extero_slices)
    encoded = torch.cat(
      (self.proprio_encoder(proprio), self.extero_encoder(extero)), dim=-1
    )
    return self.fusion_trunk(encoded)


class DualStreamMLPModel(MLPModel):
  """RSL-RL model adapter that replaces only the Actor MLP."""

  def __init__(
    self,
    obs: TensorDict,
    obs_groups: dict[str, list[str]],
    obs_set: str,
    output_dim: int,
    hidden_dims: tuple[int, ...] | list[int] = (256, 128),
    activation: str = "elu",
    obs_normalization: bool = False,
    distribution_cfg: dict[str, Any] | None = None,
    *,
    stream_config: dict[str, Any],
  ) -> None:
    if obs_set != "actor":
      raise ValueError("DualStreamMLPModel is reserved for the Actor")
    distribution_copy = dict(distribution_cfg) if distribution_cfg is not None else None
    super().__init__(
      obs,
      obs_groups,
      obs_set,
      output_dim,
      hidden_dims=hidden_dims,
      activation=activation,
      obs_normalization=obs_normalization,
      distribution_cfg=distribution_copy,
    )

    mlp_output_dim = (
      self.distribution.input_dim if self.distribution is not None else output_dim
    )
    if not isinstance(mlp_output_dim, int):
      mlp_output_dim = prod(mlp_output_dim)

    self.mlp = DualStreamMLP(
      self.obs_dim,
      mlp_output_dim,
      proprio_slices=stream_config["proprio_slices"],
      extero_slices=stream_config["extero_slices"],
      proprio_hidden_dims=tuple(stream_config.get("proprio_hidden_dims", (256, 128))),
      extero_hidden_dims=tuple(stream_config.get("extero_hidden_dims", (128, 64))),
      fusion_hidden_dims=tuple(stream_config.get("fusion_hidden_dims", hidden_dims)),
      activation=activation,
    )
    if self.distribution is not None:
      self.distribution.init_mlp_weights(self.mlp)


PROPRIO_TERM_NAMES = frozenset(
  {"base_ang_vel", "projected_gravity", "joint_pos", "joint_vel", "actions"}
)
EXTERO_TERM_NAMES = frozenset({"command", "phase", "height_scan"})


def compute_stream_config_from_env(
  observation_manager: Any,
  *,
  group_name: str = "actor",
  proprio_hidden_dims: Sequence[int] = (256, 128),
  extero_hidden_dims: Sequence[int] = (128, 64),
  fusion_hidden_dims: Sequence[int] = (256, 128),
) -> dict[str, Any]:
  """Derive non-contiguous stream slices from ObservationManager metadata."""
  term_names = tuple(observation_manager.active_terms[group_name])
  term_shapes = tuple(observation_manager.group_obs_term_dim[group_name])
  if len(term_names) != len(term_shapes):
    raise ValueError("Observation term names and dimensions have different lengths")

  proprio_slices: list[StreamSlice] = []
  extero_slices: list[StreamSlice] = []
  unknown_terms: list[str] = []
  offset = 0
  term_dims: dict[str, int] = {}
  for term_name, term_shape in zip(term_names, term_shapes, strict=True):
    term_dim = prod(term_shape)
    term_dims[term_name] = term_dim
    term_slice = (offset, offset + term_dim)
    if term_name in PROPRIO_TERM_NAMES:
      proprio_slices.append(term_slice)
    elif term_name in EXTERO_TERM_NAMES:
      extero_slices.append(term_slice)
    else:
      unknown_terms.append(term_name)
    offset += term_dim

  if unknown_terms:
    raise ValueError(
      "Unclassified Actor observation terms: " + ", ".join(unknown_terms)
    )

  proprio_dim = sum(end - start for start, end in proprio_slices)
  extero_dim = sum(end - start for start, end in extero_slices)
  validate_stream_slices(
    input_dim=offset,
    proprio_slices=tuple(proprio_slices),
    extero_slices=tuple(extero_slices),
  )
  return {
    "proprio_slices": tuple(proprio_slices),
    "extero_slices": tuple(extero_slices),
    "proprio_hidden_dims": tuple(proprio_hidden_dims),
    "extero_hidden_dims": tuple(extero_hidden_dims),
    "fusion_hidden_dims": tuple(fusion_hidden_dims),
    "term_dims": term_dims,
    "input_dim": offset,
    "proprio_dim": proprio_dim,
    "extero_dim": extero_dim,
  }
