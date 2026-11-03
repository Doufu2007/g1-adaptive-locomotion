"""Causal features, matched sequence models and cycle supervision (2026-10-25)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

UPDATES, BATCH_CYCLES, CONTEXT, DT = 2000, 8, 127, .02
SEEDS = (20261025, 20261026, 20261027)
ROWS = {
    "direct_motion": ("direct", 29, False),
    "observer_motion": ("observer", 29, False),
    "direct_mixed": ("direct", 29, True),
    "observer_mixed": ("observer", 29, True),
    "observer_sensors_mixed": ("observer", 64, True),
}


def matrix():
    return [dict(row=row, seed=seed) for seed in SEEDS for row in ROWS]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def causal_features(q, quaternion_wxyz, dt=DT):
    """Positions, backward joint velocity, body gravity and backward body gyro."""
    from scipy.spatial.transform import Rotation
    q = np.asarray(q, dtype=np.float64)
    quat = np.asarray(quaternion_wxyz, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 29 or quat.shape != (len(q), 4) or not len(q):
        raise ValueError("expected nonempty aligned [T,29] positions and [T,4] WXYZ")
    if not np.isfinite(q).all() or not np.isfinite(quat).all():
        raise ValueError("nonfinite motion features")
    rotation = Rotation.from_quat(quat[:, [1, 2, 3, 0]])
    velocity = np.zeros_like(q); velocity[1:] = np.diff(q, axis=0) / dt
    gravity = rotation.inv().apply(np.tile([0., 0., -1.], (len(q), 1)))
    angular = np.zeros((len(q), 3))
    angular[1:] = (rotation[:-1].inv() * rotation[1:]).as_rotvec() / dt
    return np.concatenate((q, velocity, gravity, angular), axis=1).astype(np.float32)


def debounce_contacts(values, frames=3):
    values = np.asarray(values, dtype=bool)
    if values.ndim != 2 or frames < 1 or not len(values):
        raise ValueError("nonempty [T,feet] contacts and positive confirmation length required")
    state = values[0].copy()
    count = np.zeros(values.shape[1], dtype=int)
    result = np.empty_like(values)
    for t, row in enumerate(values):
        count = np.where(row != state, count + 1, 0)
        switch = count >= frames
        state = np.where(switch, row, state)
        count[switch] = 0
        result[t] = state
    return result


def contact_labels(contacts, debounce=3):
    stable = debounce_contacts(contacts, debounce)
    if stable.shape[1] != 2:
        raise ValueError("contact teacher requires left/right feet")
    # Opening stance is already ongoing, never a fabricated touchdown.
    event = stable & ~np.concatenate((stable[:1], stable[:-1]), axis=0)
    left, right = event[:, 0], event[:, 1]
    phase = np.full(len(stable), np.nan, dtype=np.float32)
    cycles = []
    events = np.flatnonzero(left)
    for start, end in zip(events[:-1], events[1:]):
        phase[start:end] = np.arange(end - start) * (2 * np.pi / (end - start))
        cycles.append((int(start), int(end), int(right[start:end].sum())))
    return phase, left, right, cycles


def fit_normalizer(records, joint_mean, joint_std):
    if not records or any(r["split"] != "train" for r in records):
        raise ValueError("normalization may fit train records only")
    extra = np.concatenate([r["features"][:, 29:] for r in records]).astype(np.float64)
    mean = np.r_[joint_mean, extra.mean(axis=0)].astype(np.float32)
    std = np.r_[joint_std, np.maximum(extra.std(axis=0), 1e-4)].astype(np.float32)
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("invalid train normalizer")
    return mean, std


class CausalBlock(nn.Module):
    def __init__(self, channels, width, dilation, project=False):
        super().__init__()
        self.conv = nn.Conv1d(channels, width, 3, dilation=dilation)
        self.skip = nn.Conv1d(channels, width, 1) if project or channels != width else nn.Identity()
        self.padding = 2 * dilation

    def forward(self, x):
        return F.silu(self.conv(F.pad(x, (self.padding, 0))) + self.skip(x))


class PhaseNet(nn.Module):
    def __init__(self, channels, mode, width=64):
        super().__init__()
        if mode not in ("direct", "observer"):
            raise ValueError("unknown phase output mode")
        self.channels, self.mode, self.width = channels, mode, width
        self.blocks = nn.Sequential(*[CausalBlock(channels if i == 0 else width, width, d, project=i == 0)
                                      for i, d in enumerate((1, 2, 4, 8, 16, 32))])
        self.head = nn.Linear(width, 4)
        with torch.no_grad():
            self.head.bias[:] = torch.tensor([0., 1., -1.0986123, -1.0986123])

    def forward(self, x, state=None):
        if x.ndim != 3 or x.shape[2] != self.channels or not x.shape[1]:
            raise ValueError("expected nonempty [batch,time,channels]")
        full = x if state is None else torch.cat((state["history"], x), dim=1)
        hidden = self.blocks(full.transpose(1, 2)).transpose(1, 2)[:, -x.shape[1]:]
        raw = self.head(hidden)
        anchor = torch.atan2(raw[..., 0], raw[..., 1])
        rate, gain = 4 * torch.sigmoid(raw[..., 2]), .25 * torch.sigmoid(raw[..., 3])
        previous = None if state is None else state["phase"]
        if self.mode == "direct":
            if previous is None:
                delta = anchor[:, 1:] - anchor[:, :-1]
                unwrapped = torch.cat((anchor[:, :1], anchor[:, :1] + torch.cumsum(
                    torch.atan2(torch.sin(delta), torch.cos(delta)), dim=1)), dim=1)
            else:
                delta = anchor - torch.cat((previous[:, None], anchor[:, :-1]), dim=1)
                unwrapped = previous[:, None] + torch.cumsum(torch.atan2(torch.sin(delta), torch.cos(delta)), dim=1)
        else:
            values = []
            for t in range(x.shape[1]):
                if previous is None:
                    previous = anchor[:, t]
                else:
                    step = rate[:, t] * (2 * np.pi * DT)
                    residual = anchor[:, t] - previous - step
                    innovation = torch.atan2(torch.sin(residual), torch.cos(residual))
                    previous = previous + (step + gain[:, t] * innovation).clamp(0, np.pi / 3)
                values.append(previous)
            unwrapped = torch.stack(values, dim=1)
        phase = torch.stack((torch.sin(unwrapped), torch.cos(unwrapped)), dim=-1)
        return dict(phase=phase, unwrapped=unwrapped, anchor=anchor, rate=rate, gain=gain,
                    state=dict(history=full[:, -(CONTEXT - 1):].detach(), phase=unwrapped[:, -1].detach()))


def cycle_objective(output, target, starts, ends):
    times = torch.arange(target.shape[1], device=target.device)[None]
    mask = (times >= starts[:, None]) & (times < ends[:, None])
    point = (1 - torch.cos(output["unwrapped"] - target))[mask].mean()
    anchor = (1 - torch.cos(output["anchor"] - target))[mask].mean()
    valid_step = mask[:, 1:] & mask[:, :-1]
    actual = output["unwrapped"][:, 1:] - output["unwrapped"][:, :-1]
    expected = target[:, 1:] - target[:, :-1]
    increment = (actual - expected).square()[valid_step].mean()
    hz = 1 / ((ends - starts) * DT)
    rate = ((output["rate"] - hz[:, None]).square())[mask].mean()
    batch = torch.arange(len(target), device=target.device)
    net = output["unwrapped"][batch, ends - 1] - output["unwrapped"][batch, starts]
    reference = target[batch, ends - 1] - target[batch, starts]
    cycle = ((net - reference) / (2 * np.pi)).square().mean()
    parts = dict(anchor=anchor, point=point, increment=increment, rate=rate, cycle=cycle)
    return anchor + point + increment + .02 * rate + cycle, parts


class CycleSampler:
    def __init__(self, records, seed, mixed):
        self.rng = np.random.default_rng(seed)
        self.mixed = mixed
        self.pools = {d: [(r, c) for r in records if r["split"] == "train" and r["domain"] == d
                          for c in r["cycles"] if c[2] == 1 and c[1] - c[0] >= 3]
                      for d in ("motion", "simulation")}
        if not self.pools["motion"] or mixed and not self.pools["simulation"]:
            raise ValueError("insufficient consistent training cycles")

    def sample(self):
        domains = ["motion"] * (4 if self.mixed else BATCH_CYCLES)
        if self.mixed:
            domains += ["simulation"] * 4
        return [self.pools[d][int(self.rng.integers(len(self.pools[d])))] for d in domains]


def make_batch(sample, mean, std, channels, device):
    slices = [(r, c, max(0, c[0] - CONTEXT)) for r, c in sample]
    length = max(c[1] - context for r, c, context in slices)
    x = np.zeros((len(sample), length, channels), np.float32)
    y = np.zeros((len(sample), length), np.float32)
    starts, ends = [], []
    frames = 0
    for i, (record, cycle, context) in enumerate(slices):
        start, end, _ = cycle
        count = end - context
        x[i, :count] = ((record["features"][context:end] - mean) / std)[:, :channels]
        y[i, :count] = np.nan_to_num(record["target"][context:end])
        starts.append(start - context); ends.append(count)
        frames += end - start
    return (torch.as_tensor(x, device=device), torch.as_tensor(y, device=device),
            torch.tensor(starts, device=device), torch.tensor(ends, device=device), frames)


def previous_ready(directory):
    directory = Path(directory)
    try:
        if json.loads((directory / "queue_status.json").read_text(encoding="utf-8"))["status"] != "COMPLETED":
            return False
        if not (directory / "comparison.json").is_file():
            return False
        for terrain in ("flat", "rough"):
            for seed in (20261401, 20261402, 20261403):
                for arm in ("clock", "learned"):
                    record = json.loads((directory / "training" / f"{terrain}_{arm}_seed{seed}" / "manifest.json").read_text(encoding="utf-8"))
                    if (record["status"] != "COMPLETED" or record["completed_learning_iterations"] != 501
                            or record["completed_environment_transitions"] != 769536 or not record["encoder_frozen_verified"]):
                        return False
                for label in ("clock_clock", "clock_learned", "learned_learned"):
                    path = directory / "evaluation" / f"{terrain}_seed{seed}_{label}"
                    record = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
                    episodes = json.loads((path / "episodes.json").read_text(encoding="utf-8"))
                    keys = {(r["seed"], r["scenario"]) for r in episodes}
                    expected = {(s, c) for s in range(20261501, 20261511) for c in ("steady", "speed_change", "push")}
                    if record["status"] != "COMPLETED" or record["episode_count"] != 30 or len(episodes) != 30 or keys != expected:
                        return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False
