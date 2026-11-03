"""Evaluate the frozen supervised encoder on contact-free G1 rollouts.

Contacts are read only to create an offline reference phase for the gate.  The
encoder itself receives joint position history only, matching deployment.
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import torch

from scripts.evaluate_phase_study import evaluate_phase_predictions, evaluate_rollout_gate
from src.tasks.velocity.phasor.data import debounce_contact_sequence
from src.tasks.velocity.phasor.encoder import PhaseEncoder

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "results" / "g1_phase_dataset_20261019"
ROLLOUT = ROOT / "results" / "phasor_contact_supervised_v3" / "run_20261016" / "seed_20261024" / "trajectories.npz"


def reference_phase(
    contact: np.ndarray,
    *,
    episode_ids: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build debounced contact references without joining episode resets."""
    left = np.asarray(contact, dtype=bool)
    if left.ndim != 1:
        raise ValueError("contact must be one-dimensional")
    if episode_ids is None:
        episodes = np.zeros(len(left), dtype=np.int64)
    else:
        episodes = np.asarray(episode_ids)
        if episodes.shape != left.shape:
            raise ValueError("episode_ids must match contact")
    phase = np.full(len(left), np.nan, dtype=np.float32)
    cycle = np.full(len(left), -1, dtype=np.int64)
    touchdown = np.zeros(len(left), dtype=bool)
    segment_starts = np.r_[0, np.flatnonzero(episodes[1:] != episodes[:-1]) + 1]
    segment_ends = np.r_[segment_starts[1:], len(left)]
    next_cycle_id = 0
    for segment_start, segment_end in zip(segment_starts, segment_ends):
        if segment_end <= segment_start:
            continue
        filtered = debounce_contact_sequence(left[segment_start:segment_end])
        local_touchdowns = np.flatnonzero(
            filtered & ~np.concatenate(([False], filtered[:-1]))
        )
        starts = local_touchdowns + segment_start
        touchdown[starts] = True
        for start, end in zip(starts[:-1], starts[1:]):
            if end <= start:
                continue
            length = int(end - start)
            phase[start:end] = (
                2.0 * np.pi * np.arange(length, dtype=np.float64) / float(length)
            ).astype(np.float32)
            cycle[start:end] = next_cycle_id
            next_cycle_id += 1
    return phase, touchdown, cycle


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    checkpoint = torch.load(DATA_DIR / "phase_encoder_checkpoint.pt", map_location="cpu", weights_only=False)
    model = PhaseEncoder(29, 10, 128, preserve_time=bool(checkpoint["preserve_time"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    arrays = np.load(ROLLOUT, allow_pickle=True)
    joint = arrays["joint_pos"]  # [time, env, 29]
    contacts = arrays["contacts"]
    mean = np.asarray(checkpoint["train_mean"], dtype=np.float32)
    std = np.asarray(checkpoint["train_std"], dtype=np.float32)
    per_env = []
    for env in range(joint.shape[1]):
        target, touchdown, cycle = reference_phase(contacts[:, env, 0])
        valid = np.isfinite(target)
        if valid.sum() < 20:
            per_env.append({"env": env, "evaluable": False, "valid_count": int(valid.sum())})
            continue
        windows = []
        indices = []
        for t in np.flatnonzero(valid):
            if t < 9:
                continue
            windows.append((joint[t - 9:t + 1, env] - mean) / std)
            indices.append(t)
        if not windows:
            per_env.append({"env": env, "evaluable": False, "valid_count": 0})
            continue
        with torch.no_grad():
            pred = model(torch.from_numpy(np.asarray(windows, dtype=np.float32))).numpy()
        idx = np.asarray(indices)
        report = evaluate_phase_predictions(
            pred, target[idx],
            left_touchdown=touchdown[idx],
            cycle_ids=cycle[idx],
        )
        report.update({"env": env, "evaluable": True, "valid_count": int(len(idx))})
        per_env.append(report)
    valid_reports = [r for r in per_env if r.get("evaluable")]
    if not valid_reports:
        raise RuntimeError("no evaluable on-policy phase trajectories")
    macro_error = float(np.mean([r["circular_phase_error_rad"] for r in valid_reports]))
    macro_cycle = float(np.mean([r.get("cycle_completed_one_turn_fraction", 0.0) for r in valid_reports]))
    heldout = json.loads((DATA_DIR / "phase_encoder_report.json").read_text(encoding="utf-8"))
    threshold = float(heldout["phase_gate_threshold_rad"])
    gate = evaluate_rollout_gate(
        {"circular_phase_error_rad": heldout["test_report"]["circular_phase_error_rad"]},
        {"circular_phase_error_rad": macro_error},
        threshold_rad=threshold,
    )
    # A low pointwise circular error is insufficient for a locomotion phase:
    # the predicted angle must complete the observed cycles.  The protocol
    # treats a rollout with fewer than 90% complete one-turn cycle traces as
    # a phase-gate failure (standing/transition traces remain evaluable and
    # are reported above).
    progression_pass = bool(macro_cycle >= 0.90)
    gate["cycle_progression_pass"] = progression_pass
    gate["phase_rollout_gate_pass"] = bool(gate["phase_rollout_gate_pass"] and progression_pass)
    checkpoint_path = DATA_DIR / "phase_encoder_checkpoint.pt"
    result = {
        "schema": "g1.onpolicy_phase_gate.v1", "rollout": str(ROLLOUT),
        "encoder_checkpoint": str(checkpoint_path),
        "encoder_checkpoint_sha256": sha256_file(checkpoint_path),
        "input_contract": "joint_pos_history_only; contacts used only for offline reference",
        "per_environment": per_env, "macro_circular_phase_error_rad": macro_error,
        "macro_cycle_completed_one_turn_fraction": macro_cycle, "gate": gate,
        "progression_gate_rule": "macro_cycle_completed_one_turn_fraction >= 0.90",
        "decision": "PASS_ONPOLICY_PHASE_GATE" if gate["phase_rollout_gate_pass"] else "FAIL_ONPOLICY_PHASE_GATE",
    }
    (DATA_DIR / "onpolicy_phase_gate.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
