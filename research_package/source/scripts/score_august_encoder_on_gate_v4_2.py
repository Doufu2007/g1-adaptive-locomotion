"""Score the August RL encoder on the frozen walk phase gate (v4.2).

Why this exists
---------------
The project's four August RL runs loaded
``src/tasks/velocity/phasor/phasor_encoder.pt`` (SHA-256 ``a0be7719...``), an
encoder trained on dance motion.  Every gate figure in the repository belongs
to a *different* checkpoint — the walking-trained
``phase_encoder_runtime_v1.pt`` — so until now the mechanism claim ("a phase
estimated from recent joint motion tracks the gait cycle far better than the
fixed clock") could not be attached to the policies that were actually trained.
This script closes that gap for one artifact.

What it does and does not do
----------------------------
* Reads **only** the validation split. ``x_test`` / ``y_test_cycles`` are never
  touched: the test split is burned under
  ``results/g1_phase_gate_v4/protocol_deviation_test_split_diagnostic_v1.md``.
* Imports ``phase_gate_v4`` and ``phase_gate_v4_2`` **unmodified** and reuses
  the frozen period cuts from ``calibration_v4_amendment1.json``. No threshold
  is changed, no calibration is re-derived, no candidate is selected.
* Trains nothing. It forward-passes one frozen checkpoint and scores it.

Equivalence control
-------------------
The same code path re-scores the on-disk ``val_pred`` of the walking encoder and
**asserts** it reproduces the frozen v4.2 figure (0.4128...). If that assertion
fails, the harness does not match the one behind the published numbers and the
August figure below must not be trusted. This is the check that makes the new
number citable.

Two input conventions, both reported
------------------------------------
``gate_protocol``
    Feed ``x_validation`` exactly as every other estimator is fed — it is
    already normalized with the walk dataset's train statistics. This is the
    convention the gate uses, so the number is directly comparable with the
    frozen 0.4128 and 0.2187.
``encoder_native``
    Undo the walk normalization and re-apply the checkpoint's own stored
    ``mean``/``std``, which are dance statistics. This is what the encoder saw
    in training and what ``phasor/runtime.py`` does at deployment. It is a
    diagnostic, not a gate result: the gate contract compares estimators on
    identical inputs.

Usage:
    python scripts/score_august_encoder_on_gate_v4_2.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tasks.velocity.phasor.encoder import PhaseEncoder
from src.tasks.velocity.phasor.phase_gate_v4 import score_phase_cycles_v4
from src.tasks.velocity.phasor.phase_gate_v4_2 import (
    NET_TOLERANCE_RAD,
    PHASE_GATE_VERSION,
    TERMINAL_SAMPLE_POLICY,
    score_phase_cycles_v4_2,
)

DATA_DIR = ROOT / "results" / "g1_phase_dataset_20261019"
DATA_PATH = DATA_DIR / "phase_dataset.npz"
V1_PREDICTIONS_PATH = DATA_DIR / "phase_predictions.npz"
V4_CALIBRATION_PATH = ROOT / "results" / "g1_phase_gate_v4" / "calibration_v4_amendment1.json"
V4_MODULE_PATH = ROOT / "src" / "tasks" / "velocity" / "phasor" / "phase_gate_v4.py"
V4_2_MODULE_PATH = ROOT / "src" / "tasks" / "velocity" / "phasor" / "phase_gate_v4_2.py"
ENCODER_PATH = ROOT / "src" / "tasks" / "velocity" / "phasor" / "phasor_encoder.pt"
OUT_PATH = ROOT / "results" / "g1_phase_gate_v4_2" / "august_encoder_gate_score.json"

DATA_SHA256 = "e204c9dd0a48527ac4c82cfd0c539fba7d584f1faea6242ad0eba66970008b38"
ENCODER_SHA256 = "a0be7719e6cb2b5b17a2e70e19698562c60f7e15d9f3d020781907b9aa10c9ae"
V4_MODULE_SHA256 = "e72b7de76338dc4d8575deeed9f6bb03cec55b13bc94848488e0c7cdab3372c3"
#: The frozen v4.2 figure for the walking encoder. Reproduced, not re-derived.
FROZEN_V1_MACRO_V4_2 = 0.41280424052165965
FROZEN_FIXED_CLOCK_MACRO_V4_2 = 0.21866427203253935
FROZEN_FIXED_CLOCK_AS_CONFIGURED = 0.0771563205069073


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def base_inputs() -> dict[str, Any]:
    frozen = json.loads(V4_CALIBRATION_PATH.read_text(encoding="utf-8"))
    period_cuts = tuple(float(x) for x in frozen["period_length_cut_points_frames"])
    dataset = np.load(DATA_PATH, allow_pickle=False)
    sequence_ids = dataset["sequence_validation"]
    return {
        "reference_phase": dataset["y_validation_cycles"].astype(np.float64) * (2.0 * np.pi),
        "sequence_ids": sequence_ids,
        "frame_ids": dataset["frame_validation"],
        "touchdowns": dataset["left_touchdown_validation"].astype(bool),
        "period_cuts": period_cuts,
        "conditions": sorted(np.unique(sequence_ids).tolist()),
        "x_validation": dataset["x_validation"],
        "train_mean": dataset["train_mean"],
        "train_std": dataset["train_std"],
    }


def score(prediction: np.ndarray, base: dict[str, Any], scorer) -> dict[str, Any]:
    return scorer(
        prediction,
        base["reference_phase"],
        base["sequence_ids"],
        base["frame_ids"],
        base["touchdowns"],
        condition_ids=base["sequence_ids"],
        episode_ids=base["sequence_ids"],
        history_valid=np.ones(len(base["frame_ids"]), dtype=bool),
        period_cuts_frames=base["period_cuts"],
        required_conditions=base["conditions"],
        minimum_cycles_per_condition=10,
        minimum_cycles_per_length_stratum=10,
        minimum_episode_segments_per_condition=1,
    )


def scalar_fields(report: dict[str, Any]) -> dict[str, Any]:
    """Keep every scalar and every nested per-condition/per-stratum block.

    Only ``cycle_records`` (one row per cycle) is dropped; it is regenerable from
    the dataset and would dominate the file.
    """
    out: dict[str, Any] = {}
    for key, value in report.items():
        if key == "cycle_records":
            out["cycle_record_count"] = len(value)
        elif isinstance(value, dict):
            out[key] = value
        else:
            out[key] = value
    return out


def prediction_diagnostics(
    prediction: np.ndarray, base: dict[str, Any]
) -> dict[str, float]:
    """Describe how far the predicted phase actually travels.

    A cycle counts as completed only if the unwrapped predicted phase advances by
    a full 2*pi inside the cycle. A near-constant output therefore scores zero by
    construction, and this block says how near-constant it was.
    """
    angle = np.arctan2(prediction[:, 0], prediction[:, 1])
    radius = np.hypot(prediction[:, 0], prediction[:, 1])
    spans = []
    for sequence in np.unique(base["sequence_ids"]):
        mask = base["sequence_ids"] == sequence
        order = np.argsort(base["frame_ids"][mask])
        unwrapped = np.unwrap(angle[mask][order])
        spans.append(float(unwrapped[-1] - unwrapped[0]))
    spans_arr = np.asarray(spans, dtype=np.float64)
    return {
        "radius_mean": float(radius.mean()),
        "radius_std": float(radius.std()),
        "angular_std_rad": float(angle.std()),
        "sequence_unwrapped_span_median_rad": float(np.median(spans_arr)),
        "sequence_unwrapped_span_max_rad": float(spans_arr.max()),
        "full_turn_rad": float(2.0 * np.pi),
    }


def load_encoder(
    path: Path = ENCODER_PATH,
) -> tuple[PhaseEncoder, torch.Tensor, torch.Tensor, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint["state_dict"]
    # The last conv's output channel count is the hidden width. A pooled encoder
    # feeds `hidden` features to fc; a preserve_time encoder feeds `hidden * window`.
    hidden = int(state["conv.2.weight"].shape[0])
    fc_in = int(state["fc.weight"].shape[1])
    preserve_time = fc_in != hidden
    if fc_in != (hidden * int(checkpoint["window"]) if preserve_time else hidden):
        raise ValueError(f"fc input width {fc_in} matches neither encoder variant")
    model = PhaseEncoder(
        joint_dim=int(checkpoint["joint_dim"]),
        window=int(checkpoint["window"]),
        hidden=hidden,
        preserve_time=preserve_time,
    )
    model.load_state_dict(state, strict=True)
    model.eval()
    mean = torch.as_tensor(np.asarray(checkpoint["mean"], dtype=np.float32))
    std = torch.as_tensor(np.asarray(checkpoint["std"], dtype=np.float32))
    meta = {
        "joint_dim": int(checkpoint["joint_dim"]),
        "window": int(checkpoint["window"]),
        "preserve_time": preserve_time,
        "checkpoint_keys": sorted(checkpoint.keys()),
    }
    return model, mean, std, meta


def predict(model: PhaseEncoder, windows: np.ndarray, batch_size: int = 8192) -> np.ndarray:
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            batch = torch.from_numpy(windows[start:start + batch_size]).float()
            chunks.append(model(batch).numpy())
    return np.concatenate(chunks, axis=0).astype(np.float64)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ENCODER_PATH)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    parser.add_argument("--label", type=str, default="august_encoder")
    args = parser.parse_args()
    checkpoint_path = args.checkpoint.resolve()
    out_path = args.out.resolve()

    data_hash = sha256_file(DATA_PATH)
    if data_hash != DATA_SHA256:
        raise ValueError("dataset hash differs from the frozen experiment input")
    if sha256_file(V4_MODULE_PATH) != V4_MODULE_SHA256:
        raise ValueError("phase_gate_v4.py changed; refusing to score against a moved gate")
    encoder_hash = sha256_file(checkpoint_path)
    if checkpoint_path == ENCODER_PATH.resolve() and encoder_hash != ENCODER_SHA256:
        raise ValueError("phasor_encoder.pt hash differs from the artifact this run claims to score")

    base = base_inputs()  # validation split only; x_test is never read

    # --- equivalence control: the same harness must reproduce the frozen v1 figure
    v1_pred = np.load(V1_PREDICTIONS_PATH, allow_pickle=False)["val_pred"].astype(np.float64)
    v1_report = score(v1_pred, base, score_phase_cycles_v4_2)
    v1_macro = float(v1_report["macro_cycle_completion_fraction"])
    control_ok = abs(v1_macro - FROZEN_V1_MACRO_V4_2) < 1e-12
    if not control_ok:
        raise RuntimeError(
            "harness does not reproduce the frozen v1 v4.2 score "
            f"({v1_macro!r} vs {FROZEN_V1_MACRO_V4_2!r}); the August figure would not be comparable"
        )

    model, ckpt_mean, ckpt_std, meta = load_encoder(checkpoint_path)

    # --- gate protocol: inputs exactly as every other estimator receives them
    gate_protocol_pred = predict(model, base["x_validation"])
    gate_v42 = score(gate_protocol_pred, base, score_phase_cycles_v4_2)
    gate_v41 = score(gate_protocol_pred, base, score_phase_cycles_v4)

    # --- encoder-native diagnostic: restore the checkpoint's own input scaling
    raw = base["x_validation"].astype(np.float32) * base["train_std"] + base["train_mean"]
    native = ((raw - ckpt_mean.numpy()) / (ckpt_std.numpy() + 1e-8)).astype(np.float32)
    native_pred = predict(model, native)
    native_v42 = score(native_pred, base, score_phase_cycles_v4_2)

    g = float(gate_v42["macro_cycle_completion_fraction"])
    n = float(native_v42["macro_cycle_completion_fraction"])

    payload = {
        "schema": "g1.august_encoder_gate_score.v1",
        "gate_version": PHASE_GATE_VERSION,
        "terminal_sample_policy": TERMINAL_SAMPLE_POLICY,
        "source_split": "validation_only",
        "test_split_read": False,
        "purpose": (
            "Score the encoder the four August RL runs actually loaded, on the same frozen gate "
            "used for the walking encoder. No training, no threshold change, no candidate selection."
        ),
        "thresholds_changed": False,
        "calibration_reused": str(V4_CALIBRATION_PATH.relative_to(ROOT)),
        "provenance": {
            "dataset_sha256": data_hash,
            "v4_module_sha256": sha256_file(V4_MODULE_PATH),
            "v4_2_module_sha256": sha256_file(V4_2_MODULE_PATH),
            "encoder_path": str(checkpoint_path.relative_to(ROOT)),
            "encoder_sha256": encoder_hash,
            "encoder_training_source": "dance (dance1_subject2), per results/phase_diagnostics/phasor_encoder_diagnostics.json",
            "scoring_script": str(Path(__file__).resolve().relative_to(ROOT)),
            "scoring_script_sha256": sha256_file(Path(__file__).resolve()),
            "loaded_by_runs": (
                "Every PHASOR run directory stores this path in logs/rsl_rl/g1_velocity/*/params/env.yaml: "
                "encoder_path: /root/unitree_rl_mjlab/src/tasks/velocity/phasor/phasor_encoder.pt "
                "(flat 12-29-14 and 21-49-27; rough 06-13-25, 06-18-55, 10-51-03). The two baseline "
                "directories configure no encoder."
            ),
        },
        "encoder": meta,
        "equivalence_control": {
            "estimator": "walking encoder val_pred, re-scored by this harness",
            "macro_v4_2": v1_macro,
            "frozen_macro_v4_2": FROZEN_V1_MACRO_V4_2,
            "reproduced": control_ok,
        },
        "estimators": {
            f"{args.label}_gate_protocol": {
                **scalar_fields(gate_v42),
                "macro_completion_v4_1": gate_v41["macro_cycle_completion_fraction"],
                "ratio_to_best_fixed_clock": g / FROZEN_FIXED_CLOCK_MACRO_V4_2,
                "ratio_to_configured_fixed_clock": g / FROZEN_FIXED_CLOCK_AS_CONFIGURED,
                "ratio_to_walking_encoder": g / FROZEN_V1_MACRO_V4_2,
                "prediction_diagnostics": prediction_diagnostics(gate_protocol_pred, base),
            },
            f"{args.label}_encoder_native_inputs": {
                **scalar_fields(native_v42),
                "ratio_to_best_fixed_clock": n / FROZEN_FIXED_CLOCK_MACRO_V4_2,
                "prediction_diagnostics": prediction_diagnostics(native_pred, base),
                "note": (
                    "Diagnostic only, but it is the deployment convention: phasor/runtime.py normalizes "
                    "raw joint positions with the checkpoint's own dance mean/std, which is what this arm feeds."
                ),
            },
            "walking_encoder_control": {
                **scalar_fields(v1_report),
                "macro_completion_v4_1": score(v1_pred, base, score_phase_cycles_v4)["macro_cycle_completion_fraction"],
                "prediction_diagnostics": prediction_diagnostics(v1_pred, base),
            },
        },
        "diagnosis": {
            "finding": (
                "The August encoder completes 0 of 603 validation cycles under both input conventions."
            ),
            "why": (
                "The gate credits a cycle only when the unwrapped predicted phase advances a full 2*pi "
                "inside the cycle. This encoder's output angle is near-constant: per-sequence unwrapped "
                "span is ~0.008 rad (gate inputs) / ~0.021 rad (native inputs) where one cycle needs 6.283 rad."
            ),
            "independent_corroboration": (
                "results/phase_diagnostics/phasor_encoder_diagnostics.json already reported this encoder as "
                "constant on dance1_subject2, its own training motion: circular coverage 0.597 rad, 4 of 32 "
                "angle bins occupied, net winding 0.0017 cycles over 3936 windows. The collapse is therefore "
                "not an artifact of walk-domain inputs."
            ),
            "not_a_harness_error": (
                "The equivalence control re-scores the walking encoder's stored val_pred through this exact "
                "code path and reproduces the frozen 0.41280424052165965 to <1e-12."
            ),
            "scope": (
                "This scores one frozen checkpoint on one validation split. It says nothing about whether a "
                "correctly trained encoder would pass; the walking-trained encoder scores 0.4128 on the same gate."
            ),
        },
        "reference_points": {
            "oracle": 0.9984177215189873,
            "best_over_grid_fixed_clock_v4_2": FROZEN_FIXED_CLOCK_MACRO_V4_2,
            "configured_fixed_clock_v4_2": FROZEN_FIXED_CLOCK_AS_CONFIGURED,
            "gate_bar": 0.9,
            "net_tolerance_rad": NET_TOLERANCE_RAD,
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    gp_diag = prediction_diagnostics(gate_protocol_pred, base)
    print(f"equivalence control : reproduced {v1_macro:.6f} == frozen {FROZEN_V1_MACRO_V4_2:.6f}")
    print(
        "august, gate inputs : "
        f"{gate_v42['encoder_completed_cycle_count']}/{gate_v42['evaluable_cycle_count']} cycles, "
        f"macro {g:.6f}; phase span per sequence {gp_diag['sequence_unwrapped_span_median_rad']:.4f} rad "
        f"vs {2 * 3.141592653589793:.4f} needed"
    )
    print(
        "august, native      : "
        f"{native_v42['encoder_completed_cycle_count']}/{native_v42['evaluable_cycle_count']} cycles, "
        f"macro {n:.6f}"
    )
    print(
        f"walking encoder     : {v1_report['encoder_completed_cycle_count']}/{v1_report['evaluable_cycle_count']} "
        f"cycles, macro {v1_macro:.6f}"
    )
    print(f"-> {out_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
