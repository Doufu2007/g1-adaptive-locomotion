"""Read the v2.1b epoch sweep and report the macro/point-error frontier.

The pre-registered selection requires BOTH v4.2 freeze eligibility AND a
condition-macro point error at or below pi/6. This script answers three
questions the single status line cannot:

  1. Was any epoch inside the required corner (both constraints)?
  2. If not, which constraint bound -- i.e. what is the best macro reachable
     at any point error at or below the cap, and the best point error at any
     macro at or above 0.90?
  3. Does the training curve agree with the validation metric? The point term
     is 1 - cos(angle error), so train point_loss maps to an angle by
     arccos(1 - point_loss). If that angle matches the validation point error
     the model has not yet fit the offset; if train is much smaller than
     validation the model is overfitting.

Reads only the sweep JSON. Never touches the test split.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SWEEP_PATH = ROOT / "results" / "g1_phase_gate_v4_2" / "v21b_epoch_sweep.json"
OUT_PATH = ROOT / "results" / "g1_phase_gate_v4_2" / "v21b_frontier.json"

CAP_RAD = math.pi / 6.0
TARGET_MACRO = 0.90


def train_point_angle_rad(point_loss: float) -> float:
    """Invert the mean 1 - cos(error) point term to an angle."""
    cosine = 1.0 - float(point_loss)
    return float(math.acos(max(-1.0, min(1.0, cosine))))


def main() -> int:
    payload = json.loads(SWEEP_PATH.read_text(encoding="utf-8"))
    epochs = payload["epochs"]
    if not epochs:
        raise ValueError("the sweep records no epochs")

    def macro(row: dict) -> float:
        return float(row["macro_completion"])

    def point(row: dict) -> float:
        return float(row["condition_macro_point_error_rad"])

    inside = [row for row in epochs if row["freeze_eligible"] and point(row) <= CAP_RAD]
    at_cap = [row for row in epochs if point(row) <= CAP_RAD]
    at_macro = [row for row in epochs if macro(row) >= TARGET_MACRO]

    best_macro_overall = max(epochs, key=macro)
    best_macro_at_cap = max(at_cap, key=macro) if at_cap else None
    best_point_overall = min(epochs, key=point)
    best_point_at_macro = min(at_macro, key=point) if at_macro else None

    curve = []
    for row in epochs:
        curve.append({
            "epoch": row["epoch"],
            "macro_completion": macro(row),
            "point_error_rad": point(row),
            "freeze_eligible": bool(row["freeze_eligible"]),
            "train_point_loss": float(row["train"]["point_loss"]),
            "train_point_angle_rad": train_point_angle_rad(row["train"]["point_loss"]),
            "train_increment_loss": float(row["train"]["increment_loss"]),
            "generalisation_gap_rad": float(
                point(row) - train_point_angle_rad(row["train"]["point_loss"])
            ),
        })

    if inside:
        binding = None
    elif not at_cap:
        binding = "point_error_cap"
    elif not at_macro:
        binding = "macro_completion"
    else:
        binding = "joint_never_satisfied"

    payload_out = {
        "schema": "g1.v21b_frontier.v1",
        "gate_version": payload.get("gate_version"),
        "run_status": payload.get("status"),
        "test_split_read": False,
        "point_error_cap_rad": CAP_RAD,
        "target_macro_completion": TARGET_MACRO,
        "epochs_inside_required_corner": [row["epoch"] for row in inside],
        "binding_constraint": binding,
        "best_macro_overall": {
            "epoch": best_macro_overall["epoch"],
            "macro_completion": macro(best_macro_overall),
            "point_error_rad": point(best_macro_overall),
        },
        "best_macro_at_or_below_point_cap": None if best_macro_at_cap is None else {
            "epoch": best_macro_at_cap["epoch"],
            "macro_completion": macro(best_macro_at_cap),
            "point_error_rad": point(best_macro_at_cap),
        },
        "best_point_overall": {
            "epoch": best_point_overall["epoch"],
            "macro_completion": macro(best_point_overall),
            "point_error_rad": point(best_point_overall),
        },
        "best_point_at_or_above_target_macro": None if best_point_at_macro is None else {
            "epoch": best_point_at_macro["epoch"],
            "macro_completion": macro(best_point_at_macro),
            "point_error_rad": point(best_point_at_macro),
        },
        "curve": curve,
    }
    OUT_PATH.write_text(json.dumps(payload_out, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    print(f"binding constraint          : {binding}")
    print(f"best macro overall          : {macro(best_macro_overall):.4f} "
          f"@ ep {best_macro_overall['epoch']} point {point(best_macro_overall):.3f}")
    if best_macro_at_cap:
        print(f"best macro at point <= cap  : {macro(best_macro_at_cap):.4f} "
              f"@ ep {best_macro_at_cap['epoch']} point {point(best_macro_at_cap):.3f}")
    else:
        print("best macro at point <= cap  : none (no epoch reached the point cap)")
    print(f"best point overall          : {point(best_point_overall):.3f} "
          f"@ ep {best_point_overall['epoch']} macro {macro(best_point_overall):.4f}")
    if best_point_at_macro:
        print(f"best point at macro >= 0.90 : {point(best_point_at_macro):.3f} @ ep {best_point_at_macro['epoch']}")
    else:
        print("best point at macro >= 0.90 : none (no epoch reached 0.90 macro)")
    print()
    print("ep  macro   point   train_pt_angle  gap")
    for row in curve:
        print(f"{row['epoch']:2d}  {row['macro_completion']:.4f}  {row['point_error_rad']:.4f}  "
              f"{row['train_point_angle_rad']:.4f}          {row['generalisation_gap_rad']:+.4f}")
    print()
    print(f"-> {OUT_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
