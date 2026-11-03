"""Measure, from scratch, how far each deployed encoder's phase actually travels.

Written to answer one question directly: is the August encoder's output really
near-constant on the *validation* split, or was that a misreading of a report?

This script deliberately does **not** import the gate scorer.  It loads the
checkpoint, runs the forward pass itself, and reports per-sequence unwrapped
span.  The label is measured the same way as a control: the label is a linear
ramp between touchdowns, so it must travel about one turn per cycle, and if it
does not the dataset or the frame ordering is the problem, not the encoder.

Read-only.  Does not touch the test split.  Does not write any artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tasks.velocity.phasor.encoder import PhaseEncoder

DATA_PATH = ROOT / "results" / "g1_phase_dataset_20261019" / "phase_dataset.npz"
AUGUST_PATH = ROOT / "src" / "tasks" / "velocity" / "phasor" / "phasor_encoder.pt"
RUNTIME_PATH = ROOT / "results" / "g1_phase_dataset_20261019" / "phase_encoder_runtime_v1.pt"
TWO_PI = 2.0 * np.pi


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_model(checkpoint: dict) -> tuple[PhaseEncoder, np.ndarray, np.ndarray]:
    state = checkpoint["state_dict"]
    hidden = int(state["conv.2.weight"].shape[0])
    fc_in = int(state["fc.weight"].shape[1])
    window = int(checkpoint["window"])
    preserve_time = fc_in != hidden
    model = PhaseEncoder(
        joint_dim=int(checkpoint["joint_dim"]),
        window=window,
        hidden=hidden,
        preserve_time=preserve_time,
    )
    model.load_state_dict(state, strict=True)
    model.eval()
    return (
        model,
        np.asarray(checkpoint["mean"], dtype=np.float32),
        np.asarray(checkpoint["std"], dtype=np.float32),
    )


def run(model: PhaseEncoder, windows: np.ndarray) -> np.ndarray:
    outputs = []
    with torch.no_grad():
        for start in range(0, len(windows), 16384):
            batch = torch.from_numpy(windows[start:start + 16384]).float()
            outputs.append(model(batch).numpy())
    return np.concatenate(outputs, axis=0).astype(np.float64)


def per_sequence_spans(angle: np.ndarray, sequence: np.ndarray, frame: np.ndarray) -> dict:
    rows = []
    for name in np.unique(sequence):
        mask = sequence == name
        order = np.argsort(frame[mask])
        unwrapped = np.unwrap(angle[mask][order])
        span = float(unwrapped[-1] - unwrapped[0])
        rows.append(
            {
                "sequence": str(name),
                "frames": int(mask.sum()),
                "span_rad": span,
                "span_turns": span / TWO_PI,
                "net_angle_min_rad": float(unwrapped.min()),
                "net_angle_max_rad": float(unwrapped.max()),
            }
        )
    spans = np.asarray([row["span_rad"] for row in rows], dtype=np.float64)
    return {
        "per_sequence": rows,
        "span_median_rad": float(np.median(spans)),
        "span_min_rad": float(spans.min()),
        "span_max_rad": float(spans.max()),
        "span_median_turns": float(np.median(spans) / TWO_PI),
        "sequences_with_at_least_half_turn": int(np.sum(spans >= np.pi)),
        "sequences_total": int(len(spans)),
    }


def describe(name: str, angle: np.ndarray, sequence: np.ndarray, frame: np.ndarray) -> dict:
    stats = per_sequence_spans(angle, sequence, frame)
    print(f"\n=== {name} ===")
    print(f"  angular std over all windows : {angle.std():.6f} rad")
    print(f"  per-sequence unwrapped span  : median {stats['span_median_rad']:+.6f} rad"
          f"  ({stats['span_median_turns']:+.6f} turns)")
    print(f"                                 min {stats['span_min_rad']:+.6f}"
          f"  max {stats['span_max_rad']:+.6f} rad")
    print(f"  full turn for comparison     : {TWO_PI:.6f} rad")
    print(f"  sequences reaching half turn : {stats['sequences_with_at_least_half_turn']}"
          f" / {stats['sequences_total']}")
    for row in stats["per_sequence"]:
        print(f"    {row['sequence']:<16} frames {row['frames']:>6}  "
              f"span {row['span_rad']:+.6f} rad  ({row['span_turns']:+.6f} turns)")
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    dataset = np.load(DATA_PATH, allow_pickle=False)
    sequence = dataset["sequence_validation"]
    frame = dataset["frame_validation"]
    train_mean = dataset["train_mean"].astype(np.float32)
    train_std = dataset["train_std"].astype(np.float32)
    raw = dataset["x_validation"].astype(np.float32) * train_std + train_mean

    print(f"dataset  : {DATA_PATH.name}  sha256 {sha256_file(DATA_PATH)[:16]}...")
    print(f"windows  : {raw.shape}  sequences: {sorted(np.unique(sequence).tolist())}")

    report: dict = {"dataset_sha256": sha256_file(DATA_PATH), "estimators": {}}

    # Control 0: the label itself.  A linear ramp between touchdowns, so its
    # per-sequence span must be large.  If this comes out flat, the problem is
    # the dataset or the frame ordering, not any encoder.
    label_angle = dataset["y_validation_cycles"].astype(np.float64) * TWO_PI
    report["estimators"]["label_ramp"] = describe(
        "label ramp (y_validation_cycles * 2*pi)", label_angle, sequence, frame
    )

    for name, path in (("august_encoder", AUGUST_PATH), ("runtime_v1_walking", RUNTIME_PATH)):
        if not path.exists():
            print(f"\n{name}: missing at {path}")
            continue
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model, ckpt_mean, ckpt_std = build_model(checkpoint)
        normalized = (raw - ckpt_mean) / (ckpt_std + np.float32(1e-8))
        prediction = run(model, normalized)
        angle = np.arctan2(prediction[:, 0], prediction[:, 1])
        entry = describe(f"{name}  ({path.name}, sha256 {sha256_file(path)[:16]}...)",
                         angle, sequence, frame)
        entry["checkpoint_sha256"] = sha256_file(path)
        entry["radius_mean"] = float(np.hypot(prediction[:, 0], prediction[:, 1]).mean())
        entry["angle_first_20_first_sequence"] = [
            float(x) for x in angle[sequence == np.unique(sequence)[0]][:20]
        ]
        report["estimators"][name] = entry
        print(f"  radius mean                  : {entry['radius_mean']:.9f}")
        print("  first 20 angles of "
              f"{np.unique(sequence)[0]} : "
              + " ".join(f"{x:+.4f}" for x in entry["angle_first_20_first_sequence"]))

    if args.json_out is not None:
        args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
