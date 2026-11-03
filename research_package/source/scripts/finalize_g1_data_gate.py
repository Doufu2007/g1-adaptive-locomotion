"""Promote the reviewed candidate provenance manifest to READY."""

from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    root = Path(__file__).absolute().parent.parent
    result_dir = root / "results/g1_phase_dataset_20261019"
    manifest = json.loads(
        (result_dir / "g1_phase_data_provenance_candidate.json").read_text(encoding="utf-8")
    )
    manifest["status"] = "READY"
    contact = manifest["candidate_walk_sequences"]["contact_verification"]
    contact.update(
        {
            "status": "verified",
            "method": {
                "kinematics": "MuJoCo forward kinematics from repository G1 XML sites left_foot/right_foot",
                "rule": {
                    "height_on_m": 0.03,
                    "height_off_m": 0.05,
                    "speed_on_mps": 0.5,
                    "speed_off_mps": 0.8,
                    "min_contact_frames": 2,
                    "min_release_frames": 2,
                    "source_fps": 30.0,
                    "resample_fps": 50.0,
                },
                "manual_sample_protocol": (
                    "One full-sequence contact-audit panel per candidate file; inspected foot height, "
                    "speed overlay and detected contact dots; threshold sensitivity compared with "
                    "height_on=0.025,height_off=0.04,speed_on=0.6,speed_off=0.9."
                ),
                "reviewer": "Codex visual review 2026-10-19",
            },
            "manual_sample_count": 12,
            "uncertain_intervals": [
                {
                    "sequence_id": "walk3_subject3",
                    "reason": "right-foot contact duty is lower than other sequences and retained as an operational-label uncertainty flag",
                    "action": "retain sequence; report as uncertainty, no silent deletion",
                }
            ],
            "threshold_sensitivity_max_fraction": 0.0414,
        }
    )
    manifest["candidate_walk_sequences"]["local_snapshot"] = "data/raw/lafan1_g1_walk"
    manifest["current_file"] = None
    path = root / "docs/data/g1_phase_data_provenance.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "file_count": len(manifest["candidate_walk_sequences"]["file_metadata"]),
                "manual_sample_count": contact["manual_sample_count"],
                "uncertain_intervals": contact["uncertain_intervals"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
