"""Freeze and verify G1 formal-matrix inputs before PPO starts."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCK = ROOT / "results" / "g1_phase_gate_v3" / "startup_hashes.json"
ENCODER_CHECKPOINT = ROOT / "results" / "g1_phase_dataset_20261019" / "phase_encoder_runtime_v1.pt"
REWARD_TEMPLATE = ROOT / "results" / "g1_phase_dataset_20261019" / "g1_reward_template_v1.json"
CALIBRATION = ROOT / "results" / "g1_phase_gate_v3" / "rollout_gate_calibration.json"
MATRIX = ROOT / "results" / "g1_phase_gate_v3" / "experiment_matrix.md"
ANALYSIS_CONTRACT = ROOT / "results" / "g1_phase_gate_v3" / "result_analysis_contract.json"
FAILURE_TREE_REPORT = ROOT / "results" / "g1_phase_gate_v3" / "failure_tree_report.json"
ENCODER_CANDIDATE_REPORT = ROOT / "results" / "g1_phase_gate_v3" / "encoder_candidate_validation.json"
CONFIG_PATHS = (
    ROOT / "src" / "tasks" / "velocity" / "config" / "g1" / "env_cfgs.py",
    ROOT / "src" / "tasks" / "velocity" / "velocity_env_cfg.py",
    ROOT / "src" / "tasks" / "velocity" / "config" / "g1" / "rl_cfg.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_hash_entries(entries: list[dict[str, str]], root: Path) -> list[str]:
    errors = []
    for entry in entries:
        path = Path(entry["path"])
        artifact = path if path.is_absolute() else root / path
        if not artifact.is_file():
            errors.append(f"{entry['name']}: missing artifact {entry['path']}")
        elif sha256_file(artifact) != entry["sha256"]:
            errors.append(f"{entry['name']}: SHA-256 mismatch for {entry['path']}")
    return errors


def _entry(name: str, path: Path) -> dict[str, str]:
    resolved = path.resolve(strict=True)
    try:
        relative = resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"locked artifact must be inside the project root: {resolved}") from error
    return {"name": name, "path": relative, "sha256": sha256_file(resolved)}


def startup_gate_blockers(
    confirmation: dict[str, Any], phase_report: dict[str, Any]
) -> list[str]:
    blockers: list[str] = []
    if confirmation.get("status") != "completed" or confirmation.get("run_kind") != "confirmation":
        blockers.append("independent confirmation rollout is not complete")
    if phase_report.get("cycle_coverage_status") != "adequate":
        blockers.append("independent confirmation cycle coverage is not adequate")
    if phase_report.get("status") != "pass" or phase_report.get("phase_rollout_gate_pass") is not True:
        blockers.append("independent confirmation phase gate failed")
    return blockers


def _collect_hash_entries(
    confirmation_manifest_path: Path,
    phase_report_path: Path,
    confirmation: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    policy_manifest_path = Path(confirmation["pilot_manifest_path"]).resolve(strict=True)
    policy_manifest = json.loads(policy_manifest_path.read_text(encoding="utf-8"))
    entries = [
        *[_entry(f"environment_config_{index}", path) for index, path in enumerate(CONFIG_PATHS)],
        _entry("encoder_runtime_checkpoint", ENCODER_CHECKPOINT),
        _entry("policy_checkpoint", Path(confirmation["policy_checkpoint_path"])),
        _entry("policy_onnx", Path(confirmation["policy_onnx_path"])),
        _entry("reward_template", REWARD_TEMPLATE),
        _entry("phase_gate_calibration", CALIBRATION),
        _entry("phase_confirmation_report", phase_report_path),
        _entry("phase_confirmation_manifest", confirmation_manifest_path),
        _entry("policy_training_manifest", policy_manifest_path),
        _entry("experiment_matrix", MATRIX),
        _entry("result_analysis_contract", ANALYSIS_CONTRACT),
        _entry("failure_tree_report", FAILURE_TREE_REPORT),
        _entry("encoder_validation_candidates", ENCODER_CANDIDATE_REPORT),
    ]
    for name, key in (
        ("policy_training_agent_config", "agent_config_path"),
        ("policy_training_environment_config", "environment_config_path"),
    ):
        if policy_manifest.get(key):
            entries.append(_entry(name, Path(policy_manifest[key])))
    return policy_manifest, entries


def preflight_startup_hashes(
    confirmation_manifest_path: Path,
    phase_report_path: Path,
    lock_path: Path = DEFAULT_LOCK,
) -> dict[str, Any]:
    """List current startup hashes and blockers without creating a ready lock."""
    confirmation_manifest_path = confirmation_manifest_path.resolve(strict=True)
    phase_report_path = phase_report_path.resolve(strict=True)
    confirmation = json.loads(confirmation_manifest_path.read_text(encoding="utf-8"))
    phase_report = json.loads(phase_report_path.read_text(encoding="utf-8"))
    blockers = startup_gate_blockers(confirmation, phase_report)
    artifacts: list[dict[str, str]] = []
    try:
        _policy_manifest, artifacts = _collect_hash_entries(
            confirmation_manifest_path, phase_report_path, confirmation
        )
    except (KeyError, OSError, ValueError) as error:
        blockers.append(f"startup artifact hash collection failed: {error}")
    return {
        "schema": "g1.experiment_startup_preflight.v1",
        "status": "blocked" if blockers else "ready_to_freeze",
        "blockers": blockers,
        "lock_path": str(lock_path.resolve()),
        "lock_written": False,
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "formal_ppo_started": False,
    }


def freeze_hash_lock(
    confirmation_manifest_path: Path,
    phase_report_path: Path,
    lock_path: Path = DEFAULT_LOCK,
) -> dict[str, Any]:
    confirmation_manifest_path = confirmation_manifest_path.resolve(strict=True)
    phase_report_path = phase_report_path.resolve(strict=True)
    confirmation = json.loads(confirmation_manifest_path.read_text(encoding="utf-8"))
    phase_report = json.loads(phase_report_path.read_text(encoding="utf-8"))
    blockers = startup_gate_blockers(confirmation, phase_report)
    if blockers:
        raise ValueError("hash lock requires passing startup gates: " + "; ".join(blockers))
    policy_manifest, entries = _collect_hash_entries(
        confirmation_manifest_path, phase_report_path, confirmation
    )
    lock = {
        "schema": "g1.experiment_startup_hashes.v1",
        "status": "ready",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_policy_seed": int(policy_manifest["seed"]),
        "confirmation_seed": int(confirmation["seed"]),
        "artifacts": entries,
        "formal_ppo_started": False,
    }
    lock_path = lock_path.resolve()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps(lock, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return lock


def check_hash_lock(lock_path: Path = DEFAULT_LOCK) -> dict[str, Any]:
    lock_path = lock_path.resolve(strict=True)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("schema") != "g1.experiment_startup_hashes.v1" or lock.get("status") != "ready":
        raise ValueError("startup hash lock is not ready")
    errors = verify_hash_entries(lock["artifacts"], ROOT)
    return {
        "schema": lock["schema"],
        "status": "pass" if not errors else "fail",
        "lock_path": str(lock_path),
        "artifact_count": len(lock["artifacts"]),
        "errors": errors,
        "formal_ppo_started": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--confirmation-manifest", type=Path)
    parser.add_argument("--phase-report", type=Path)
    args = parser.parse_args()
    if args.freeze and args.preflight:
        parser.error("--freeze and --preflight are mutually exclusive")
    if args.freeze:
        if args.confirmation_manifest is None or args.phase_report is None:
            parser.error("--freeze requires --confirmation-manifest and --phase-report")
        result = freeze_hash_lock(args.confirmation_manifest, args.phase_report, args.lock)
    elif args.preflight:
        if args.confirmation_manifest is None or args.phase_report is None:
            parser.error("--preflight requires --confirmation-manifest and --phase-report")
        result = preflight_startup_hashes(
            args.confirmation_manifest, args.phase_report, args.lock
        )
    else:
        result = check_hash_lock(args.lock)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result.get("status") in {"fail", "blocked"}:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
