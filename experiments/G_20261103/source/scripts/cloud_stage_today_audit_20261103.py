"""Read-only G audit. No training, simulation, probe, signal or gate mutations.

The only publications are results/cloud_stage_today_20261103/latest_audit.json
and artifacts/cloud_today_latest.txt. All other inputs remain read-only.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import cloud_stage_today_20261103 as stage
from scripts import cloud_v2_campaign as frozen


def file_metadata(path):
    if not path.is_file():
        return {"exists": False}
    stat = path.stat()
    return {"exists": True, "bytes": stat.st_size,
        "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat()}


def command_for(pid):
    try:
        return Path(f"/proc/{int(pid)}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except (OSError, TypeError, ValueError):
        return None


def scalar_window(items):
    # A duplicate step is reported; the last-written scalar is used for trends.
    ordered = sorted(enumerate(items), key=lambda p: (int(p[1].step), p[0]))
    by_step = {int(item.step): item for _, item in ordered}
    values = [by_step[key] for key in sorted(by_step)][-200:]

    def summarize(part):
        finite = [float(item.value) for item in part if math.isfinite(float(item.value))]
        return {"samples": len(part), "finite_samples": len(finite),
            "first_step": int(part[0].step) if part else None,
            "last_step": int(part[-1].step) if part else None,
            "mean": sum(finite) / len(finite) if finite else None,
            "minimum": min(finite) if finite else None, "maximum": max(finite) if finite else None}

    recent = values[-100:]
    previous = values[-200:-100] if len(values) > 100 else []
    return {"available_unique_steps": len(by_step), "duplicate_step_records": len(items) - len(by_step),
        "trend_scope": "last up to 200 unique absolute TensorBoard steps; last-write scalar on duplicate steps",
        "previous100": summarize(previous), "recent100": summarize(recent),
        "complete_100_vs_100": len(previous) == len(recent) == 100,
        "latest_step": int(values[-1].step) if values else None,
        "latest_value": float(values[-1].value) if values and math.isfinite(float(values[-1].value)) else None}


def curves(run):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    accumulator = EventAccumulator(str(run), size_guidance={"scalars": 0})
    accumulator.Reload()
    tags = accumulator.Tags().get("scalars", [])
    selected = [tag for tag in tags if any(key in tag.lower() for key in
        ("track_linear_velocity", "style_imitation", "style_unmasked_error", "style_weight_mass",
         "episode_length", "surrogate", "value_function", "entropy", "learning_rate", "dropout"))]
    return {tag: scalar_window(accumulator.Scalars(tag)) for tag in selected}


def nvidia_processes():
    # Query-only CLI; this does not initialize a CUDA context or import torch.
    import subprocess
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits"], text=True, capture_output=True, timeout=8, check=True)
    rows = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 3:
            rows.append({"pid": int(fields[0]), "process_name": fields[1], "used_memory_mib": fields[2]})
    return rows


def audit():
    sampled = stage.stamp()
    began = time.monotonic()
    report = {"status": "running", "completed": False, "sampled_at_utc": sampled,
        "purpose": "READ_ONLY_GOAL_AND_MECHANISM_AUDIT_NO_SIMULATION_OR_TRAINING",
        "source_sha256": stage.sha(Path(__file__)), "issues": [], "models": [], "evaluations": []}
    issues = report["issues"]
    try:
        value = stage.verify()
        report["protocol_sha256"] = value["sha256"]
        report["protocol_verified"] = True
    except Exception as error:
        value = stage.read(stage.CONTROL / "protocol.json")
        report["protocol_verified"] = False
        issues.append({"kind": "protocol", "detail": f"{type(error).__name__}: {error}"})
    try:
        stage.safety()
        report["frozen_gate_and_E8_verified"] = True
    except Exception as error:
        report["frozen_gate_and_E8_verified"] = False
        issues.append({"kind": "admission_integrity", "detail": str(error)})
    report["state"] = stage.read(stage.CONTROL / "state.json")
    report["state_file"] = file_metadata(stage.CONTROL / "state.json")
    report["frozen_state"] = stage.read(frozen.CONTROL / "state.json")
    report["frozen_gate"] = stage.read(frozen.CONTROL / "plan_gate.json")
    report["frozen_E_launch_receipts_actual"] = {str(p.relative_to(frozen.CONTROL)): stage.sha(p)
        for p in (frozen.CONTROL / "E/train").glob("*/launch.json")}
    report["frozen_E_launch_receipts_expected"] = value.get("frozen_E_launch_receipts", {})
    report["attention"] = stage.read(stage.CONTROL / "attention.json")
    try:
        report["active_E_G"] = stage.launched_training()
        if len(report["active_E_G"]) > 8:
            issues.append({"kind": "parallelism", "detail": "More than eight real E+G training workers"})
    except Exception as error:
        issues.append({"kind": "process_identity", "detail": str(error)})
    try:
        _, report["resources"] = stage.resource_ready()
        report["cuda_processes"] = nvidia_processes()
    except Exception as error:
        issues.append({"kind": "resource_sample", "detail": f"{type(error).__name__}: {error}"})
    cuda_pids = {p["pid"] for p in report.get("cuda_processes", [])}
    for parent in value.get("parents", []):
        directory = stage.directory_for("train", parent)
        launch = stage.read(directory / "launch.json")
        result = stage.read(directory / "result.json")
        resource = stage.read(directory / "resource_state.json")
        mechanism = stage.read(directory / "training_audit.json")
        resume = stage.read(directory / "resume_receipt.json")
        run = ROOT / "logs/rsl_rl/g1_dropout_style_G_today_20261103" / f"{parent['group']}_seed{parent['seed']}"
        model = {"group": parent["group"], "seed": parent["seed"], "launch": launch, "result": result,
            "process_state": frozen.process_state(launch.get("pid")), "command": command_for(launch.get("pid")),
            "cuda_process_seen": launch.get("pid") in cuda_pids, "resource_state": resource,
            "resume_receipt": resume, "configuration_receipt": stage.read(directory / "configuration_receipt.json"),
            "mechanism": mechanism, "mechanism_file": file_metadata(directory / "training_audit.json"),
            "parent_sha256_expected": parent["sha256"]}
        try:
            model["parent_sha256_actual"] = stage.sha(Path(parent["checkpoint"]))
            if model["parent_sha256_actual"] != parent["sha256"]:
                issues.append({"kind": "parent_hash", "group": parent["group"], "seed": parent["seed"]})
        except Exception as error:
            issues.append({"kind": "parent_unreadable", "group": parent["group"], "seed": parent["seed"], "detail": str(error)})
        if resume and (resume.get("next_ppo_iteration") != 2000 or resume.get("env_common_step_counter") != 48000
                or resume.get("target_updates") != 3000 or resume.get("added_updates") != 1000):
            issues.append({"kind": "resume_contract", "group": parent["group"], "seed": parent["seed"]})
        if mechanism.get("mechanism_violation"):
            issues.append({"kind": "mechanism_violation", "group": parent["group"], "seed": parent["seed"]})
        if launch and not mechanism:
            model["mechanism_evidence_scope"] = "No mechanism report yet; startup or unavailable evidence, not proof of correctness"
        if result.get("status") in ("failed", "stopped"):
            issues.append({"kind": "worker_ended", "group": parent["group"], "seed": parent["seed"], "status": result["status"]})
        if result.get("status") == "failed":
            log = directory / "worker.log"
            if log.is_file():
                model["worker_log_tail"] = log.read_text(errors="replace").splitlines()[-60:]
        if list(run.glob("events.out.tfevents*")):
            try:
                model["curves"] = curves(run)
                for tag, window in model["curves"].items():
                    for name in ("previous100", "recent100"):
                        if window[name]["finite_samples"] != window[name]["samples"]:
                            issues.append({"kind": "nonfinite_scalar", "group": parent["group"], "seed": parent["seed"], "tag": tag, "window": name})
            except Exception as error:
                model["curves_error"] = f"{type(error).__name__}: {error}"
                issues.append({"kind": "curves_unavailable", "group": parent["group"], "seed": parent["seed"], "detail": str(error)})
        if result.get("final_checkpoint"):
            checkpoint = Path(result["final_checkpoint"])
            model["final_checkpoint_file"] = file_metadata(checkpoint)
            if checkpoint.is_file():
                model["final_checkpoint_sha256_actual"] = stage.sha(checkpoint)
                if model["final_checkpoint_sha256_actual"] != result.get("final_sha256"):
                    issues.append({"kind": "final_hash", "group": parent["group"], "seed": parent["seed"]})
        report["models"].append(model)
    for path in sorted((stage.CONTROL / "eval").glob("*/eval_seed*/evaluation.json")):
        evaluation = stage.read(path)
        report["evaluations"].append({"path": str(path), "file": file_metadata(path), **evaluation})
    report["evaluation_summary"] = stage.read(stage.CONTROL / "evaluation.json")
    report["backup_manifest"] = stage.read(ROOT / "artifacts/cloud_stage_today_20261103_results.sha256.json")
    report["evidence_limits"] = [
        "Audit completion means this read-only collection finished, not training/evaluation completion or research success.",
        "Fewer than 200 available scalar steps do not form a complete 100-versus-100 trend.",
        "32-environment mechanism samples do not describe every training trajectory; missing evidence is not a confirmed bug.",
        "Recovery weights and target holding do not prove physically frozen joint angles, visual naturalness, or method superiority.",
        "No complete local package verification is performed by this cloud read-only audit."]
    report.update(status="completed", completed=True, completed_at_utc=stage.stamp(), wall_seconds=time.monotonic()-began)
    target = stage.CONTROL / "latest_audit.json"
    stage.publish(target, report)
    envelope = {"audit": report, "audit_file": file_metadata(target)}
    stage.publish(ROOT / "artifacts/cloud_today_latest.txt", envelope)
    print("G_AUDIT_READY", target, "COMPLETED", report["completed_at_utc"], "ISSUES", len(issues))
    return 0


if __name__ == "__main__":
    raise SystemExit(audit())
