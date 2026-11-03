"""Isolated continuation study and cooperative scheduling; never alters v1 outputs.

The supervisor admits jobs at boundaries. It never sends training signals or
writes/removes stop requests. In-process guards honor user stops and deadlines.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CONTROL = ROOT / "results/cloud_v2"
REFERENCE = ROOT / "data/style_dropout/walking_reference.npz"
AUDIT = ROOT / "artifacts/cloud_design_audit_20261103.txt"
TRAIN_END = "2026-11-04T08:00:00+00:00"
EVAL_END = "2026-11-04T10:00:00+00:00"
EXPORT_END = "2026-11-04T10:30:00+00:00"
REFERENCE_SHA = "4dcbf6eda5e43a44f9256f0a97f0ae511cf0a26ca0ec24c9289fe746c572e647"
V1_ZIP_SHA = "b85c1728198a82958e0f31f631c8740499ae66705db65a79681953b32e69a1e8"
NEW_SOURCES = ("scripts/cloud_v2_campaign.py", "scripts/cloud_v2_runtime.py", "scripts/cloud_v2_probe.py")


def stamp():
    return datetime.now(timezone.utc).isoformat()


def epoch(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {} if default is None else default


def publish(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            h.update(block)
    return h.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def manual_stop_paths(directory=None):
    paths = [ROOT / "results/cloud_campaign/stop.request",
             ROOT / "results/style_dropout_training/stop.request", CONTROL / "stop.request"]
    if directory is not None:
        paths.append(Path(directory) / "stop.request")
    return [str(p) for p in paths if p.exists()]


def admission(now, phase, training_seconds, evaluation_seconds):
    """Conservative whole-block admission, including evaluation/export reserves."""
    if now >= epoch(EXPORT_END):
        return False
    if phase == "training":
        return (now + training_seconds < epoch(TRAIN_END)
                and now + training_seconds + evaluation_seconds < epoch(EVAL_END) - 1800)
    return now + evaluation_seconds < epoch(EVAL_END) - 600


def environment(threads=3):
    env = os.environ.copy()
    env.update(G1_CLOUD_MODE="1", PYTHONUNBUFFERED="1", OMP_NUM_THREADS=str(threads),
               OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS=str(threads), MUJOCO_GL="egl",
               WANDB_MODE="offline", PYTHONPATH=str(ROOT), CUDA_VISIBLE_DEVICES="0",
               XDG_CACHE_HOME=str(ROOT.parent / "cache"),
               TORCHINDUCTOR_CACHE_DIR=str(ROOT.parent / "cache/torchinductor"))
    return env


def prepare():
    from scripts.style_resume import saved_iteration
    from scripts.style_evaluation_protocol import protocol
    if (CONTROL / "protocol.json").exists():
        raise FileExistsError("v2 protocol already exists; do not replace a frozen study.")
    if manual_stop_paths():
        raise RuntimeError(f"Respect existing manual stop: {manual_stop_paths()}")
    audit = read(AUDIT)
    models = audit["models"]
    assert {(m["group"], m["seed"]) for m in models} == {(g, s) for g in "ABCD" for s in range(42, 47)}
    assert sha(REFERENCE) == REFERENCE_SHA
    assert sha(ROOT / "artifacts/cloud_latest_results.zip") == V1_ZIP_SHA
    for rel, source in audit["sources"].items():
        if sha(ROOT / rel) != source["sha256"]:
            raise RuntimeError(f"Audited v1 source changed: {rel}")
    parents = []
    for seed in range(42, 47):
        groups = "ABCD"[seed % 4:] + "ABCD"[:seed % 4]
        for group in groups:
            model = next(m for m in models if m["group"] == group and m["seed"] == seed)
            checkpoint = Path(model["checkpoint"])
            assert saved_iteration(checkpoint) == 1999
            assert sha(checkpoint) == model["checkpoint_sha256"]
            assert int(model["scene_num_envs"]) == 1024
            assert int(model["agent"]["num_steps_per_env"]) == 24
            assert int(model["agent"]["max_iterations"]) == 2000
            assert float(model["decimation"]) * float(model["simulation"]["mujoco"]["timestep"]) == .02
            assert list(map(float, model["action"]["dropout_probability_range"])) == [.25, .75]
            assert list(map(int, model["action"]["duration_steps_range"])) == [5, 20]
            parents.append({"group": group, "seed": seed, "checkpoint": str(checkpoint),
                            "sha256": model["checkpoint_sha256"], "completed_updates": 2000})
    value = {"version": "v2_E_F_20261103", "created_at_utc": stamp(),
             "purpose": "DEVELOPMENT_ROBOT_CONTROL_CONTINUATION_NOT_CONFIRMATORY",
             "parents": parents, "num_envs": 1024, "rollout": 24,
             "parent_updates": 2000, "added_updates": 2000, "target_updates": 4000,
             "branches": {"E": {"change": "same_distribution_continuation"},
                          "F": {"change": "expanded_fault_distribution_only", "ramp_added_updates": 500,
                                "max_probability_start": .75, "max_probability_end": 1.,
                                "max_duration_steps_start": 20, "max_duration_steps_end": 40,
                                "full_loss_long_event_mass_start": 0., "full_loss_long_event_mass_end": .25,
                                "full_loss_duration": "current ramp maximum; other events uniform in expanded bounds"}},
             "resume": "strict actor/critic/optimizer; carry optimizer LR into adaptive scalar; new env RNG, not bitwise trajectory replay",
             "reference_sha256": REFERENCE_SHA, "evaluation": protocol(),
             "midcourse_checkpoints": [2500, 3000, 3500],
             "midcourse_rule": "diagnose configuration/NaN/severe failures and matched checkpoint panels; never select arms by reward or one unmatched probe",
             "comparisons": ["within_branch_D_vs_B_C_A", "F_vs_E_within_group_paired_parent", "E_vs_v1_budget_diagnostic"],
             "deadlines": {"training": TRAIN_END, "evaluation": EVAL_END, "export": EXPORT_END},
             "v1_zip_sha256": V1_ZIP_SHA,
             "sources": {**{k: v["sha256"] for k, v in audit["sources"].items()},
                         **{str(p.relative_to(ROOT)): sha(p) for p in (ROOT / "src").rglob("*.py")},
                         **{k: sha(ROOT / k) for k in (*NEW_SOURCES, "scripts/style_evaluation_protocol.py")}},
             "interpretation": "Report all five seed pairs and nominal/severe scenarios. No guaranteed positive effect, no humanlike appearance claim from joint RMSE alone."}
    value["sha256"] = canonical_sha(value)
    publish(CONTROL / "protocol.json", value)
    shutil.copy2(AUDIT, CONTROL / "parent_design_audit.json")
    for branch in "EF":
        publish(CONTROL / branch / "protocol.json", {"branch": branch, "campaign_protocol_sha256": value["sha256"], **value["branches"][branch]})
    publish(CONTROL / "decision.json", {"at_utc": stamp(), "decision": "admit E then F, then complete paired evaluation", "reason": "tail tracking still improving; original max 0.4s faults do not cover severe 0.8s evaluation; v1 preserved"})
    print("V2_PREPARED", value["sha256"], "40 jobs; same 20 parent checkpoints")


def verify_protocol():
    value = read(CONTROL / "protocol.json")
    digest = value["sha256"]
    body = {k: v for k, v in value.items() if k != "sha256"}
    if canonical_sha(body) != digest:
        raise RuntimeError("Frozen protocol integrity failed")
    for rel, expected in value["sources"].items():
        if sha(ROOT / rel) != expected:
            raise RuntimeError(f"Source drift; new version required: {rel}")
    if sha(REFERENCE) != value["reference_sha256"]:
        raise RuntimeError("Reference drift")
    return value


def directory_for(branch, kind, parent):
    return CONTROL / branch / kind / f"{parent['group']}_seed{parent['seed']}"


def process_state(pid):
    try:
        return Path(f"/proc/{int(pid)}/stat").read_text().split(") ", 1)[1].split()[0]
    except (OSError, TypeError, ValueError):
        return None


def launch(kind, branch, index, directory):
    directory.mkdir(parents=True, exist_ok=True)
    log = (directory / "worker.log").open("a")
    command = [sys.executable, str(Path(__file__)), kind, "--branch", branch, "--index", str(index)]
    process = subprocess.Popen(command, cwd=ROOT, env=environment(), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    log.close()
    publish(directory / "launch.json", {"at_utc": stamp(), "pid": process.pid, "command": command})
    return process


def estimates(value, stage, jobs, started):
    # Include a slow-job tail as well as aggregate work. Projections are estimates.
    done_steps = 0
    tails = []
    for parent in value["parents"]:
        directory = directory_for(stage, "train", parent)
        r = read(directory / "resource_state.json")
        completed = max(0, int(r.get("iteration", 1999)) + 1 - 2000)
        if read(directory / "result.json").get("status") == "completed":
            completed = 2000
        done_steps += min(completed, 2000) * 24576
        telemetry = Path(r.get("log_directory", "/nonexistent")) / "resource_telemetry.csv"
        if telemetry.is_file():
            with telemetry.open() as stream:
                rows = list(csv.DictReader(stream))[-100:]
            intervals = [epoch(b["updated_at_utc"]) - epoch(a["updated_at_utc"]) for a, b in zip(rows, rows[1:])]
            if intervals:
                tails.append((2000 - completed) * statistics.median(intervals))
    elapsed = max(1, time.time() - started)
    measured = done_steps / elapsed if elapsed > 300 and done_steps else None
    throughput = min(34529., measured) if measured else 34529.
    remaining = (20 * 2000 * 24576 - done_steps) / max(throughput, 1)
    return {"elapsed_s": elapsed, "completed_added_env_steps": done_steps,
            "aggregate_end_to_end_steps_s": measured, "projected_remaining_s_with_margin": 1.25 * max(remaining, max(tails, default=0)),
            "projection_uses_historical_rate": measured is None}


def supervise():
    import fcntl
    value = verify_protocol()
    lock = (CONTROL / "supervisor.lock").open("a+")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    if manual_stop_paths():
        raise RuntimeError(f"Manual stop present: {manual_stop_paths()}")
    # No automatic retries of previously stopped/failed/launched jobs.
    if list(CONTROL.glob("[EF]/train/*/launch.json")) or list(CONTROL.glob("[EF]/eval/*/launch.json")):
        raise RuntimeError("Existing v2 launches require explicit audit; automatic restart is disabled")
    started = time.time()
    for branch in "EF":
        # Admission is for all 20 models, with both evaluation matrices reserved.
        training_left = (2 if branch == "E" else 1) * 983040000 / 34529 * 1.25 + 1800
        evaluation_left = 2280 * (13880 / 1140) * 1.25
        if not admission(time.time(), "training", training_left, evaluation_left):
            publish(CONTROL / "state.json", {"status": "admission_deferred", "at_utc": stamp(), "stage": branch, "reason": "complete comparison plus evaluation reserve does not fit; replan required"})
            return 4
        pending = list(enumerate(value["parents"]))
        workers = []
        completed = 0
        failure = None
        phase_started = time.time()
        while pending or workers:
            for item in list(workers):
                process, parent, directory = item
                code = process.poll()
                if code is not None:
                    workers.remove(item)
                    if code == 0 and read(directory / "result.json").get("status") == "completed":
                        completed += 1
                    else:
                        failure = {"group": parent["group"], "seed": parent["seed"], "exit_code": code, "directory": str(directory)}
            stops = manual_stop_paths()
            if stops or time.time() >= epoch(TRAIN_END):
                failure = {"reason": "manual_stop_or_deadline", "paths": stops}
            if failure:
                pending = []  # Drain existing workers; their own guards honor stops.
            # Stagger allocations; cached resource guard still runs in every worker.
            gate = read(CONTROL / "plan_gate.json")
            if pending and len(workers) < 8 and gate.get("admit_new_jobs", True):
                index, parent = pending.pop(0)
                directory = directory_for(branch, "train", parent)
                workers.append((launch("train", branch, index, directory), parent, directory))
            estimate = estimates(value, branch, workers, phase_started)
            publish(CONTROL / "state.json", {"status": "draining_after_issue" if failure else "training" if gate.get("admit_new_jobs", True) else "new_admission_held", "at_utc": stamp(),
                    "supervisor_pid": os.getpid(), "stage": branch, "completed_models": completed, "planned_models": 20,
                    "pending_models": len(pending), "workers": [{"pid": p.pid, "group": j["group"], "seed": j["seed"], "directory": str(d)} for p, j, d in workers],
                    "failure": failure, "estimate": estimate, "plan_gate": gate, "protocol_sha256": value["sha256"]})
            time.sleep(5)
        if failure:
            publish(CONTROL / "attention.json", {"at_utc": stamp(), "stage": branch, "failure": failure, "automatic_retry": False})
            return 3
        backup()
    # Formal evaluation starts only after both equal-budget branches finish.
    pending = [(branch, index, parent) for index, parent in enumerate(value["parents"]) for branch in "EF"]
    workers = []
    completed = 0
    failure = None
    if not admission(time.time(), "evaluation", 0, 2280 * (13880 / 1140) * 1.25):
        publish(CONTROL / "state.json", {"status": "evaluation_admission_deferred", "at_utc": stamp(), "reason": "full evaluation reserve no longer fits; audit required"})
        return 4
    while pending or workers:
        for item in list(workers):
            process, branch, parent, directory = item
            if process.poll() is not None:
                workers.remove(item)
                if process.returncode == 0 and read(directory / "evaluation.json").get("status") == "completed":
                    completed += 1
                else:
                    failure = {"branch": branch, "group": parent["group"], "seed": parent["seed"], "exit_code": process.returncode}
        if manual_stop_paths() or time.time() >= epoch(EVAL_END):
            failure = {"reason": "manual_stop_or_deadline"}
        if failure:
            pending = []
        if pending and len(workers) < 8 and read(CONTROL / "plan_gate.json").get("admit_new_jobs", True):
            branch, index, parent = pending.pop(0)
            directory = directory_for(branch, "eval", parent)
            workers.append((launch("eval", branch, index, directory), branch, parent, directory))
        reports = [read(p) for p in CONTROL.glob("[EF]/eval/*/evaluation.json")]
        publish(CONTROL / "state.json", {"status": "draining_after_issue" if failure else "evaluating", "at_utc": stamp(),
                "supervisor_pid": os.getpid(), "completed_model_evaluations": completed, "completed_cases": sum(r.get("completed_cases", 0) for r in reports),
                "planned_cases": 2280, "pending_models": len(pending), "failure": failure,
                "workers": [{"pid": p.pid, "branch": b, "group": j["group"], "seed": j["seed"], "directory": str(d)} for p, b, j, d in workers]})
        time.sleep(5)
    backup()
    publish(CONTROL / "state.json", {"status": "attention" if failure else "completed", "at_utc": stamp(),
            "completed_model_evaluations": completed, "planned_model_evaluations": 40,
            "failure": failure, "off_server_backup_verified": False, "elapsed_s": time.time() - started})
    return 3 if failure else 0


def backup():
    """New v2 package, excludes live intermediate checkpoints and old frozen ZIP."""
    files = list(CONTROL.rglob("*.json")) + list(CONTROL.rglob("*.jsonl")) + list(CONTROL.rglob("*.log"))
    for branch in "EF":
        for run in (ROOT / "logs/rsl_rl" / f"g1_dropout_style_v2_{branch}").glob("*"):
            files += list(run.glob("model_3999.pt")) + list(run.glob("params/*.yaml")) + list(run.glob("*.json"))
            files += list(run.glob("events.out.tfevents*")) + list(run.glob("resource_telemetry.csv"))
    files += [ROOT / s for s in read(CONTROL / "protocol.json").get("sources", NEW_SOURCES)] + [REFERENCE, REFERENCE.with_suffix(".json")]
    target = ROOT / "artifacts/cloud_v2_results.zip"
    temporary = target.with_suffix(".tmp.zip")
    unique = sorted({p for p in files if p.is_file()})
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in unique:
            archive.write(path, path.relative_to(ROOT))
    temporary.replace(target)
    publish(target.with_suffix(".sha256.json"), {"at_utc": stamp(), "sha256": sha(target), "bytes": target.stat().st_size, "files": len(unique), "off_server_verified": False})


def inspect_status():
    from scripts.cloud_training_monitor import resources
    value = verify_protocol()
    result = {"sampled_at_utc": stamp(), "state": read(CONTROL / "state.json"), "resources": resources(),
              "manual_stop_paths": manual_stop_paths(), "protocol_sha256": value["sha256"], "models": [], "evaluations": []}
    for branch in "EF":
        for parent in value["parents"]:
            d = directory_for(branch, "train", parent)
            r = read(d / "resource_state.json")
            launch_receipt = read(d / "launch.json")
            item = {"branch": branch, "group": parent["group"], "seed": parent["seed"],
                    "resource_state": r, "launch": launch_receipt, "process_state": process_state(launch_receipt.get("pid")),
                    "result": read(d / "result.json"), "resume": read(d / "resume_receipt.json"),
                    "configuration": read(d / "configuration_receipt.json"),
                    "mechanism": read(d / "training_audit.json")}
            run = Path(r.get("log_directory", "/nonexistent"))
            if run.is_dir():
                from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
                ea = EventAccumulator(str(run), size_guidance={"scalars": 0})
                ea.Reload()
                item["curves"] = {}
                for tag in ea.Tags().get("scalars", []):
                    if any(s in tag for s in ("style_", "dropout_", "track_linear", "mean_episode_length", "Loss/")) and not tag.endswith("/time"):
                        xs = ea.Scalars(tag)
                        item["curves"][tag] = {"last_step": xs[-1].step, "last": xs[-1].value,
                                                "recent100_mean": statistics.mean(x.value for x in xs[-100:]),
                                                "previous100_mean": statistics.mean(x.value for x in xs[-200:-100]) if len(xs) > 100 else None}
            result["models"].append(item)
            e = read(directory_for(branch, "eval", parent) / "evaluation.json")
            if e:
                result["evaluations"].append(e)
    result["v1_zip_unchanged"] = sha(ROOT / "artifacts/cloud_latest_results.zip") == V1_ZIP_SHA
    result["completed_at_utc"] = stamp()
    publish(CONTROL / "latest_audit.json", result)
    export = ROOT / "artifacts/cloud_v2_latest_audit.txt"
    publish(export, result)
    print("V2_AUDIT_READY", export, "PHASE", result["state"].get("status"), "MODELS", len(result["models"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "supervise", "train", "eval", "audit", "backup", "validate"))
    parser.add_argument("--branch", choices=list("EF"))
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("Cloud-only campaign; local workstation training is disabled")
    if args.action == "prepare":
        return prepare()
    if args.action == "supervise":
        return supervise()
    if args.action == "audit":
        return inspect_status()
    if args.action == "backup":
        return backup()
    from scripts.cloud_v2_runtime import train_worker, evaluate_worker, validate
    if args.action == "validate":
        return validate()
    value = verify_protocol()
    if args.index is None or not 0 <= args.index < 20 or args.branch is None:
        parser.error("Worker requires --branch E/F and --index 0..19")
    return (train_worker if args.action == "train" else evaluate_worker)(value, args.branch, args.index)


if __name__ == "__main__":
    raise SystemExit(main())
