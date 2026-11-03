"""Independent G development experiment. Never reopens or resumes frozen E/F.

This controller owns only its new G workers. Existing E workers drain naturally.
All commands except validate are intended for the already rented Linux host.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import cloud_v2_campaign as frozen

CONTROL = ROOT / "results/cloud_stage_today_20261103"
REFERENCE = frozen.REFERENCE
TRAIN_ADMIT_END = "2026-11-03T20:00:00+08:00"
EVAL_END = "2026-11-03T22:00:00+08:00"
EXPORT_END = "2026-11-03T23:00:00+08:00"
SCENARIOS = ("nominal", "left_knee_75pct_0p4s", "right_ankle_75pct_0p4s", "right_ankle_100pct_0p8s")
EVAL_SEEDS = (1042, 1043, 1044)
ANALYSIS_RESERVE_S = 2700
EXPORT_RESERVE_S = 3600
SLOW_EVAL_CASE_SECONDS = 14.878
SOURCES = ("scripts/cloud_stage_today_20261103.py", "scripts/cloud_stage_today_runtime_20261103.py")
stamp, read, publish, sha, epoch = frozen.stamp, frozen.read, frozen.publish, frozen.sha, frozen.epoch


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def directory_for(kind, parent):
    return CONTROL / kind / f"{parent['group']}_seed{parent['seed']}"


def process_alive(pid):
    return frozen.process_state(pid) not in (None, "Z", "X")


def safety(acknowledgement=False):
    stops = frozen.manual_stop_paths(CONTROL)
    require(not stops, f"User stop remains active: {stops}")
    gate = read(frozen.CONTROL / "plan_gate.json")
    require(gate.get("admit_new_jobs") is False, "Frozen E/F gate must remain explicitly false")
    require(not list((frozen.CONTROL / "F/train").glob("*/launch.json")), "F launch exists; explicit re-audit required")
    require(not list(frozen.CONTROL.glob("[EF]/eval/*/launch.json")), "Frozen formal evaluation already launched")
    original_e = {str(path.relative_to(frozen.CONTROL)): sha(path) for path in (frozen.CONTROL / "E/train").glob("*/launch.json")}
    expected_e = read(CONTROL / "protocol.json").get("frozen_E_launch_receipts")
    if expected_e is not None:
        require(original_e == expected_e, "Original E launch set/receipt changed after independent G freeze")
    if acknowledgement:
        state = read(frozen.CONTROL / "state.json")
        require(state.get("plan_gate", {}).get("admit_new_jobs") is False,
                "Wait for original supervisor's false-gate acknowledgement")
        require(not state.get("failure"), "Original supervisor reports a failure")
        gate_time = gate.get("at_utc") or gate.get("updated_at_utc")
        if gate_time:
            require(epoch(state["at_utc"]) >= epoch(gate_time), "False-gate acknowledgement predates current gate")
    return gate


def verify():
    value = read(CONTROL / "protocol.json")
    require(value, "Prepare the independent G protocol first")
    require(frozen.canonical_sha({k: v for k, v in value.items() if k != "sha256"}) == value.get("sha256"),
            "Independent protocol integrity failed")
    original = frozen.verify_protocol()
    require(original["sha256"] == value["frozen_campaign_protocol_sha256"], "Frozen campaign identity changed")
    for rel, expected in value["sources"].items():
        require(sha(ROOT / rel) == expected, f"Independent source drift: {rel}")
    return value


def prepare():
    safety(acknowledgement=True)
    require(not (CONTROL / "protocol.json").exists(), "Protocol already frozen; do not overwrite it")
    original = frozen.verify_protocol()
    validation = read(CONTROL / "software_validation.json")
    require(validation.get("status") == "passed" and validation.get("cpu_only") is True,
            "Run validate --runtime-checks on the rented host before preparing")
    require(validation.get("sources") == {s: sha(ROOT / s) for s in SOURCES}, "CPU validation source identity is stale")
    require(sha(ROOT / "artifacts/cloud_latest_results.zip") == frozen.V1_ZIP_SHA, "Frozen v1 ZIP changed")
    from scripts.style_resume import saved_iteration
    from scripts.style_evaluation_protocol import scenarios
    parents = original["parents"]
    e_receipts = {str(path.relative_to(frozen.CONTROL)): sha(path) for path in (frozen.CONTROL / "E/train").glob("*/launch.json")}
    expected_e_paths = {str(Path("E/train") / f"{g}_seed{s}" / "launch.json") for g in "ABCD" for s in (42,43)}
    require(set(e_receipts) == expected_e_paths, "Expected exactly the eight original E seed42/43 launches; re-audit required")
    require({(p["group"], p["seed"]) for p in parents} == {(g, s) for g in "ABCD" for s in range(42, 47)}, "Need all 20 parents")
    for parent in parents:
        require(sha(Path(parent["checkpoint"])) == parent["sha256"] and saved_iteration(Path(parent["checkpoint"])) == 1999,
                "Parent checkpoint identity or 2000-update budget mismatch")
    grid = scenarios()
    value = {
        "version": "G_today_20261103", "created_at_utc": stamp(),
        "purpose": "DEVELOPMENT_ONLY_NO_GUARANTEED_POSITIVE_EFFECT",
        "frozen_campaign_protocol_sha256": original["sha256"], "parents": parents,
        "frozen_E_launch_receipts": e_receipts,
        "branch": "G", "parent_updates": 2000, "added_updates": 1000, "target_updates": 3000,
        "num_envs": 1024, "rollout": 24, "maximum_total_E_G_train_workers": 8,
        "cpu_threads_per_worker": 3, "reference_sha256": original["reference_sha256"],
        "distribution": {"origin_steps": 48000, "ramp_steps": 12000, "stable_steps": 12000,
            "probability_range_start": [.25, .75], "probability_range_end": [.25, 1.],
            "duration_steps_start": [5, 20], "duration_steps_end": [5, 40],
            "full_loss_long_event_mass_start": 0., "full_loss_long_event_mass_end": .25,
            "event_probability": .005, "fault_joint_count_range": [1, 2]},
        "evaluation": {"scenarios": {s: grid[s] for s in SCENARIOS}, "evaluation_seeds": list(EVAL_SEEDS),
            "num_envs": 128, "duration_s": 20., "planned_cases": 240,
            "order": "complete all 20 models and four scenarios for one paired evaluation seed before next seed",
            "analysis_unit": "five training seeds; average paired evaluation repeats within each seed"},
        "deadlines": {"train_admission": TRAIN_ADMIT_END, "evaluation": EVAL_END, "export": EXPORT_END},
        "analysis_reserve_s": ANALYSIS_RESERVE_S, "export_reserve_s": EXPORT_RESERVE_S,
        "sources": {s: sha(ROOT / s) for s in SOURCES},
        "interpretation": "G-v1 changes both budget and fault distribution. E8 are incomplete budget diagnostics, excluded from G comparisons. Joint RMSE is not visual or real-robot naturalness proof."}
    value["sha256"] = frozen.canonical_sha(value)
    publish(CONTROL / "protocol.json", value)
    publish(CONTROL / "decision.json", {"at_utc": stamp(), "decision": "independent_G_equal_budget_all20",
        "frozen_E_F_gate": False, "original_E_unlaunched": "deferred", "F": "deferred",
        "no_training_stop_signals": True, "off_server_verified": False})
    print("G_PREPARED", value["sha256"])


def launched_training():
    """Read every E/F/G receipt; never infer completion from the old queue count."""
    found = []
    paths = list(frozen.CONTROL.glob("[EF]/train/*/launch.json")) + list((CONTROL / "train").glob("*/launch.json"))
    for path in paths:
        receipt = read(path)
        pid = receipt.get("pid")
        if process_alive(pid):
            # Detect reused PIDs before counting them or relying on their progress.
            command = Path(f"/proc/{int(pid)}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
            require("cloud_v2_campaign.py train" in command or "cloud_stage_today_20261103.py train" in command,
                    f"Training PID identity changed: {pid}")
            found.append({"pid": pid, "directory": str(path.parent), "resource": read(path.parent / "resource_state.json")})
    return found


def train_inventory(value):
    pending, completed, problems = [], [], []
    for index, parent in enumerate(value["parents"]):
        directory = directory_for("train", parent)
        launch = read(directory / "launch.json")
        result = read(directory / "result.json")
        if result.get("status") == "completed":
            require(result.get("completed_updates") == 3000 and result.get("added_updates") == 1000,
                    "G completion budget mismatch")
            require(result.get("protocol_sha256") == value["sha256"], "G result protocol mismatch")
            completed.append((index, parent, result))
        elif launch:
            if result.get("status") in ("failed", "stopped") or not process_alive(launch.get("pid")):
                problems.append({"index": index, "status": result.get("status", "missing"), "directory": str(directory),
                                 "reason": "Previously launched G work must never be automatically restarted"})
        else:
            require(not (directory / "claim.json").exists(), f"Claimed job without launch receipt requires audit: {directory}")
            require(not result, f"Unexplained result without launch receipt: {directory}")
            pending.append((index, parent))
        audit = read(directory / "training_audit.json")
        if audit.get("mechanism_violation"):
            problems.append({"index": index, "reason": "Synchronized mechanism invariant violation", "directory": str(directory)})
    return pending, completed, problems


def resource_ready():
    from src.tasks.velocity.rl.resource_control import read_gpu, available_ram_mib, read_host_memory
    sample = {**read_gpu(), "available_ram_mib": available_ram_mib(), **read_host_memory()}
    ready = sample["temperature_c"] < 75 and sample["memory_used_mib"] < 27648
    ready &= sample["available_ram_mib"] >= 12288 and sample["host_available_ram_mib"] >= 12288
    return ready, sample


def time_projection(value, active, completed):
    """Aggregate work plus a conservative slow-worker tail; estimates, not promises."""
    done = {i for i, _, _ in completed}
    remaining, progress = 0, {}
    for index, parent in enumerate(value["parents"]):
        if index in done:
            progress[f"G_{index}"] = 3000
            continue
        state = read(directory_for("train", parent) / "resource_state.json")
        updates = min(3000, max(2000, int(state.get("iteration", 1999)) + 1))
        remaining += (3000 - updates) * 24576
        progress[f"G_{index}"] = updates
    old_remaining = sum(max(0, 4000 - int(x["resource"].get("iteration", 1999)) - 1) * 24576
                        for x in active if Path(x["directory"]).is_relative_to(frozen.CONTROL / "E/train"))
    for path in (frozen.CONTROL / "E/train").glob("*/launch.json"):
        state = read(path.parent / "resource_state.json")
        progress[f"E_{path.parent.name}"] = min(4000,max(2000,int(state.get("iteration",1999))+1))
    now = time.time()
    samples = read(CONTROL / "projection_samples.json", default=[])
    candidates = [sample for sample in samples if 60 <= now - sample["epoch"] <= 300]
    pids = sorted(item["pid"] for item in active)
    warm_pids = sorted(item["pid"] for item in active
        if int(item["resource"].get("iteration",1999))+1 >= 2010
        and item["resource"].get("status") == "training")
    rates, tails = [], []
    for previous in candidates:
        # Only a stable, warm, full pool measures eight-slot capacity. A cold
        # or draining pool must not prevent the controller from filling slots.
        if not (len(pids) == len(warm_pids) == 8
                and previous.get("pids") == pids and previous.get("warm_pids") == pids):
            continue
        elapsed = now - previous["epoch"]
        delta = sum(max(0, updates - previous["progress"].get(key, updates)) for key, updates in progress.items())
        rates.append(delta * 24576 / elapsed)
    capacity_history = read(CONTROL / "capacity_history.json", default=[])
    if rates:
        capacity_history.append({"epoch":now,"steps_s":min(rates)})
        capacity_history = capacity_history[-60:]
        publish(CONTROL / "capacity_history.json",capacity_history)
    recent_capacity = [entry["steps_s"] for entry in capacity_history if now-entry["epoch"] <= 1800]
    rate = min([34529.] + recent_capacity)
    rate = max(1., rate)
    for item in active:
        matching = next((i for i,p in enumerate(value["parents"]) if str(directory_for("train",p)) == item["directory"]), None)
        if matching is None:
            continue
        key = f"G_{matching}"
        left = max(0,3000-progress[key])
        durations = []
        for previous in candidates:
            delta = progress[key] - previous["progress"].get(key,progress[key])
            if delta > 0:
                durations.append((now-previous["epoch"])/delta)
            elif previous["progress"].get(key,2000) > 2000:
                durations.append(600.)
        tails.append(left * max(durations,default=5.5))
    if not samples or now-samples[-1]["epoch"] >= 30:
        samples.append({"epoch":now,"progress":progress,"pids":pids,"warm_pids":warm_pids})
        publish(CONTROL / "projection_samples.json", samples[-20:])
    training_s = 1.25 * max((remaining + old_remaining) / rate, max(tails,default=0)) + 600
    evaluation_s = 240 * SLOW_EVAL_CASE_SECONDS * 1.25 + 600
    return {"sampled_at_utc": stamp(), "remaining_G_env_steps": remaining, "remaining_old_E_env_steps": old_remaining,
            "conservative_training_seconds": training_s, "evaluation_seconds": evaluation_s,
            "analysis_reserve_seconds": ANALYSIS_RESERVE_S, "export_reserve_seconds": EXPORT_RESERVE_S,
            "rate_cap_env_steps_s": 34529, "conservative_measured_env_steps_s":rate,
            "progress_sample_windows":len(candidates), "slowest_observed_tail_seconds":max(tails,default=0),
            "stable_full_pool_capacity_windows":len(rates), "active_workers":len(pids), "warm_workers":len(warm_pids),
            "capacity_basis":"stable_warm_eight_worker_measurement" if recent_capacity else "historical_cap_until_full_pool_warms",
            "projected_train_end": datetime.fromtimestamp(time.time() + training_s, timezone.utc).isoformat(),
            "projected_eval_end": datetime.fromtimestamp(time.time() + training_s + evaluation_s, timezone.utc).isoformat(),
            "not_a_guarantee": True}


def launch(kind, index, eval_seed=None):
    value = verify()
    safety()
    if kind == "train":
        require(len(launched_training()) < 8, "No free real E+G training slot immediately before launch")
    parent = value["parents"][index]
    directory = directory_for("train", parent) if kind == "train" else directory_for("eval", parent) / f"eval_seed{eval_seed}"
    require(not (directory / "launch.json").exists(), "Never relaunch a previously launched job")
    directory.mkdir(parents=True, exist_ok=True)
    # Claim persists across failure. No process can automatically steal or reset it.
    with (directory / "claim.json").open("x", encoding="utf-8") as claim:
        json.dump({"at_utc": stamp(), "controller_pid": os.getpid(), "kind": kind, "index": index,
                   "eval_seed": eval_seed, "protocol_sha256": value["sha256"]}, claim)
    command = [sys.executable, str(ROOT / SOURCES[0]), kind, "--index", str(index)]
    if eval_seed is not None:
        command += ["--eval-seed", str(eval_seed)]
    with (directory / "worker.log").open("x") as output:
        process = subprocess.Popen(command, cwd=ROOT, env=frozen.environment(3), stdout=output,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    publish(directory / "launch.json", {"at_utc": stamp(), "pid": process.pid, "command": command,
                                        "protocol_sha256": value["sha256"]})
    return process, directory


def checkpoint_manifest(value, completed):
    from scripts.style_resume import saved_iteration, stage_private_checkpoint
    require(len(completed) == 20, "All twenty G models must complete before evaluation")
    records = []
    for index, parent, result in completed:
        path = Path(result["final_checkpoint"])
        require(saved_iteration(path) == 2999 and sha(path) == result["final_sha256"], "Final G checkpoint identity mismatch")
        private = stage_private_checkpoint(path)
        records.append({"index": index, "group": parent["group"], "seed": parent["seed"], "parent_sha256": parent["sha256"],
                        "checkpoint": str(path), "sha256": result["final_sha256"], "private_snapshot": private["private_snapshot"],
                        "completed_updates": 3000})
    existing = read(CONTROL / "checkpoints.json")
    manifest = {"protocol_sha256": value["sha256"], "models": records}
    if existing:
        require(existing == manifest, "Immutable final checkpoint manifest changed")
    else:
        publish(CONTROL / "checkpoints.json", manifest)
    return manifest


def backup():
    value = verify()
    files = [p for p in CONTROL.rglob("*") if p.is_file() and p.suffix in
             (".json", ".jsonl", ".log", ".md", ".csv", ".png", ".pdf", ".svg", ".npz")]
    for run in (ROOT / "logs/rsl_rl/g1_dropout_style_G_today_20261103").glob("*"):
        files += list(run.glob("model_2999.pt")) + list(run.glob("params/*.yaml")) + list(run.glob("*.json"))
        files += list(run.glob("events.out.tfevents*")) + list(run.glob("resource_telemetry.csv"))
    files += [ROOT / p for p in value["sources"]] + [ROOT / p for p in frozen.verify_protocol()["sources"]]
    files += [REFERENCE, REFERENCE.with_suffix(".json"), frozen.CONTROL / "protocol.json", frozen.AUDIT]
    target = ROOT / "artifacts/cloud_stage_today_20261103_results.zip"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp.zip")
    unique = sorted({p for p in files if p.is_file()})
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in unique:
            archive.write(path, path.relative_to(ROOT))
    with zipfile.ZipFile(temporary) as archive:
        require(archive.testzip() is None, "Stage ZIP CRC verification failed")
    temporary.replace(target)
    manifest = {"at_utc": stamp(), "sha256": sha(target), "bytes": target.stat().st_size, "files": len(unique),
                "protocol_sha256": value["sha256"], "off_server_verified": False}
    publish(target.with_suffix(".sha256.json"), manifest)
    return manifest


def supervise():
    import fcntl
    CONTROL.mkdir(parents=True, exist_ok=True)
    lock = (CONTROL / "supervisor.lock").open("a+")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    value = verify()
    safety(acknowledgement=True)
    children = []
    while True:
        try:
            safety()
            pending, completed, problems = train_inventory(value)
            active = launched_training()
            require(len(active) <= 8, "Total real E+G train workers exceeds eight")
            projection = time_projection(value, active, completed)
            ready, resources = resource_ready()
            if problems:
                publish(CONTROL / "attention.json", {"at_utc": stamp(), "problems": problems, "automatic_retry": False})
                publish(CONTROL / "state.json", {"at_utc": stamp(), "status": "new_admission_held", "completed_models": len(completed),
                    "active": active, "problems": problems, "protocol_sha256": value["sha256"]})
                return 3
            if len(completed) == 20:
                if not active:
                    checkpoint_manifest(value, completed)
                    break
            now = time.time()
            fits = now < epoch(TRAIN_ADMIT_END) and now + projection["conservative_training_seconds"] + projection["evaluation_seconds"] < epoch(EVAL_END)
            fits &= now + projection["conservative_training_seconds"] + projection["evaluation_seconds"] + ANALYSIS_RESERVE_S + EXPORT_RESERVE_S < epoch(EXPORT_END)
            if pending and len(active) < 8 and ready and fits:
                safety()
                index, _ = pending[0]
                children.append(launch("train", index))
            for process, _ in children:
                process.poll()
            publish(CONTROL / "state.json", {"at_utc": stamp(), "status": "training" if fits else "new_admission_held_time",
                "supervisor_pid": os.getpid(), "branch": "G", "completed_models": len(completed), "planned_models": 20,
                "unlaunched_models": len(pending), "active_E_G": active, "projection": projection, "resources": resources,
                "protocol_sha256": value["sha256"], "frozen_E_F_gate": False})
            if pending and not fits and not active:
                return 4
            time.sleep(5)
        except Exception as error:
            publish(CONTROL / "attention.json", {"at_utc": stamp(), "error": f"{type(error).__name__}: {error}",
                "existing_training_left_running": True, "automatic_retry": False})
            return 3
    backup()
    try:
        return evaluate_all(value)
    except Exception as error:
        publish(CONTROL / "attention.json", {"at_utc": stamp(), "phase": "evaluation",
            "error": f"{type(error).__name__}: {error}", "existing_workers_left_running": True,
            "automatic_retry": False})
        return 3


def evaluate_all(value):
    """Balanced 80-case blocks, one common evaluation seed per block."""
    children = []
    for eval_seed in value["evaluation"]["evaluation_seeds"]:
        previous_reports = [read(p) for p in (CONTROL / "eval").glob("*/eval_seed*/evaluation.json")]
        completed_before_block = sum(len(r.get("records", [])) for r in previous_reports)
        block_reserve = (240-completed_before_block) * SLOW_EVAL_CASE_SECONDS * 1.25 + 600
        require(time.time()+block_reserve < epoch(EVAL_END)
            and time.time()+block_reserve+ANALYSIS_RESERVE_S+EXPORT_RESERVE_S < epoch(EXPORT_END),
            "Cannot admit the complete next 80-case block plus all remaining evaluation/analysis/export")
        publish(CONTROL / f"eval_seed{eval_seed}_block_admission.json", {"at_utc":stamp(), "block_cases":80,
            "remaining_total_cases":240-completed_before_block, "evaluation_reserve_seconds":block_reserve,
            "analysis_reserve_seconds":ANALYSIS_RESERVE_S, "export_reserve_seconds":EXPORT_RESERVE_S})
        pending = list(range(20))
        while pending or children:
            safety()
            require(not launched_training(), "Wait for all existing training processes to exit")
            for item in list(children):
                process, directory = item
                if process.poll() is not None:
                    children.remove(item)
                    require(process.returncode == 0 and read(directory / "evaluation.json").get("status") == "completed",
                            f"Evaluation failed; no automatic retry: {directory}")
            ready, resources = resource_ready()
            reports = [read(p) for p in (CONTROL / "eval").glob("*/eval_seed*/evaluation.json")]
            records = [r for report in reports for r in report.get("records", [])]
            remaining = 240 - len(records)
            remaining_s = 1.25 * max(remaining * SLOW_EVAL_CASE_SECONDS, min(4, remaining) * 100) + 600
            fits = time.time() + remaining_s < epoch(EVAL_END) and time.time() + remaining_s + ANALYSIS_RESERVE_S + EXPORT_RESERVE_S < epoch(EXPORT_END)
            if pending and len(children) < 8 and ready and fits:
                children.append(launch("eval", pending.pop(0), eval_seed))
            publish(CONTROL / "state.json", {"at_utc": stamp(), "status": "evaluating", "supervisor_pid": os.getpid(),
                "evaluation_seed_block": eval_seed, "completed_cases": len(records), "planned_cases": 240,
                "active": [{"pid": p.pid, "directory": str(d)} for p, d in children], "resources": resources,
                "protocol_sha256": value["sha256"], "remaining_evaluation_seconds_estimate": remaining_s,
                "complete_evaluation_and_export_fit": fits})
            if not fits and pending and not children:
                raise RuntimeError("Complete evaluation/export no longer fits today; pending cases remain unlaunched")
            time.sleep(5)
    reports = [read(p) for p in (CONTROL / "eval").glob("*/eval_seed*/evaluation.json")]
    records = [r for report in reports for r in report.get("records", [])]
    keys = {(r["group"], r["train_seed"], r["evaluation_seed"], r["scenario"]) for r in records}
    expected = {(g, s, e, c) for g in "ABCD" for s in range(42, 47) for e in EVAL_SEEDS for c in SCENARIOS}
    require(keys == expected and len(records) == 240 and all(r.get("status") == "completed" for r in reports), "240-case matrix is incomplete or duplicated")
    publish(CONTROL / "evaluation.json", {"status": "completed", "at_utc": stamp(), "protocol_sha256": value["sha256"], "records": records,
        "completed_cases": 240, "planned_cases": 240, "off_server_verified": False})
    publish(CONTROL / "state.json", {"status": "completed", "at_utc": stamp(), "completed_models": 20,
        "completed_cases": 240, "protocol_sha256": value["sha256"], "off_server_verified": False})
    backup()
    return 0


def validate(runtime_checks=False):
    """Standard-library software checks only; no torch import or GPU allocation."""
    import ast
    import math
    for source in SOURCES:
        ast.parse((ROOT / source).read_text(encoding="utf-8"))
    require(len({(g, s) for g in "ABCD" for s in range(42, 47)}) == 20, "Parent count")
    require(20 * 1000 * 1024 * 24 == 491520000, "Equal training budget")
    require(20 * len(SCENARIOS) * len(EVAL_SEEDS) == 240, "Evaluation matrix")
    require(epoch(TRAIN_ADMIT_END) < epoch(EVAL_END) < epoch(EXPORT_END), "Today deadlines")
    def bounds(step):
        fraction = min(1., max(0., (step - 48000) / 12000))
        return .75 + .25 * fraction, 20 + round(20 * fraction), .25 * fraction
    require(bounds(48000) == (.75, 20, 0.) and bounds(60000) == (1., 40, .25), "F-style extension endpoints")
    require(bounds(72000) == (1., 40, .25), "Last 500 updates remain at full distribution")
    trace = 0.
    for _ in range(20):
        trace = math.exp(-.02/.08) * trace + 1 - math.exp(-.02/.08)
    low = 1 - .75 * trace
    for _ in range(70):
        trace *= math.exp(-.02/.25)
    require(low < .26 and 1 - .75 * trace > .99, "Trace endpoints")
    print(json.dumps({"status": "passed", "cpu_only": True, "torch_imported": "torch" in sys.modules,
        "training_env_steps": 491520000, "evaluation_cases": 240, "no_gpu_training_or_simulation": True}))
    if runtime_checks:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        from scripts.cloud_stage_today_runtime_20261103 import validate_cpu
        validate_cpu()
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "validate", "supervise", "train", "eval", "status", "backup"))
    parser.add_argument("--index", type=int)
    parser.add_argument("--eval-seed", type=int, choices=EVAL_SEEDS)
    parser.add_argument("--runtime-checks", action="store_true", help="CPU-only dependency/configuration and hook checks, no simulator")
    args = parser.parse_args()
    if args.action == "validate":
        return validate(args.runtime_checks)
    if args.action == "prepare":
        return prepare()
    if args.action == "status":
        print(json.dumps(read(CONTROL / "state.json"), ensure_ascii=False, indent=2))
        return 0
    if args.action == "backup":
        print(json.dumps(backup()))
        return 0
    if args.action == "supervise":
        return supervise()
    require(args.index is not None and 0 <= args.index < 20, "Worker requires index 0..19")
    value = verify()
    safety()
    from scripts import cloud_stage_today_runtime_20261103 as runtime
    if args.action == "train":
        return runtime.train_worker(value, "G", args.index)
    require(args.eval_seed is not None, "Evaluation worker requires paired evaluation seed")
    return runtime.evaluate_worker(value, args.index, args.eval_seed)


if __name__ == "__main__":
    raise SystemExit(main())
