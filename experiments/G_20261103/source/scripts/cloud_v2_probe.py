"""One bounded midcourse diagnostic for the isolated v2 robot-control study.

Only this invocation's evaluator child may be terminated. Training processes,
training stop requests, study budgets, and formal evaluation files are read-only.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.cloud_v2_campaign import (  # noqa: E402
    CONTROL, REFERENCE, REFERENCE_SHA, TRAIN_END, EXPORT_END, canonical_sha,
    directory_for, environment, epoch, manual_stop_paths, process_state,
    publish, read, sha, stamp,
)

PROBES = CONTROL / "probes"
WALL_BUDGET_S = 120.0
SCENARIOS = ("nominal", "left_knee_75pct_0p4s", "right_ankle_100pct_0p8s")
PURPOSE = "EXPLORATORY_32ENV_10S_MIDCOURSE_DIAGNOSTIC_NOT_FORMAL_EVALUATION"
SCOPE = ("A single 32-environment, 10-second diagnostic; completed update budgets may differ. "
         "Do not rank methods against 128-environment, 20-second formal evaluation or unmatched checkpoints. "
         "Read survival, first-life errors, recovery eligibility, and valid frame counts together.")


def formal_evaluation_started():
    state = read(CONTROL / "state.json")
    if state.get("status") == "evaluating":
        return True
    # A past formal start still blocks further probes, including during startup,
    # cooling, failure handling, or after formal evaluation has completed.
    return any(CONTROL.glob("[EF]/eval/*/launch.json"))


def resource_sample():
    from scripts.cloud_training_monitor import resources
    return resources()


def training_snapshot():
    """Read progress and /proc state; never send even a zero-valued signal."""
    models = []
    throughput = []
    for state_path in sorted(CONTROL.glob("[EF]/train/*/resource_state.json")):
        value = read(state_path)
        pid = value.get("pid")
        proc = process_state(pid)
        item = {"branch": state_path.parents[2].name,
                **{key: value.get(key) for key in ("group", "seed", "pid", "iteration", "status", "updated_at_utc", "log_directory")},
                "process_state": proc}
        models.append(item)
        if proc in (None, "Z", "X") or value.get("status") != "training":
            continue
        telemetry = Path(value.get("log_directory", "/nonexistent")) / "resource_telemetry.csv"
        if telemetry.is_file():
            try:
                with telemetry.open() as stream:
                    rows = list(csv.DictReader(stream))[-30:]
                if len(rows) >= 2:
                    wall = epoch(rows[-1]["updated_at_utc"]) - epoch(rows[0]["updated_at_utc"])
                    updates = int(rows[-1]["iteration"]) - int(rows[0]["iteration"])
                    if wall > 0 and updates > 0:
                        throughput.append(updates * 1024 * 24 / wall)
            except (OSError, ValueError, KeyError):
                pass
    return {"at_utc": stamp(), "campaign_state": read(CONTROL / "state.json"),
            "models": models, "aggregate_recent_env_steps_s": sum(throughput) if throughput else None}


def admission(snapshot, previous, resources):
    if manual_stop_paths():
        return "manual_stop_present"
    if time.time() + WALL_BUDGET_S >= min(epoch(TRAIN_END), epoch(EXPORT_END)):
        return "insufficient_time_before_training_or_export_deadline"
    if formal_evaluation_started():
        return "formal_evaluation_has_started"
    state = snapshot["campaign_state"]
    if state.get("status") not in ("training", "new_admission_held") or state.get("failure"):
        return "training_inactive_or_unresolved_failure"
    if not any(m["status"] == "training" and m["process_state"] not in (None, "Z", "X") for m in snapshot["models"]):
        return "no_live_training_worker"
    if resources.get("temperature_c", 100) >= 70 or resources.get("memory_used_mib", 99999) > 24576:
        return "gpu_headroom"
    if resources.get("available_ram_mib", 0) < 16384 or resources.get("free_disk_gib", 0) < 8:
        return "host_headroom"
    slowdown = previous.get("sampled_training_slowdown_fraction")
    if slowdown is not None and slowdown > .10:
        return "previous_probe_estimated_slowdown_over_10pct_skip_one_round"
    return None


def select_checkpoint(protocol, snapshot, sequence, scenario):
    """Prefer a fresh saved milestone, rotating groups without selecting on reward."""
    import yaml
    from scripts.style_resume import saved_iteration
    desired_group = "DBCA"[sequence % 4]
    prior_keys = {
        (report.get("job", {}).get("source_checkpoint"), report.get("job", {}).get("scenario"))
        for file in PROBES.glob("*/report.json")
        if (report := read(file)).get("status") == "completed"
    }
    active = {(m["branch"], m["group"], int(m["seed"])) for m in snapshot["models"]
              if m.get("seed") is not None and m["process_state"] not in (None, "Z", "X")}
    candidates = []
    for branch in "EF":
        for parent in protocol.get("parents", []):
            directory = directory_for(branch, "train", parent)
            resource = read(directory / "resource_state.json")
            run = Path(resource.get("log_directory", "/nonexistent"))
            if not run.is_dir():
                continue
            try:
                env = yaml.load((run / "params/env.yaml").read_text(), Loader=yaml.BaseLoader)
                agent = yaml.load((run / "params/agent.yaml").read_text(), Loader=yaml.BaseLoader)
                provenance = read(run / "reference_provenance.json")
                if (int(agent["seed"]) != parent["seed"] or int(agent["max_iterations"]) != 4000
                        or int(agent["num_steps_per_env"]) != 24 or int(env["scene"]["num_envs"]) != 1024
                        or env["commands"]["gait_reference"]["mode"] != {"A": "none", "B": "fixed", "C": "global", "D": "selective"}[parent["group"]]
                        or provenance.get("sha256") != REFERENCE_SHA):
                    continue
                for path in run.glob("model_*.pt"):
                    file_iteration = int(path.stem.split("_")[-1])
                    updates = file_iteration + 1
                    if not 2400 <= updates <= 3601 or time.time() - path.stat().st_mtime < 20:
                        continue
                    if (str(path), scenario) in prior_keys:
                        continue
                    distance = min(abs(updates - milestone) for milestone in (2500, 3000, 3500))
                    if distance > 101:
                        continue
                    if saved_iteration(path) != file_iteration:
                        continue
                    rank = (parent["group"] != desired_group,
                            (branch, parent["group"], parent["seed"]) not in active,
                            -updates, distance, parent["seed"], branch)
                    candidates.append((rank, {"branch": branch, "group": parent["group"], "training_seed": parent["seed"],
                            "source_checkpoint": str(path), "completed_updates": updates,
                            "nearest_planned_milestone": min((2500, 3000, 3500), key=lambda x: abs(updates - x)),
                            "scenario": scenario, "protocol_sha256": protocol["sha256"]}))
            except (OSError, ValueError, KeyError, StopIteration):
                continue
    if not candidates:
        return None
    job = min(candidates, key=lambda pair: pair[0])[1]
    job["checkpoint_sha256"] = sha(Path(job["source_checkpoint"]))
    return job


def stop_owned_probe(process, deadline):
    """Only accepts the Popen object created by this invocation, never a train PID."""
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=min(4., max(.1, deadline - time.monotonic())))
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=max(.1, deadline - time.monotonic()))


def runtime_reason():
    if manual_stop_paths():
        return "manual_stop_present"
    if time.time() >= min(epoch(TRAIN_END), epoch(EXPORT_END)):
        return "training_or_export_deadline"
    if formal_evaluation_started():
        return "formal_evaluation_started_during_probe"
    state = read(CONTROL / "state.json")
    if state.get("status") not in ("training", "new_admission_held") or state.get("failure"):
        return "training_phase_ended_or_failed"
    sample = resource_sample()
    if sample["temperature_c"] >= 70 or sample["memory_used_mib"] > 26624 or sample["available_ram_mib"] < 12288:
        return "probe_resource_exit"
    return None


def once():
    import fcntl
    PROBES.mkdir(parents=True, exist_ok=True)
    with (PROBES / "probe.lock").open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("V2_PROBE_DEFERRED another probe owns the lock; no duplicate process started")
            return 0
        previous = read(PROBES / "latest.json")
        sequence = int(previous.get("sequence", 0))
        report = {"started_at_utc": stamp(), "sequence": sequence, "purpose": PURPOSE,
                  "scope": SCOPE, "budget": {"num_envs": 32, "simulation_seconds": 10., "cpu_threads": 1, "wall_limit_s": WALL_BUDGET_S},
                  "status": "checking", "training_before": training_snapshot()}
        directory = PROBES / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        directory.mkdir()
        try:
            protocol = read(CONTROL / "protocol.json")
            if not protocol or protocol.get("reference_sha256") != REFERENCE_SHA:
                raise RuntimeError("Frozen v2 protocol/reference identity missing")
            if canonical_sha({k: v for k, v in protocol.items() if k != "sha256"}) != protocol.get("sha256"):
                raise RuntimeError("Frozen v2 protocol digest differs")
            sample = resource_sample()
            report["resources_before"] = sample
            reason = admission(report["training_before"], previous, sample)
            job = None if reason else select_checkpoint(protocol, report["training_before"], sequence, SCENARIOS[sequence % len(SCENARIOS)])
            if reason or job is None:
                report.update(status="deferred", reason=reason or "no_new_stable_compatible_midcourse_checkpoint")
                return_code = 0
            else:
                report["job"] = job
                publish(directory / "job.json", job)
                env = environment(1)
                started = time.monotonic()
                deadline = started + WALL_BUDGET_S
                process = None
                reason = None
                try:
                    with (directory / "probe.log").open("w") as output:
                        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker", "--directory", str(directory)],
                            cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT)
                        report["probe_pid"] = process.pid
                        while process.poll() is None:
                            # Reserve eight seconds of the 120-second envelope
                            # for stopping this diagnostic child if necessary.
                            if time.monotonic() >= deadline - 8:
                                reason = "120_second_wall_envelope_with_cleanup_reserve"
                            else:
                                try:
                                    reason = runtime_reason()
                                except Exception as ex:
                                    reason = f"resource_or_phase_inspection_failed: {type(ex).__name__}"
                            if reason:
                                stop_owned_probe(process, deadline)
                                break
                            time.sleep(min(2., max(.1, deadline - 8 - time.monotonic())))
                finally:
                    if process is not None and process.poll() is None:
                        stop_owned_probe(process, deadline)
                worker = read(directory / "results.json")
                records = worker.get("records", [])
                report.update(status="completed" if process.returncode == 0 and len(records) == 1 else "incomplete",
                              reason=reason or worker.get("reason"), return_code=process.returncode,
                              wall_seconds=time.monotonic() - started, records=records, worker=worker)
                report["sequence"] = sequence + 1
                return_code = 0 if report["status"] == "completed" else 2
        except Exception as ex:
            report.update(status="failed", reason=f"{type(ex).__name__}: {ex}")
            return_code = 2
        finally:
            report["training_after"] = training_snapshot()
            before = report["training_before"].get("aggregate_recent_env_steps_s")
            after = report["training_after"].get("aggregate_recent_env_steps_s")
            report["sampled_training_slowdown_fraction"] = max(0., 1. - after / before) if before and after and report["status"] != "deferred" else None
            report["impact_scope"] = "Recent telemetry estimate; completion, initialization, and normal fluctuation confound causality."
            report["completed_at_utc"] = stamp()
            report["directory"] = str(directory)
            publish(directory / "report.json", report)
            publish(PROBES / "latest.json", report)
            publish(ROOT / "artifacts/cloud_v2_latest_probe.txt", report)
            print("V2_PROBE_REPORT_READY", ROOT / "artifacts/cloud_v2_latest_probe.txt", report["status"], report.get("reason"))
        return return_code


def worker(directory):
    directory = directory.resolve(strict=True)
    if not directory.is_relative_to(PROBES.resolve()):
        raise ValueError("Probe worker directory must remain inside results/cloud_v2/probes")
    os.nice(10)
    import torch
    from scripts import evaluate_style_dropout as evaluator
    from scripts import style_evaluation_metrics as metrics
    from scripts.style_evaluation_protocol import scenarios
    from scripts.style_resume import stage_private_checkpoint, saved_iteration
    from src.tasks.velocity.rl.resource_control import ResourceLimits, TrainingStopRequested
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    job = read(directory / "job.json")
    report = {"purpose": PURPOSE, "scope": SCOPE, "started_at_utc": stamp(), "records": []}
    try:
        if sha(REFERENCE) != REFERENCE_SHA:
            raise ValueError("Frozen reference changed")
        source = Path(job["source_checkpoint"])
        if sha(source) != job["checkpoint_sha256"] or saved_iteration(source) + 1 != job["completed_updates"]:
            raise ValueError("Checkpoint identity/update metadata changed")
        snapshot = stage_private_checkpoint(source)
        limits = ResourceLimits(cpu_threads=1, duty_cycle=1, minimum_rest_s=0,
                                maximum_gpu_memory_mib=26624, minimum_available_ram_mib=12288,
                                minimum_host_available_ram_mib=12288, minimum_start_host_available_ram_mib=16384)
        original_budget = evaluator.check_budget

        def guarded_budget(resource_limits):
            reason = runtime_reason()
            if reason:
                raise TrainingStopRequested(reason, category="probe_admission")
            return original_budget(resource_limits)

        evaluator.check_budget = guarded_budget
        original_rmse = metrics.alive_rmse
        counts = []

        def counted_rmse(values, alive, start, end):
            selected = values[start:end][alive[start:end]]
            counts.append({"frame_environment_samples": selected.numel(), "finite_frame_environment_samples": int(torch.isfinite(selected).sum()),
                           "window_start": start, "window_end": min(end, len(values))})
            return original_rmse(values, alive, start, end)

        metrics.alive_rmse = counted_rmse
        scenario = job["scenario"]
        record = evaluator.evaluate_one(job["group"], Path(snapshot["private_snapshot"]), scenario, 7042, REFERENCE, 32, limits,
            training_seed=job["training_seed"], control_directory=directory / "control", scenario_config=scenarios()[scenario], simulation_seconds=10.)
        labels = ["first_life_xy", "first_life_reference", "fault_xy", "fault_reference", "fault_unaffected_reference", "post_fault_reference"]
        if scenario == "nominal":
            labels = labels[:2]
        if len(counts) != len(labels):
            raise RuntimeError("Valid-sample metric hook contract changed")
        record.update(purpose=PURPOSE, scope=SCOPE, branch=job["branch"], completed_training_updates=job["completed_updates"],
                      checkpoint_sha256=job["checkpoint_sha256"], protocol_sha256=job["protocol_sha256"],
                      valid_samples=dict(zip(labels, counts)), metric_schema_version=3,
                      post_fault_observation_horizon_s=10. - 6. - scenarios()[scenario]["duration_s"] if scenario != "nominal" else None)
        report.update(status="completed", records=[record])
        return_code = 0
    except TrainingStopRequested as ex:
        report.update(status="stopped", reason=str(ex), stop_category=ex.category)
        return_code = 3
    except Exception as ex:
        report.update(status="failed", reason=f"{type(ex).__name__}: {ex}")
        return_code = 2
    finally:
        report["completed_at_utc"] = stamp()
        publish(directory / "results.json", report)
    return return_code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--once", action="store_true")
    actions.add_argument("--worker", action="store_true")
    parser.add_argument("--directory", type=Path)
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("Cloud-only diagnostic: local workstation execution is disabled")
    if args.worker and args.directory is None:
        parser.error("--worker requires --directory")
    os.chdir(ROOT)
    return worker(args.directory) if args.worker else once()


if __name__ == "__main__":
    raise SystemExit(main())
