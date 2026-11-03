"""Separate future-admission governance for the immutable G experiment.

Only the supervisor process is wrapped. Worker commands, training source,
configuration, checkpoints and already running processes are unchanged.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import cloud_stage_today_20261103 as stage

GATE = stage.CONTROL / "plan_gate.json"
SUPPORT = stage.CONTROL / "admission_protocol.json"
LOCK = stage.CONTROL / "admission.lock"


def verdict(gate, protocol_sha256, now):
    if gate.get("admit_new_jobs") is not True:
        return False, "Gate missing, invalid, or explicitly closed"
    if gate.get("training_protocol_sha256") != protocol_sha256:
        return False, "Gate identifies another training protocol"
    try:
        valid_until = stage.epoch(gate["valid_until_utc"])
        sampled = stage.epoch(gate["audit_sampled_at_utc"])
    except (KeyError, TypeError, ValueError):
        return False, "Gate has no valid audit/expiry timestamps"
    if sampled > now + 30 or valid_until <= now or valid_until - sampled > 45 * 60:
        return False, "Audit is future dated, expired, or exceeds the 45-minute validity limit"
    return True, "Fresh explicit audit admission; all original resource/time limits still apply"


def verify_support():
    value = stage.verify()
    support = stage.read(SUPPORT)
    stage.require(support.get("training_protocol_sha256") == value["sha256"], "Admission protocol mismatch")
    stage.require(support.get("wrapper_sha256") == stage.sha(Path(__file__)), "Admission source drift")
    stage.require(support.get("training_sources") == value["sources"], "Training source identity mismatch")
    stage.require(support.get("cpu_validation_sha256") == stage.sha(stage.CONTROL / "software_validation.json"),
                  "CPU validation identity changed")
    stage.require(support.get("initial_admission_sha256") == stage.sha(stage.CONTROL / "initial_admission_receipt.json"),
                  "Initial admission identity changed")
    return value, support


def admission_state(value):
    gate = stage.read(GATE)
    allowed, reason = verdict(gate, value["sha256"], time.time())
    state = {"at_utc": stage.stamp(), "admit_new_jobs_effective": allowed, "reason": reason,
             "scope": "new G training AND new formal evaluation workers only",
             "active_workers_untouched": True, "training_protocol_sha256": value["sha256"],
             "gate": gate, "wrapper_sha256": stage.sha(Path(__file__))}
    stage.publish(stage.CONTROL / "admission_state.json", state)
    return state


def set_gate(value):
    import fcntl
    verify_support()
    stage.require(type(value.get("admit_new_jobs")) is bool, "Gate requires a boolean admission decision")
    stage.require(value.get("training_protocol_sha256") == stage.verify()["sha256"], "Wrong gate protocol")
    if value["admit_new_jobs"]:
        allowed, reason = verdict(value, value["training_protocol_sha256"], time.time())
        stage.require(allowed, reason)
        stage.safety(acknowledgement=True)
    with LOCK.open("a+") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        stage.publish(GATE, value)
    state = admission_state(stage.verify())
    print(json.dumps(state, ensure_ascii=False))
    return 0


def supervise():
    import fcntl
    stage.require(os.name == "posix", "Supervisor runs only on the rented Linux host")
    value, support = verify_support()
    original_ready, original_launch = stage.resource_ready, stage.launch

    def ready():
        resource_ok, sample = original_ready()
        state = admission_state(value)
        sample["future_admission"] = state
        return resource_ok and state["admit_new_jobs_effective"], sample

    def launch(kind, index, eval_seed=None):
        # Gate updates use this same lock. A closure is effective after any
        # already authorized claim/Popen transaction finishes, never midway.
        with LOCK.open("a+") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            state = admission_state(value)
            stage.require(state["admit_new_jobs_effective"],
                          "Future admission closed before claim; supervisor exits safely, existing workers continue")
            return original_launch(kind, index, eval_seed)

    stage.resource_ready, stage.launch = ready, launch
    try:
        result = stage.supervise()
    finally:
        stage.resource_ready, stage.launch = original_ready, original_launch
    state = admission_state(value)
    state.update(supervisor_exit_code=result, supervisor_pid=os.getpid(), automatic_worker_retry=False)
    stage.publish(stage.CONTROL / "admission_state.json", state)
    return result


def validate():
    now = 1_000_000.
    timestamp = lambda epoch: datetime.fromtimestamp(epoch, timezone.utc).isoformat()
    gate = {"admit_new_jobs": True, "training_protocol_sha256": "isolated_test",
            "audit_sampled_at_utc": timestamp(now), "valid_until_utc": timestamp(now + 2400)}
    stage.require(verdict(gate, "isolated_test", now)[0], "Valid gate rejected")
    invalid = [{}, {**gate, "admit_new_jobs": False}, {**gate, "admit_new_jobs": "true"},
               {**gate, "training_protocol_sha256": "other"}, {**gate, "valid_until_utc": timestamp(now)},
               {**gate, "valid_until_utc": "invalid"}, {**gate, "valid_until_utc": timestamp(now + 3000)},
               {**gate, "audit_sampled_at_utc": timestamp(now + 100)}]
    stage.require(all(not verdict(item, "isolated_test", now)[0] for item in invalid), "Fail-closed gate checks")
    print(json.dumps({"status": "passed", "checks": 9, "cpu_only": True,
                      "no_training_simulation_or_gpu": True, "torch_imported": "torch" in __import__("sys").modules}))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("validate", "supervise", "set-gate"))
    parser.add_argument("--gate-json")
    args = parser.parse_args()
    if args.action == "validate":
        return validate()
    if args.action == "set-gate":
        stage.require(args.gate_json is not None, "Provide the complete reviewed gate decision")
        return set_gate(json.loads(args.gate_json))
    return supervise()


if __name__ == "__main__":
    raise SystemExit(main())
