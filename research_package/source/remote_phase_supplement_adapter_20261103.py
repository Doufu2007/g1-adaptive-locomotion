"""Linux execution adapter for the sealed rough-phase supplement.

This file only translates the Windows paths recorded by the frozen protocol,
selects the remote output/control roots, and supplies the CUDA/MuJoCo bootstrap.
It does not change scientific constants, seeds, budgets, models, or metrics.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from scripts import g1_phase_supplement_20261031 as supplement
from scripts import g1_phase_lr_factorial_20261027 as anchor

CONTROL = ROOT / "results/g1_phase_supplement_20261031/formal_recovery_01"
OUT = ROOT / "remote_output/formal_recovery_01"
SELF = Path(__file__).resolve()


def translate_path(value):
    """Map a sealed Windows path to the corresponding remote project path."""
    path = Path(value)
    if path.exists():
        return path
    text = str(value).replace("\\", "/")
    marker = "/phasor/"
    index = text.lower().rfind(marker)
    if index >= 0:
        return ROOT / text[index + len(marker):]
    return path


_native_sha = anchor.sha


def remote_sha(value):
    return _native_sha(translate_path(value))


def bootstrap():
    """Use the existing POSIX bootstrap used by the earlier Linux runner."""
    from scripts.run_g1_stateful_remote_20261025 import bootstrap as native
    return native()


def install_runtime_hooks():
    # g1_phase_feedback_probe imports this symbol inside setup(); replace the
    # Windows-only bootstrap before setup is called.
    from scripts import run_g1_cause_audit_windows_20261026 as windows_bootstrap
    windows_bootstrap.bootstrap = bootstrap

    anchor.sha = remote_sha
    supplement.a.sha = remote_sha

    supplement.ROOT = ROOT
    supplement.OUT = OUT
    supplement.CONTROL = CONTROL
    supplement.SPEC = ROOT / "docs/superpowers/specs/2026-10-31-rough-phase-supplement-design.md"
    supplement.PLAN = ROOT / "docs/superpowers/plans/2026-10-31-rough-phase-supplement.md"

    def verify_seal_remote():
        value = anchor.read(CONTROL / "protocol.json")
        for entry in value["files"]:
            if remote_sha(entry["path"]) != entry["sha256"]:
                raise ValueError("sealed input changed: " + entry["path"])
        return value

    supplement.verify_seal = verify_seal_remote

    def check_disk_remote():
        import shutil
        OUT.mkdir(parents=True, exist_ok=True)
        for path, minimum in ((ROOT, 10), (OUT, 15)):
            free = shutil.disk_usage(path).free
            if free < minimum * 1024 ** 3:
                raise ValueError(f"{path}: below locked {minimum}GiB free disk floor")

    supplement.check_disk = check_disk_remote

    def child_args(command, **kwargs):
        args = [sys.executable, "-X", "utf8", str(SELF), command]
        for key, value in kwargs.items():
            if value is True:
                args.append("--" + key.replace("_", "-"))
            elif value is not None and value is not False:
                args.extend(["--" + key.replace("_", "-"), str(value)])
        return args

    supplement.child_args = child_args

    def failure_evidence_remote(error):
        import psutil
        import shutil
        value = dict(
            at=anchor.now(),
            error=repr(error),
            traceback=traceback.format_exc(),
            memory=psutil.virtual_memory()._asdict(),
            disk={str(path): shutil.disk_usage(path)._asdict() for path in (ROOT, OUT)},
            processes=[],
        )
        for proc in psutil.process_iter(["pid", "name", "cmdline", "cpu_times", "memory_info"]):
            if "python" in (proc.info["name"] or "").lower():
                info = proc.info.copy()
                if info["cpu_times"] is not None:
                    info["cpu_times"] = info["cpu_times"]._asdict()
                if info["memory_info"] is not None:
                    info["memory_info"] = info["memory_info"]._asdict()
                value["processes"].append(info)
        try:
            value["gpu"] = subprocess.run(
                ["nvidia-smi"], capture_output=True, text=True, timeout=20
            ).stdout
        except Exception as exc:
            value["gpu_error"] = repr(exc)
        CONTROL.mkdir(parents=True, exist_ok=True)
        anchor.write(CONTROL / f"failure_evidence_remote_pid{os.getpid()}.json", value)
        anchor.write(CONTROL / "failure_evidence_remote.json", value)
        supplement.cost_ledger()

    supplement.failure_evidence = failure_evidence_remote
    CONTROL.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)


def main():
    install_runtime_hooks()
    if len(sys.argv) < 2:
        raise SystemExit("usage: remote_phase_supplement_adapter_20261103.py COMMAND [ARGS]")
    supplement.main()


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        if CONTROL.exists():
            try:
                supplement.failure_evidence(error)
            except Exception:
                traceback.print_exc()
        raise
