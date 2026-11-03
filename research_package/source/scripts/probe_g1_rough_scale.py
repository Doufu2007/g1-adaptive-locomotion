"""Measure whether rough-terrain PPO fits on this GPU and how fast it steps.

Option B is an equal-budget rough-terrain comparison.  The historical rough
runs used ``num_envs=4096`` on a cloud GPU (``scripts/reproduce.sh`` states it
needs >= 32 GB of VRAM).  This machine has an RTX 4050 Laptop.  Before any
budget can be registered, the achievable scale has to be *measured*, not
guessed: a step/VRAM reading per ``num_envs`` for both arms.

Each configuration runs in its own process so that a CUDA OOM in one does not
poison the next.  The parent writes one JSON record per attempt.

Usage:
    python scripts/probe_g1_rough_scale.py --driver          # sweep
    python scripts/probe_g1_rough_scale.py --one <task> <n>  # single config
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PILOT_HARNESS = ROOT / "results" / "phasor_pilot_20261016" / "run_pilot.py"
OUT_PATH = ROOT / "results" / "g1_phase_gate_v4_2" / "rough_scale_probe.json"

TASKS = ("Unitree-G1-Rough", "Unitree-G1-Rough-PHASOR")
ENV_COUNTS = (128, 256, 512, 1024, 2048)
WARMUP_STEPS = 3
TIMED_STEPS = 20


def _load_harness():
    """Import the pilot harness so the ASCII path bootstrap has one definition."""
    spec = importlib.util.spec_from_file_location("_phasor_pilot_harness", PILOT_HARNESS)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _gpu_used_mib() -> float | None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return float(out.strip().splitlines()[0])
    except (OSError, subprocess.CalledProcessError, ValueError, IndexError):
        return None


def measure_one(task: str, num_envs: int) -> dict:
    harness = _load_harness()
    harness.bootstrap_runtime()
    try:
        import numpy as np  # noqa: F401
        import torch

        import mjlab.tasks  # noqa: F401
        import src.tasks  # noqa: F401
        from mjlab.envs import ManagerBasedRlEnv
        from mjlab.tasks.registry import load_env_cfg

        cfg = load_env_cfg(task)
        cfg.scene.num_envs = num_envs

        baseline_used = _gpu_used_mib()
        env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
        try:
            action_dim = int(env.action_manager.total_action_dim)
            actions = torch.zeros(num_envs, action_dim, device="cuda:0")
            # ``extras["log"]`` is only created by ``_reset_idx``; stepping a
            # freshly constructed env hits reward terms that write to it.
            env.reset()
            for _ in range(WARMUP_STEPS):
                env.step(actions)
            torch.cuda.synchronize()
            started = time.perf_counter()
            for _ in range(TIMED_STEPS):
                env.step(actions)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            peak_used = _gpu_used_mib()
            allocated = torch.cuda.max_memory_allocated() / (1024**2)
        finally:
            env.close()
            torch.cuda.empty_cache()

        seconds_per_step = elapsed / TIMED_STEPS
        return {
            "task": task,
            "num_envs": num_envs,
            "status": "ok",
            "action_dim": action_dim,
            "seconds_per_env_step": seconds_per_step,
            "env_steps_per_second": num_envs / seconds_per_step,
            "gpu_used_before_mib": baseline_used,
            "gpu_used_after_mib": peak_used,
            "torch_peak_allocated_mib": allocated,
        }
    except Exception as error:  # noqa: BLE001 - report, do not mask
        return {
            "task": task,
            "num_envs": num_envs,
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error)[:800],
        }
    finally:
        harness._remove_dos_devices()


def driver() -> int:
    records: list[dict] = []
    for task in TASKS:
        for num_envs in ENV_COUNTS:
            started = time.perf_counter()
            completed = subprocess.run(
                [sys.executable, str(Path(__file__).relative_to(ROOT)), "--one", task, str(num_envs)],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                # The child prints UTF-8 tables; the default locale codec here
                # is GBK and would raise instead of decoding them.
                encoding="utf-8",
                errors="replace",
                timeout=1800,
            )
            wall = time.perf_counter() - started
            payload = None
            for line in reversed(completed.stdout.splitlines()):
                if line.startswith("RESULT "):
                    payload = json.loads(line[len("RESULT "):])
                    break
            if payload is None:
                payload = {
                    "task": task,
                    "num_envs": num_envs,
                    "status": "failed",
                    "error_type": "no_result",
                    "error": (completed.stderr or completed.stdout)[-800:],
                }
            payload["wall_seconds"] = wall
            records.append(payload)
            print(json.dumps(payload, ensure_ascii=False), flush=True)
            if payload["status"] != "ok":
                break  # Larger counts cannot fit if this one did not.

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        json.dumps(
            {
                "schema": "g1.rough_scale_probe.v1",
                "purpose": "measure achievable rough-terrain PPO scale on the local GPU",
                "gpu": _gpu_name(),
                "warmup_steps": WARMUP_STEPS,
                "timed_steps": TIMED_STEPS,
                "records": records,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"-> {OUT_PATH.relative_to(ROOT)}", flush=True)
    return 0


def _gpu_name() -> str | None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return out.strip().splitlines()[0]
    except (OSError, subprocess.CalledProcessError, IndexError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--driver", action="store_true")
    parser.add_argument("--one", nargs=2, metavar=("TASK", "NUM_ENVS"))
    args = parser.parse_args()
    if args.driver:
        return driver()
    if args.one:
        record = measure_one(args.one[0], int(args.one[1]))
        print("RESULT " + json.dumps(record, ensure_ascii=False), flush=True)
        return 0 if record["status"] == "ok" else 1
    parser.error("choose --driver or --one TASK NUM_ENVS")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
