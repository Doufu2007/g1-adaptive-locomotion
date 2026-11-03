#!/usr/bin/env python3
"""Quick training status dashboard — no arguments, instant readout."""
import subprocess, sys, os
from pathlib import Path
from datetime import datetime

LOG_DIR = Path("/root/unitree_rl_mjlab/logs/rsl_rl/g1_velocity")
TARGET = 10001

def latest_phasor():
    dirs = sorted(LOG_DIR.glob("*_rough_phasor"), reverse=True)
    for d in dirs:
        models = sorted(d.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
        if models:
            return d, models
    return None, []

def main():
    d, models = latest_phasor()
    if not d:
        print("No rough PHASOR run found!")
        return
    iters = sorted(int(m.stem.split("_")[1]) for m in models)
    latest = iters[-1]
    prev = iters[-2] if len(iters) > 1 else 0
    pct = 100 * latest / TARGET

    # Compute rate from last 2 checkpoints
    import time
    now = time.time()
    latest_mtime = max(m.stat().st_mtime for m in models)
    age = now - latest_mtime  # seconds since last checkpoint

    # Rate based on recent progress
    if len(iters) >= 3:
        recent = iters[-1] - iters[-3]
        rate = recent / ((now - min(m.stat().st_mtime for m in models[-3:])) / 60) if recent > 0 else 0
    else:
        rate = 0

    eta_min = (TARGET - latest) / rate if rate > 0 else 999
    eta_time = datetime.now().strftime('%H:%M') if rate == 0 else \
               datetime.fromtimestamp(now + eta_min * 60).strftime('%H:%M')

    # GPU
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,temperature.gpu",
                         "--format=csv,noheader"], capture_output=True, text=True)
    gpu_str = gpu.stdout.strip()

    # Process
    ps = subprocess.run(["ps", "aux"], capture_output=True, text=True)
    train_alive = "train.py" in ps.stdout and "Rough-PHASOR" in ps.stdout
    mon_alive = "complete_project" in ps.stdout

    print(f"╔══════════════════════════════════════════════════════════════╗")
    print(f"║  ROUGH PHASOR TRAINING STATUS  │  {datetime.now().strftime('%H:%M:%S')}               ║")
    print(f"╠══════════════════════════════════════════════════════════════╣")
    print(f"║  Progress:  {latest:>5d} / {TARGET}  ({pct:5.1f}%)                          ║")
    bar_len = 40
    filled = int(bar_len * latest / TARGET)
    bar = "█" * filled + "░" * (bar_len - filled)
    print(f"║  [{bar}]  ║")
    print(f"║  Rate:      {rate:>5.1f} iter/min                              ║")
    print(f"║  ETA:       {eta_min:>5.0f} min  (~{eta_time})                         ║")
    print(f"║  Last ckpt: {latest} (saved {age:.0f}s ago)                         ║")
    print(f"║  GPU:       {gpu_str:<45s} ║")
    print(f"║  Train:     {'✅ RUNNING' if train_alive else '❌ DEAD':<20s}                          ║")
    print(f"║  Monitor:   {'✅ RUNNING' if mon_alive else '❌ DEAD':<20s}                          ║")
    print(f"║  Run dir:   {d.name:<40s} ║")
    print(f"╚══════════════════════════════════════════════════════════════╝")

if __name__ == "__main__":
    main()
