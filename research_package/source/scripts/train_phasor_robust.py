"""Robust PHASOR trainer — auto-resumes and retries on any crash."""
import subprocess, sys, os, time, re, glob
from pathlib import Path

LOG_DIR = Path("/root/unitree_rl_mjlab/logs/rsl_rl/g1_velocity")
TARGET = 10001
PYTHON = "/root/miniconda3/envs/unitree_rl_mjlab/bin/python"
TRAIN_SCRIPT = "/root/unitree_rl_mjlab/scripts/train.py"

os.environ["WANDB_MODE"] = "disabled"

def latest_run_dir():
    dirs = sorted(LOG_DIR.glob("*_phasor_full"), reverse=True)
    for d in dirs:
        models = sorted(d.glob("model_*.pt"))
        if models:
            return d, models
    return None, []

def current_iter(models):
    if not models:
        return 0
    latest = sorted(models, key=lambda p: int(p.stem.split("_")[1]))[-1]
    return int(latest.stem.split("_")[1])

def clean_empty():
    for d in LOG_DIR.glob("*_phasor_full"):
        if not list(d.glob("model_*.pt")):
            import shutil
            shutil.rmtree(d, ignore_errors=True)
            print(f"[CLEAN] Removed empty dir: {d.name}")

for attempt in range(1, 100):
    clean_empty()
    run_dir, models = latest_run_dir()
    ckpt_iter = current_iter(models) if models else 0

    print(f"\n{'='*60}")
    print(f"Attempt {attempt} | Current iter: {ckpt_iter} | Target: {TARGET}")
    print(f"{'='*60}")

    if ckpt_iter >= TARGET:
        print(f"[DONE] Already at {ckpt_iter} >= {TARGET}")
        break

    if run_dir:
        # Find previous run with most iterations for resume
        prev_runs = sorted(LOG_DIR.glob("*_phasor_full"), reverse=True)
        best_dir = None
        best_iter = 0
        for d in prev_runs:
            ms = sorted(d.glob("model_*.pt"))
            if ms:
                it = int(ms[-1].stem.split("_")[1])
                if it > best_iter:
                    best_iter = it
                    best_dir = d

        print(f"[INFO] Resuming from {best_dir.name}, model_{best_iter}.pt")
        cmd = [
            PYTHON, "-u", TRAIN_SCRIPT,
            "Unitree-G1-Flat-PHASOR",
            "--env.scene.num-envs=4096",
            f"--agent.max_iterations={TARGET}",
            "--agent.resume", "True",
            "--agent.load-run", best_dir.name,  # exact match
            "--agent.load-checkpoint", "model_.*",
            "--agent.run-name", "phasor_full",
        ]
    else:
        print("[INFO] Fresh start")
        cmd = [
            PYTHON, "-u", TRAIN_SCRIPT,
            "Unitree-G1-Flat-PHASOR",
            "--env.scene.num-envs=4096",
            f"--agent.max_iterations={TARGET}",
            "--agent.run-name", "phasor_full",
        ]

    print(f"[CMD] {' '.join(cmd)}")
    start = time.time()

    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1)
        for line in proc.stdout:
            line = line.rstrip()
            if "Learning iteration" in line or "Error" in line or "Traceback" in line:
                print(f"  {line.strip()}")

        proc.wait()
        elapsed = time.time() - start
        print(f"[EXIT] code={proc.returncode}, elapsed={elapsed/60:.1f}min")

    except Exception as e:
        elapsed = time.time() - start
        print(f"[CRASH] {e}, elapsed={elapsed/60:.1f}min")

    # Check progress
    _, models = latest_run_dir()
    ckpt_iter = current_iter(models) if models else 0
    print(f"[PROGRESS] {ckpt_iter}/{TARGET} ({100*ckpt_iter/TARGET:.1f}%)")

    if ckpt_iter >= TARGET:
        print("[DONE] Training complete!")
        break

    print("[INFO] Restarting in 10s...")
    time.sleep(10)

print(f"\nFinal models: {list((LOG_DIR.glob('*_phasor_full')[-1] if list(LOG_DIR.glob('*_phasor_full')) else Path()).glob('model_*.pt'))[:5]}")
