"""Auto-pilot for gait adjustment project — trains, compares, reports, extends.

Phases:
1. Monitor rough PHASOR training until TARGET reached
2. Run comparison + generate report
3. If before 15:00 CST, increase TARGET and resume training
4. Repeat until 15:00 or diminishing returns
5. Check every 5 minutes — no idle states
"""

import subprocess, sys, os, time
from pathlib import Path
from datetime import datetime, time as dtime

PROJECT_ROOT = Path("/root/unitree_rl_mjlab")
LOG_DIR = PROJECT_ROOT / "logs" / "rsl_rl" / "g1_velocity"
PYTHON = "/root/miniconda3/envs/unitree_rl_mjlab/bin/python"
TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "train.py"
TASK = "Unitree-G1-Rough-PHASOR"
DEADLINE = dtime(15, 0)  # 3 PM CST
CHECK_INTERVAL = 300  # 5 minutes

INITIAL_TARGET = 9500
EXTEND_TARGETS = [15000, 20000, 25000, 30000]  # Progressive targets for extended training

os.environ["WANDB_MODE"] = "disabled"


def log(msg: str):
    t = datetime.now().strftime("%m-%d %H:%M:%S")
    print(f"[{t}] {msg}", flush=True)


def latest_run():
    """Find latest rough PHASOR directory with checkpoints."""
    dirs = sorted(LOG_DIR.glob("*_rough_phasor"), reverse=True)
    for d in dirs:
        models = sorted(d.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
        if models:
            return d, models
    return None, []


def current_iter(models):
    if not models:
        return 0
    return int(sorted(models, key=lambda p: int(p.stem.split("_")[1]))[-1].stem.split("_")[1])


def is_training_alive():
    ps = subprocess.run(["pgrep", "-f", "train.py.*Rough"], capture_output=True, text=True)
    ps2 = subprocess.run(["ps", "aux"], capture_output=True, text=True)
    return bool(ps.stdout.strip()) and "Unitree-G1-Rough-PHASOR" in ps2.stdout


def is_before_deadline():
    return datetime.now().time() < DEADLINE


def run_post_processing(run_dir: Path):
    """Run comparison + report + verification."""
    log("=" * 60)
    log("POST-PROCESSING START")
    log("=" * 60)

    rough_baseline = LOG_DIR / "2026-08-10_23-06-29_rough_baseline"
    rough_output = PROJECT_ROOT / "results" / "rough_comparison"
    rough_output.mkdir(parents=True, exist_ok=True)

    # 1. Comparison
    log("Step 1: Rough terrain comparison...")
    r = subprocess.run(
        [PYTHON, str(PROJECT_ROOT / "scripts" / "compare_phasor.py"),
         "--baseline_dir", str(rough_baseline),
         "--phasor_dir", str(run_dir),
         "--output_dir", str(rough_output)],
        capture_output=True, text=True, timeout=300, cwd=str(PROJECT_ROOT)
    )
    log(r.stdout[-1500:] if r.stdout else "No output")
    if r.stderr:
        log(f"STDERR: {r.stderr[-500:]}")

    # 2. Final report
    log("Step 2: Generating final report...")
    r = subprocess.run(
        [PYTHON, str(PROJECT_ROOT / "scripts" / "generate_final_report.py"),
         "--rough_phasor_dir", str(run_dir)],
        capture_output=True, text=True, timeout=120, cwd=str(PROJECT_ROOT)
    )
    log(r.stdout[-2000:] if r.stdout else "No output")

    # 3. Verification
    log("Step 3: Verification...")
    r = subprocess.run(
        [PYTHON, str(PROJECT_ROOT / "scripts" / "verify_project.py")],
        capture_output=True, text=True, timeout=60, cwd=str(PROJECT_ROOT)
    )
    log(r.stdout[-1000:] if r.stdout else "No output")

    log("POST-PROCESSING DONE")
    return True


def start_training(target_iter: int):
    """Resume or start training with given target."""
    run_dir, models = latest_run()
    ckpt_iter = current_iter(models) if models else 0

    if ckpt_iter >= target_iter:
        log(f"Already at {ckpt_iter} >= {target_iter}, skipping")
        return None

    if run_dir and ckpt_iter > 0:
        # Find best checkpoint to resume from
        best_dir, best_iter = None, 0
        for d in sorted(LOG_DIR.glob("*_rough_phasor"), reverse=True):
            ms = sorted(d.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
            if ms:
                it = int(ms[-1].stem.split("_")[1])
                if it > best_iter:
                    best_iter, best_dir = it, d

        log(f"Resuming from {best_dir.name}, model_{best_iter}.pt → target {target_iter}")
        cmd = [
            PYTHON, "-u", str(TRAIN_SCRIPT), TASK,
            "--env.scene.num-envs=4096",
            f"--agent.max_iterations={target_iter}",
            "--agent.resume", "True",
            "--agent.load-run", best_dir.name,
            "--agent.load-checkpoint", "model_.*",
            "--agent.run-name", "rough_phasor",
        ]
    else:
        log(f"Fresh start, target {target_iter}")
        cmd = [
            PYTHON, "-u", str(TRAIN_SCRIPT), TASK,
            "--env.scene.num-envs=4096",
            f"--agent.max_iterations={target_iter}",
            "--agent.run-name", "rough_phasor",
        ]

    # Kill any existing training process
    subprocess.run(["pkill", "-f", "train.py.*Rough"], capture_output=True)
    time.sleep(10)  # Allow GPU memory to be freed

    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    log(f"Started training PID={proc.pid}, target={target_iter}")
    return proc


def wait_for_target(target_iter: int) -> Path | None:
    """Monitor until training reaches target_iter. Returns the run directory."""
    log(f"Waiting for training to reach {target_iter}...")
    training_was_dead_count = 0

    while True:
        run_dir, models = latest_run()
        ckpt_iter = current_iter(models) if models else 0
        alive = is_training_alive()
        pct = 100 * ckpt_iter / target_iter if target_iter > 0 else 0

        log(f"Progress: {ckpt_iter}/{target_iter} ({pct:.1f}%) | "
            f"Training: {'✅' if alive else '❌'} | Time: {datetime.now().strftime('%H:%M')}")

        if ckpt_iter >= target_iter:
            log(f"✅ Target {target_iter} reached!")
            return run_dir

        if not alive:
            training_was_dead_count += 1
            if training_was_dead_count >= 2:
                log(f"⚠️ Training dead for {training_was_dead_count} checks, restarting...")
                start_training(target_iter)
                training_was_dead_count = 0
                time.sleep(30)
                continue
        else:
            training_was_dead_count = 0

        # Wait 5 minutes before next check
        time.sleep(CHECK_INTERVAL)


def cleanup_old_checkpoints(run_dir: Path, keep_last: int = 3):
    """Keep only the last N checkpoints to save disk space."""
    models = sorted(run_dir.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    if len(models) > keep_last:
        for m in models[:-keep_last]:
            m.unlink()
        log(f"Cleaned {len(models) - keep_last} old checkpoints from {run_dir.name}")


# ═══════════════════════════════════════════════════════════════════
# MAIN AUTO-PILOT LOOP
# ═══════════════════════════════════════════════════════════════════

def main():
    log("=" * 60)
    log("GAIT ADJUSTMENT — AUTO-PILOT STARTED")
    log(f"Deadline: {DEADLINE.strftime('%H:%M')} CST")
    log(f"Check interval: {CHECK_INTERVAL}s (5 min)")
    log("=" * 60)

    # Merge initial target into extend targets
    all_targets = [INITIAL_TARGET] + EXTEND_TARGETS
    completed_targets = set()

    for target in all_targets:
        if not is_before_deadline():
            log(f"Deadline {DEADLINE.strftime('%H:%M')} passed. Stopping auto-pilot.")
            break

        if target in completed_targets:
            continue

        # Wait for training to reach this target
        run_dir = wait_for_target(target)

        if run_dir is None:
            log("ERROR: No run directory found!")
            continue

        # Run post-processing
        run_post_processing(run_dir)

        # Cleanup old checkpoints
        cleanup_old_checkpoints(run_dir)

        completed_targets.add(target)

        # Check if we should extend
        if not is_before_deadline():
            log("Deadline passed. Training complete.")
            break

        # Check if there's a next target
        next_idx = all_targets.index(target) + 1
        if next_idx < len(all_targets):
            next_target = all_targets[next_idx]
            log(f"Before deadline, extending training to {next_target}...")
            start_training(next_target)
        else:
            log("All targets reached. No more extensions available.")
            break

    log("=" * 60)
    log("AUTO-PILOT FINISHED")
    log(f"Completed targets: {sorted(completed_targets)}")
    log("=" * 60)


if __name__ == "__main__":
    main()
