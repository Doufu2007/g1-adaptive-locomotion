"""Verify gait adjustment project completeness — check all models, results, artifacts."""
import sys, json, os
from pathlib import Path
import torch
import numpy as np

PROJECT_ROOT = Path("/root/unitree_rl_mjlab")
LOG_DIR = PROJECT_ROOT / "logs/rsl_rl/g1_velocity"

CHECKS = []

def check(name: str, condition: bool, detail: str = ""):
    status = "✅" if condition else "❌"
    CHECKS.append(f"{status} {name}: {detail}")
    return condition

def verify():
    print("=" * 70)
    print("GAIT ADJUSTMENT PROJECT — VERIFICATION")
    print("=" * 70)

    # ── Trained Models ──────────────────────────────────────────────
    print("\n--- Trained Models ---")

    flat_baseline = LOG_DIR / "2026-08-10_09-35-45_baseline_full"
    flat_phasor = LOG_DIR / "2026-08-10_21-49-27_phasor_resume_final"
    rough_baseline = LOG_DIR / "2026-08-10_23-06-29_rough_baseline"

    # Flat baseline
    fb_model = flat_baseline / "model_10000.pt"
    has_model = check("Flat baseline model", fb_model.exists(), str(fb_model))
    fb_onnx = flat_baseline / "policy.onnx"
    has_onnx = check("Flat baseline ONNX", fb_onnx.exists(), f"{fb_onnx.stat().st_size/1024:.0f}KB" if fb_onnx.exists() else "")

    # Flat PHASOR
    fp_model = flat_phasor / "model_9999.pt"
    has_model &= check("Flat PHASOR model", fp_model.exists(), str(fp_model))
    fp_onnx = flat_phasor / "policy.onnx"
    has_onnx &= check("Flat PHASOR ONNX", fp_onnx.exists(), f"{fp_onnx.stat().st_size/1024:.0f}KB" if fp_onnx.exists() else "")

    # Rough baseline
    rb_models = sorted(rough_baseline.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    rb_iter = int(rb_models[-1].stem.split("_")[1]) if rb_models else 0
    check(f"Rough baseline model (iter {rb_iter})", rb_iter >= 9900, str(rb_models[-1] if rb_models else "N/A"))
    rb_onnx = rough_baseline / "policy.onnx"
    check("Rough baseline ONNX", rb_onnx.exists(), f"{rb_onnx.stat().st_size/1024:.0f}KB" if rb_onnx.exists() else "")

    # Rough PHASOR (training)
    rough_phasor_dirs = sorted(LOG_DIR.glob("*_rough_phasor"), reverse=True)
    rp_iter = 0
    for d in rough_phasor_dirs:
        models = sorted(d.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
        if models:
            rp_iter = int(models[-1].stem.split("_")[1])
            break
    check(f"Rough PHASOR model (iter {rp_iter})", rp_iter > 0, "Training in progress" if rp_iter < 10000 else "COMPLETE!")

    # ── Phase Encoder ───────────────────────────────────────────────
    print("\n--- Phase Encoder ---")
    enc_path = PROJECT_ROOT / "src/tasks/velocity/phasor/phasor_encoder.pt"
    phase_circle = PROJECT_ROOT / "src/tasks/velocity/phasor/phase_circle.png"

    check("Encoder checkpoint", enc_path.exists(), f"{enc_path.stat().st_size/1024:.0f}KB" if enc_path.exists() else "")
    check("Phase circle plot", phase_circle.exists(), "")

    # Try loading encoder
    if enc_path.exists():
        try:
            ckpt = torch.load(enc_path, map_location="cpu", weights_only=False)
            check("Encoder state_dict", "state_dict" in ckpt, f"joint_dim={ckpt.get('joint_dim')}, window={ckpt.get('window')}")
            check("Encoder normalization", "mean" in ckpt and "std" in ckpt, "")
        except Exception as e:
            check("Encoder loadable", False, str(e))

    # ── Results & Reports ───────────────────────────────────────────
    print("\n--- Results & Reports ---")
    results = PROJECT_ROOT / "results"

    check("Flat comparison summary", (results / "phasor_comparison" / "summary.txt").exists(), "")
    check("Flat comparison chart v2", (results / "phasor_comparison" / "training_comparison_v2.png").exists(), "")
    check("Flat comparison chart", (results / "phasor_comparison" / "training_comparison.png").exists(), "")
    check("Flat final report", (results / "phasor_comparison" / "FINAL_REPORT.md").exists(), "")
    check("Rough baseline precomputed", (results / "rough_baseline_precomputed.json").exists(), "")
    check("PINN results", (results / "results.json").exists(), "")
    check("PINN summary", (results / "SUMMARY.md").exists(), "")

    final_report = results / "final_report" / "FINAL_REPORT.md"
    check("Final comprehensive report", final_report.exists(), "")

    # ── Data ────────────────────────────────────────────────────────
    print("\n--- Data ---")
    trajectory = PROJECT_ROOT / "data" / "g1_trajectory.npz"
    check("G1 trajectory data", trajectory.exists(), f"{trajectory.stat().st_size/1024/1024:.0f}MB" if trajectory.exists() else "")

    # ── PINN Models ─────────────────────────────────────────────────
    print("\n--- PINN Models ---")
    models_dir = PROJECT_ROOT / "models"
    pinn_models = sorted(models_dir.glob("pinn_*.pt"))
    check(f"PINN models ({len(pinn_models)})", len(pinn_models) >= 5, str([m.name for m in pinn_models]))

    # ── Training Process ────────────────────────────────────────────
    print("\n--- Training Status ---")
    import subprocess
    ps = subprocess.run(["pgrep", "-f", "train.py"], capture_output=True, text=True)
    train_pids = ps.stdout.strip().split()
    check(f"Training process{'es' if len(train_pids) > 1 else ''}", len(train_pids) > 0,
          f"PID(s): {', '.join(train_pids)}")

    ps2 = subprocess.run(["pgrep", "-f", "complete_project"], capture_output=True, text=True)
    mon_pids = ps2.stdout.strip().split()
    check(f"Monitor pipeline", len(mon_pids) > 0, f"PID(s): {', '.join(mon_pids)}")

    # ── GPU ─────────────────────────────────────────────────────────
    try:
        import subprocess
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                             "--format=csv,noheader"], capture_output=True, text=True)
        if gpu.returncode == 0:
            check("GPU available", True, gpu.stdout.strip())
    except:
        pass

    # ── Summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    passed = sum(1 for c in CHECKS if c.startswith("✅"))
    failed = sum(1 for c in CHECKS if c.startswith("❌"))
    print(f"Passed: {passed}, Failed: {failed}, Total: {len(CHECKS)}")

    if failed > 0:
        print("\nFAILED CHECKS:")
        for c in CHECKS:
            if c.startswith("❌"):
                print(f"  {c}")
    else:
        print("\n🎉 ALL CHECKS PASSED!" if rp_iter >= 10000 or "Rough PHASOR" not in str(CHECKS)
              else f"\n⏳ Rough PHASOR still training ({rp_iter}/10001) — will be complete soon.")

    print(f"\nChecklist ({passed}/{len(CHECKS)}):")
    for c in CHECKS:
        print(f"  {c}")

    return failed == 0

if __name__ == "__main__":
    ok = verify()
    sys.exit(0 if ok else 1)
