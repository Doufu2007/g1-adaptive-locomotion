"""Preflight, isolated software smoke checks, and explicit A/B/C/D training."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import sys
from datetime import datetime, timezone

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

import mjlab.tasks  # noqa: E402,F401
import src.tasks  # noqa: E402,F401
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg  # noqa: E402
from src.tasks.velocity.config.g1.style_dropout_cfg import STYLE_TASKS  # noqa: E402
from src.tasks.velocity.mdp.style_reference import (  # noqa: E402
  GaitReference, load_gait_reference, reference_term,
)
from scripts.train import TrainConfig, launch_training  # noqa: E402
from scripts.system_check import EXPECTED_PACKAGES  # noqa: E402
from scripts.style_resume import discover_job, stage_private_checkpoint  # noqa: E402
from src.tasks.velocity.rl.resource_control import (  # noqa: E402
  ResourceController, ResourceLimits, TrainingStopRequested, write_state,
)


def write_json(path: Path, data: dict) -> None:
  write_state(path, data)


def publish_terminal(args, limits: ResourceLimits, status: str, reason: str, **extra):
  path = args.resource_directory / "resource_state.json"
  try:
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("pid") != os.getpid():
      state = {}
  except (OSError, ValueError):
    state = {}
  state.update({"updated_at_utc": datetime.now(timezone.utc).isoformat(),
                "pid": os.getpid(), "status": status, "reason": reason,
                "limits": asdict(limits), **extra})
  write_state(path, state)


def configure_reference(cfg, reference: GaitReference, *, synthetic: bool = False):
  term = cfg.commands["gait_reference"]
  term.reference_file = str(reference.path)
  term.allow_synthetic_reference = synthetic
  term.target_velocity = reference.target_velocity
  twist = cfg.commands["twist"]
  vx, vy, yaw = reference.target_velocity
  twist.ranges.lin_vel_x = (vx, vx)
  twist.ranges.lin_vel_y = (vy, vy)
  twist.ranges.ang_vel_z = (yaw, yaw)


def runtime_preflight() -> dict:
  versions = {}
  errors = []
  for name, expected in EXPECTED_PACKAGES.items():
    try:
      actual = importlib.metadata.version(name)
      versions[name] = actual
      if actual.split("+")[0] != expected:
        errors.append(f"{name}: expected {expected}, found {actual}")
    except importlib.metadata.PackageNotFoundError:
      errors.append(f"{name}: not installed")
  if sys.version_info[:2] != (3, 11):
    errors.append("Python 3.11 is required by this locked environment.")
  if not torch.cuda.is_available():
    errors.append("CUDA GPU is unavailable.")
  for task, _ in STYLE_TASKS.values():
    load_env_cfg(task); load_rl_cfg(task)
  return {
    "passed": not errors, "errors": errors, "packages": versions,
    "python": sys.version, "executable": sys.executable,
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "repository": str(ROOT), "tasks": {k: v[0] for k, v in STYLE_TASKS.items()},
  }


def make_software_fixture(directory: Path) -> Path:
  """Constant home pose, deliberately NOT a human or robot walking dataset."""
  import mujoco
  from src.assets.robots.unitree_g1.g1_constants import get_spec, HOME_KEYFRAME

  model = get_spec().compile()
  names = [model.joint(i).name for i in range(model.njnt)
           if model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE]
  pose = np.zeros(len(names), dtype=np.float32)
  for pattern, value in HOME_KEYFRAME.joint_pos.items():
    for index, name in enumerate(names):
      if re.fullmatch(pattern, name):
        pose[index] = value
  directory.mkdir(parents=True, exist_ok=True)
  path = directory / "SYNTHETIC_SOFTWARE_FIXTURE_NOT_WALKING.npz"
  np.savez(
    path, joint_names=np.asarray(names), joint_pos=np.tile(pose, (30, 1)),
    fps=np.array(50.0), gait_period_s=np.array(0.6),
    target_velocity=np.array([0.7, 0.0, 0.0]), loop=np.array(True),
  )
  write_json(path.with_suffix(".json"), {
    "schema_version": 1, "kind": "synthetic_test", "split": "software_test",
    "source_url": "local://constant-g1-home-pose-software-fixture",
    "license": "Generated software fixture using the local robot configuration",
    "description": "CONSTANT HOME POSE. NOT WALKING DATA. NOT RESEARCH RESULTS.",
    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
  })
  return path


def probe_weights(reference: GaitReference) -> dict:
  cfg = load_env_cfg(STYLE_TASKS["D"][0])
  configure_reference(cfg, reference, synthetic=True)
  cfg.scene.num_envs = 8
  cfg.seed = 1042
  action = cfg.actions["joint_pos"]
  action.forced_joint_names = ("left_knee_joint",)
  action.forced_start_step = 4
  action.forced_end_step = 20
  action.forced_dropout_probability = 1.0
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
  try:
    obs, _ = env.reset()
    dims = {name: list(value.shape) for name, value in obs.items()}
    term = reference_term(env)
    samples = []
    for index in range(48):
      env.step(torch.zeros((8, env.action_manager.total_action_dim), device="cuda:0"))
      if index in (1, 13, 21, 45):
        knee = term.joint_names.index("left_knee_joint")
        samples.append({"control_step": index + 1,
                        "left_knee_loss_trace": float(term.loss_trace[:, knee].mean()),
                        "unmasked_error": float(term.diagnostics()[1].mean())})
    if not (samples[1]["left_knee_loss_trace"] > samples[0]["left_knee_loss_trace"]
            and samples[-1]["left_knee_loss_trace"] < samples[1]["left_knee_loss_trace"]):
      raise RuntimeError("Loss trace did not increase during loss and decay afterward.")
    if dims["actor"][-1] != 98 or dims["critic"][-1] != 113:
      raise RuntimeError(f"Observation contract changed: {dims}")
    return {"passed": True, "observation_shapes": dims, "samples": samples}
  finally:
    env.close()


def run_one(group: str, seed: int, reference: GaitReference, args, *, smoke: bool):
  task, _ = STYLE_TASKS[group]
  config = TrainConfig.from_task(task)
  configure_reference(config.env, reference, synthetic=smoke)
  config.env.scene.num_envs = 32 if smoke else args.num_envs
  config.env.seed = seed
  config.agent.seed = seed
  config.agent.max_iterations = 2 if smoke else args.iterations
  config.agent.logger = "tensorboard"
  config.agent.upload_model = False
  config.agent.save_interval = 1 if smoke else 50
  config.agent.run_name = f"{'SOFTWARE_SMOKE' if smoke else 'style_pilot'}_{group}_seed{seed}"
  resumed = None
  if not smoke and args.resume:
    job = args.job_plans[(group, seed)]
    if job["action"] == "skip":
      print(f"[INFO] Retaining completed group {group}/seed{seed}: {job['checkpoint']}", flush=True)
      return {"group": group, "seed": seed, "final_checkpoint": job["checkpoint"],
              "log_directory": str(Path(job["checkpoint"]).parent), "skipped_completed": True,
              "purpose": "DEVELOPMENT_TRAINING"}
    if job["action"] == "resume":
      resumed = stage_private_checkpoint(Path(job["checkpoint"]))
      config.agent.resume = True
  if smoke:
    config.agent.experiment_name = "g1_dropout_style_SOFTWARE_SMOKE"
    action = config.env.actions["joint_pos"]
    action.forced_joint_names = ("left_knee_joint",)
    action.forced_start_step = 4
    action.forced_end_step = 20
    action.forced_dropout_probability = 1.0
  config = TrainConfig(
    env=config.env, agent=config.agent, enable_nan_guard=True, gpu_ids=[0],
    resume_checkpoint_file=resumed["private_snapshot"] if resumed else None,
  )
  launch_training(task, config)
  root = ROOT / "logs" / "rsl_rl" / config.agent.experiment_name
  run = max(root.glob(f"*_{config.agent.run_name}"), key=lambda p: p.stat().st_mtime_ns)
  final = run / f"model_{config.agent.max_iterations - 1}.pt"
  if not final.is_file():
    raise RuntimeError(f"Expected final checkpoint missing: {final}")
  write_json(run / "reference_provenance.json", {
    "reference": str(reference.path), "sha256": reference.sha256,
    "provenance": reference.provenance,
    "purpose": "SOFTWARE_TEST_ONLY" if smoke else "DEVELOPMENT_TRAINING",
    "group": group, "seed": seed,
    "resumed_from": resumed,
  })
  return {"group": group, "seed": seed, "log_directory": str(run),
          "final_checkpoint": str(final), "purpose": "SOFTWARE_TEST_ONLY" if smoke else "DEVELOPMENT_TRAINING"}


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  action = parser.add_mutually_exclusive_group()
  action.add_argument("--check-only", action="store_true")
  action.add_argument("--smoke-test", action="store_true")
  action.add_argument("--train", action="store_true")
  action.add_argument("--plan-only", action="store_true")
  parser.add_argument("--reference-file", type=Path, default=ROOT / "data/style_dropout/walking_reference.npz")
  parser.add_argument("--groups", nargs="+", choices=STYLE_TASKS, default=list(STYLE_TASKS))
  parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
  parser.add_argument("--num-envs", type=int, default=1024)
  parser.add_argument("--iterations", type=int, default=2000)
  parser.add_argument("--cpu-threads", type=int, default=2)
  parser.add_argument("--duty-cycle", type=float, default=0.5)
  parser.add_argument("--pause-temperature", type=float, default=75.0)
  parser.add_argument("--resume-temperature", type=float, default=68.0)
  parser.add_argument("--gpu-memory-budget-mib", type=int, default=4096)
  parser.add_argument("--minimum-rest-s", type=float, default=1.0)
  parser.add_argument("--minimum-host-available-ram-mib", type=int, default=4096)
  parser.add_argument("--minimum-start-host-available-ram-mib", type=int, default=7168)
  parser.add_argument("--resume", action="store_true")
  parser.add_argument("--resource-directory", type=Path, default=ROOT / "results/style_dropout_training")
  parser.add_argument("--report-file", type=Path, default=ROOT / "results/style_dropout_setup/preflight.json")
  args = parser.parse_args()
  if not 1 <= args.num_envs <= 8192 or args.iterations < 1:
    parser.error("num-envs must be 1..8192 and iterations must be positive.")
  if len(set(args.groups)) != len(args.groups) or len(set(args.seeds)) != len(args.seeds):
    parser.error("Groups and seeds must be unique.")
  limits = ResourceLimits(
    cpu_threads=args.cpu_threads, duty_cycle=args.duty_cycle,
    pause_temperature_c=args.pause_temperature,
    resume_temperature_c=args.resume_temperature,
    maximum_gpu_memory_mib=args.gpu_memory_budget_mib,
    minimum_rest_s=args.minimum_rest_s,
    minimum_host_available_ram_mib=args.minimum_host_available_ram_mib,
    minimum_start_host_available_ram_mib=args.minimum_start_host_available_ram_mib,
  )
  os.environ["G1_RESOURCE_LIMITS_JSON"] = json.dumps(asdict(limits))
  os.environ["G1_RESOURCE_DIRECTORY"] = str(args.resource_directory.resolve())
  torch.set_num_threads(limits.cpu_threads)
  if args.train or args.smoke_test:
    import fcntl
    args.resource_directory.mkdir(parents=True, exist_ok=True)
    # Retain the handle for the lifetime of this process; no second local queue.
    queue_lock = (args.resource_directory / "training.lock").open("a+")
    try:
      fcntl.flock(queue_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
      parser.error("Another style training queue holds the single-job lock.")
    if (args.resource_directory / "stop.request").exists():
      parser.error("A manual stop request exists; resolve it before starting a new queue.")
  report = {
    "checked_at_utc": datetime.now(timezone.utc).isoformat(),
    "runtime": runtime_preflight(), "research_reference_ready": False,
    "software_smoke": None, "runs": [], "status": "checking",
    "resource_limits": asdict(limits), "pid": os.getpid(),
  }
  try:
    if not report["runtime"]["passed"]:
      raise RuntimeError("Runtime preflight failed: " + "; ".join(report["runtime"]["errors"]))
    try:
      reference = load_gait_reference(args.reference_file)
      if reference.provenance["split"] != "development":
        raise ValueError("Training requires development references, not held-out test data.")
      report["research_reference_ready"] = True
      report["reference"] = {"path": str(reference.path), "sha256": reference.sha256,
                             "kind": reference.provenance["kind"]}
    except (FileNotFoundError, ValueError, KeyError, json.JSONDecodeError) as error:
      report["reference_issue"] = str(error)
    if args.train or args.plan_only:
      if not report["research_reference_ready"]:
        raise RuntimeError(report["reference_issue"])
      jobs = [discover_job(ROOT, group, seed, reference, args.iterations, args.num_envs)
              if args.resume or args.plan_only else
              {"group": group, "seed": seed, "action": "new", "completed_iterations": 0,
               "target_iterations": args.iterations, "checkpoint": None}
              for seed in args.seeds for group in args.groups]
      args.job_plans = {(job["group"], job["seed"]): job for job in jobs}
      report["resume_plan"] = jobs
      write_state(args.resource_directory / "resume_plan.json", {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(), "jobs": jobs,
        "reference_sha256": reference.sha256,
      })
    if args.smoke_test:
      fixture = make_software_fixture(ROOT / "results/style_dropout_setup/software_fixture")
      reference = load_gait_reference(fixture, allow_synthetic=True)
      report["software_smoke"] = {"purpose": "SOFTWARE_TEST_ONLY", "probe": probe_weights(reference)}
      for group in args.groups:
        report["runs"].append(run_one(group, 42, reference, args, smoke=True))
      report["software_smoke"]["passed"] = True
    elif args.train:
      if not report["research_reference_ready"]:
        raise RuntimeError(report["reference_issue"])
      report["configuration_ready"] = True
      report["formal_training_ready"] = True
      report["planned_jobs"] = len(args.seeds) * len(args.groups)
      report["protocol_stage"] = "PILOT_DEVELOPMENT_NOT_FINAL_PAPER_EXPERIMENT"
      for seed in args.seeds:
        # Rotate arm order by seed to reduce a fixed time/thermal ordering.
        groups = args.groups[seed % len(args.groups):] + args.groups[:seed % len(args.groups)]
        for group in groups:
          os.environ["G1_ACTIVE_STYLE_GROUP"] = group
          os.environ["G1_ACTIVE_STYLE_SEED"] = str(seed)
          report.update({"status": "preparing", "active_group": group, "active_seed": seed})
          write_state(args.report_file, report)
          # Check the host before constructing a simulator or allocating tensors.
          # Completed jobs still pass checkpoint/provenance validation above.
          if args.job_plans[(group, seed)]["action"] != "skip":
            gate = ResourceController(limits, args.resource_directory, args.resource_directory)
            gate.before_allocation(args.iterations)
          report.update({"status": "training", "active_group": group, "active_seed": seed})
          write_state(args.report_file, report)
          report["runs"].append(run_one(group, seed, reference, args, smoke=False))
          write_json(args.report_file, report)
    report["configuration_ready"] = report["runtime"]["passed"]
    report["formal_training_ready"] = report["runtime"]["passed"] and report["research_reference_ready"]
    report["status"] = "completed" if args.train or args.smoke_test else "checked"
    write_json(args.report_file, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if args.smoke_test or report["formal_training_ready"] else 2
  except TrainingStopRequested as error:
    report["status"] = "stopped"
    report["stop_reason"] = str(error)
    report["stop_category"] = error.category
    write_state(args.report_file, report)
    publish_terminal(args, limits, "stopped", str(error), stop_category=error.category,
                     group=report.get("active_group"), seed=report.get("active_seed"))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 3
  except Exception as error:
    report["status"] = "failed"
    report["error"] = f"{type(error).__name__}: {error}"
    write_json(args.report_file, report)
    if args.train or args.smoke_test:
      publish_terminal(args, limits, "failed", report["error"])
    raise


if __name__ == "__main__":
  raise SystemExit(main())
