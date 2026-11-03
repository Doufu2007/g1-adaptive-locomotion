"""Fixed-seed pilot evaluation for the completed reference/dropout policies.

The run remains conditional on one training seed and one reference cycle. These
metrics describe this development pilot, not a confirmatory paper test.
"""

from __future__ import annotations

from dataclasses import asdict
import csv
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

import mjlab.tasks  # noqa: E402,F401
import src.tasks  # noqa: E402,F401
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.envs.mdp.actions import JointPositionActionCfg  # noqa: E402
from mjlab.rl import RslRlVecEnvWrapper  # noqa: E402
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls  # noqa: E402
from mjlab.utils.torch import configure_torch_backends  # noqa: E402
from src.tasks.velocity.config.g1.style_dropout_cfg import STYLE_TASKS  # noqa: E402
from src.tasks.velocity.mdp.action_dropout import make_joint_target_dropout_action_cfg  # noqa: E402
from src.tasks.velocity.mdp.style_reference import (  # noqa: E402
  GaitReferenceCommand, load_gait_reference,
)
from src.tasks.velocity.mdp.action_dropout import JointTargetDropoutAction  # noqa: E402
from src.tasks.velocity.rl.resource_control import (  # noqa: E402
  TrainingStopRequested, write_state,
)
from src.tasks.velocity.rl.resource_control import ResourceLimits, available_ram_mib, read_gpu  # noqa: E402


SCENARIOS = {
  "nominal": ((), 0.0),
  "left_knee_75pct_0p4s": (("left_knee_joint",), 0.75),
  "right_ankle_75pct_0p4s": (("right_ankle_pitch_joint",), 0.75),
}
EVAL_SEEDS = (1042, 1043, 1044)


def configure_reference(cfg, reference):
  term = cfg.commands["gait_reference"]
  term.reference_file = str(reference.path)
  term.target_velocity = reference.target_velocity
  twist = cfg.commands["twist"]
  vx, vy, yaw_rate = reference.target_velocity
  twist.ranges.lin_vel_x = (vx, vx)
  twist.ranges.lin_vel_y = (vy, vy)
  twist.ranges.ang_vel_z = (yaw_rate, yaw_rate)


def recovery_statistics(errors: torch.Tensor, start: int, end: int, dt: float) -> tuple[float, float]:
  if end <= start:
    return 0.0, 1.0
  base = errors[max(0, start - round(1.0 / dt)):start].mean(dim=0)
  threshold = torch.maximum(base + 0.15, torch.full_like(base, 0.25))
  hold = max(1, round(0.1 / dt))
  recovered = torch.zeros(errors.shape[1], dtype=torch.bool)
  first = torch.full((errors.shape[1],), errors.shape[0] - end, dtype=torch.long)
  for index in range(end, errors.shape[0] - hold + 1):
    stable = torch.all(errors[index:index + hold] <= threshold.unsqueeze(0), dim=0)
    new = stable & ~recovered
    first[new] = index - end
    recovered |= stable
  return float(first.float().mean()) * dt, float(recovered.float().mean())


def check_budget(limits: ResourceLimits) -> dict:
  sample = {**read_gpu(), "available_ram_mib": available_ram_mib()}
  if sample["memory_used_mib"] > limits.maximum_gpu_memory_mib:
    raise RuntimeError("GPU memory reserve exceeded; evaluation did not start.")
  if sample["available_ram_mib"] < limits.minimum_available_ram_mib:
    raise RuntimeError("WSL memory reserve exceeded; evaluation did not start.")
  if sample["temperature_c"] >= limits.critical_temperature_c:
    raise RuntimeError("GPU is above the critical temperature; evaluation did not start.")
  return sample


def evaluate_one(group: str, checkpoint: Path, scenario: str, eval_seed: int,
                 reference_path: Path, num_envs: int, limits: ResourceLimits,
                 training_seed: int = 42, control_directory: Path | None = None,
                 scenario_config: dict | None = None, simulation_seconds: float = 20.0) -> dict:
  if not math.isfinite(simulation_seconds) or not 8.0 <= simulation_seconds <= 20.0:
    raise ValueError("Evaluation simulation duration must be between 8 and 20 seconds.")
  control = control_directory or ROOT / "results/style_dropout_training"
  control.mkdir(parents=True, exist_ok=True)
  budget_before = check_budget(limits)
  task_id, mode = STYLE_TASKS[group]
  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)
  reference = load_gait_reference(reference_path)
  configure_reference(env_cfg, reference)
  env_cfg.seed = eval_seed
  env_cfg.scene.num_envs = num_envs
  env_cfg.episode_length_s = simulation_seconds
  env_cfg.curriculum = {}
  env_cfg.events.pop("push_robot", None)
  env_cfg.observations["actor"].enable_corruption = False
  twist = env_cfg.commands["twist"]
  twist.heading_command = False
  twist.ranges.heading = None
  twist.rel_standing_envs = 0.0
  base_action = env_cfg.actions["joint_pos"]
  if not isinstance(base_action, JointPositionActionCfg):
    raise TypeError("G1 reference evaluation requires joint position targets.")
  if scenario_config is None:
    joints, probability = SCENARIOS[scenario]
    fault_duration_s = 0.4
  else:
    joints = tuple(scenario_config['joints'])
    probability = float(scenario_config['probability'])
    fault_duration_s = float(scenario_config['duration_s'])
  configured_dt = env_cfg.decimation * env_cfg.sim.mujoco.timestep
  fault_start = round(6.0 / configured_dt)
  fault_end = fault_start + max(1, round(fault_duration_s / configured_dt))
  action = make_joint_target_dropout_action_cfg(
    base_action, enabled=True, event_probability=0.0,
    forced_joint_names=joints,
    forced_start_step=fault_start if joints else -1,
    forced_end_step=fault_end if joints else -1,
    forced_dropout_probability=probability if joints else -1.0,
  )
  env_cfg.actions["joint_pos"] = action
  device = "cuda:0"
  base_env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
  env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(task_id)
  runner = runner_cls(env, asdict(agent_cfg), device=device) if runner_cls else None
  # The study task has a checkpoint-aware OnPolicyRunner; reject a registry drift.
  if runner is None:
    env.close()
    raise RuntimeError(f"No inference runner registered for {task_id}.")
  try:
    runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device)
    policy = runner.get_inference_policy(device=device)
    raw_env = env.unwrapped
    raw_env.common_step_counter = 0
    obs = env.get_observations()
    reference_term = raw_env.command_manager.get_term("gait_reference")
    if not isinstance(reference_term, GaitReferenceCommand):
      raise TypeError("Reference command did not load into the evaluation environment.")
    action_term = raw_env.action_manager.get_term("joint_pos")
    robot = raw_env.scene["robot"]
    steps = round(simulation_seconds / raw_env.step_dt)
    errors = torch.empty((steps, num_envs), dtype=torch.float32)
    reference_errors = torch.empty((steps, num_envs), dtype=torch.float32)
    alive = torch.empty((steps, num_envs), dtype=torch.bool)
    unaffected_errors = torch.empty_like(reference_errors)
    unaffected = [i for i,n in enumerate(robot.joint_names) if n not in joints]
    fall_envs = torch.zeros(num_envs, dtype=torch.bool, device=device)
    fall_count = 0
    episodes = 0
    dropped_samples = 0
    selected_samples = 0
    tilt_sq = torch.zeros((), device=device)
    action_delta_sq = torch.zeros((), device=device)
    reward_sum = torch.zeros((), device=device)
    previous_actions = None
    start = time.monotonic()
    with torch.inference_mode():
      for index in range(steps):
        if index % 25 == 0:
          stop_file = control / "stop.request"
          if stop_file.exists():
            raise TrainingStopRequested("Manual stop request received during evaluation.")
          budget = check_budget(limits)
          cooling_started = None
          if budget["temperature_c"] >= limits.pause_temperature_c:
            cooling_started = time.monotonic()
            while budget["temperature_c"] > limits.resume_temperature_c:
              if stop_file.exists():
                raise TrainingStopRequested("Manual stop request received during evaluation cooling.")
              write_state(control / "resource_state.json", {
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                "pid": os.getpid(), "status": "evaluation_pause", "iteration": index,
                "total_iterations": steps, "evaluation_group": group,
                "evaluation_seed": eval_seed, "scenario": scenario,
                "telemetry": budget, "limits": asdict(limits),
              })
              if time.monotonic() - cooling_started > limits.maximum_pause_s:
                raise TrainingStopRequested("Evaluation thermal pause exceeded its time limit.")
              time.sleep(5)
              budget = check_budget(limits)
          write_state(control / "resource_state.json", {
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "pid": os.getpid(), "status": "evaluating", "iteration": index,
            "total_iterations": steps, "evaluation_group": group,
            "evaluation_seed": eval_seed, "scenario": scenario,
            "telemetry": budget, "limits": asdict(limits),
          })
        actions = policy(obs)
        obs, rewards, dones, extras = env.step(actions.to(device))
        reference_pos = reference_term.command
        command = raw_env.command_manager.get_command("twist")
        speed_error = command[:, :2] - robot.data.root_link_lin_vel_b[:, :2]
        errors[index] = torch.linalg.vector_norm(speed_error, dim=1).cpu()
        joint_error = (robot.data.joint_pos - reference_pos) / reference_term.cfg.joint_sigma_rad
        reference_errors[index] = torch.sqrt(torch.mean(joint_error.square(), dim=1)).cpu()
        unaffected_errors[index] = joint_error[:,unaffected].square().mean(dim=1).sqrt().cpu()
        tilt_sq += robot.data.projected_gravity_b[:, :2].square().sum()
        reward_sum += rewards.sum()
        if previous_actions is not None:
          action_delta_sq += (actions - previous_actions).square().sum()
        previous_actions = actions.clone()
        timeouts = extras.get("time_outs", torch.zeros_like(dones, dtype=torch.bool))
        fell = dones.bool() & ~timeouts.bool()
        fall_count += fell.sum()
        episodes += dones.sum()
        dropped_samples += int(action_term.dropout_mask.sum().item())
        selected_samples += int(action_term.selected_fault_mask.sum().item())
        fall_envs |= fell
        alive[index] = (~fall_envs).cpu()
    fall_count = int(fall_count.item())
    episodes = int(episodes.item())
    full_error, full_style = errors.square().mean().sqrt().item(), reference_errors.mean().item()
    fault_error, fault_style = errors[fault_start:fault_end].square().mean().sqrt().item(), reference_errors[fault_start:fault_end].mean().item()
    post_error, post_style = errors[fault_end:fault_end + 100].square().mean().sqrt().item(), reference_errors[fault_end:fault_end + 100].mean().item()
    recovery_s, recovery_rate = recovery_statistics(errors, fault_start, fault_end, raw_env.step_dt) if joints else (0.0, 1.0)
    action_rms = torch.sqrt(action_delta_sq / (num_envs * max(steps - 1, 1) * actions.shape[-1])).item()
    after = check_budget(limits)
    enhanced = {}
    if scenario_config is not None:
      from scripts.style_evaluation_metrics import alive_rmse, recovery_without_falls
      enhanced = {
        'metric_schema_version':2, 'fault_probability':probability,
        'fault_duration_s':fault_duration_s, 'fault_start_s':fault_start*raw_env.step_dt,
        'first_life_xy_rmse_mps':alive_rmse(errors,alive,0,steps),
        'first_life_reference_rmse_scaled':alive_rmse(reference_errors,alive,0,steps),
        'fault_alive_xy_rmse_mps':alive_rmse(errors,alive,fault_start,fault_end) if joints else None,
        'fault_alive_reference_rmse_scaled':alive_rmse(reference_errors,alive,fault_start,fault_end) if joints else None,
        'fault_alive_unaffected_joint_rmse_scaled':alive_rmse(unaffected_errors,alive,fault_start,fault_end) if joints else None,
        'post_fault_alive_reference_rmse_scaled':alive_rmse(reference_errors,alive,fault_end,fault_end+100) if joints else None,
        'fault_entry_survivors':int(alive[fault_start-1].sum()),
        'fault_exit_survivors':int(alive[fault_end-1].sum()),
        'legacy_metrics_scope':'Legacy aggregate metrics include automatic reset episodes; use first-life fields and survival together.',
      }
      enhanced.update(recovery_without_falls(errors,alive,fault_start,fault_end,raw_env.step_dt) if joints else {
        'recovery_without_fall_rate':None, 'recovery_eligible_envs':None,
        'recovery_restricted_mean_time_s':None, 'recovery_censor_horizon_s':None})
    return {
      "group": group, "mode": mode, "train_seed": training_seed, "evaluation_seed": eval_seed,
      "scenario": scenario, "num_envs": num_envs, "duration_s": simulation_seconds,
      "reference_sha256": reference.sha256, "checkpoint": str(checkpoint),
      "xy_speed_rmse_mps": full_error,
      "fault_window_xy_rmse_mps": fault_error if joints else None,
      "post_fault_xy_rmse_mps": post_error if joints else None,
      "reference_joint_rmse_scaled": full_style,
      "fault_reference_rmse_scaled": fault_style if joints else None,
      "post_fault_reference_rmse_scaled": post_style if joints else None,
      "survival_rate": 1.0 - float(fall_envs.float().mean()),
      "fall_rate_per_episode": fall_count / max(episodes, 1),
      "episodes": episodes, "falls": fall_count,
      "recovery_time_s": recovery_s if joints else None,
      "recovery_rate": recovery_rate if joints else None,
      "observed_target_dropouts": dropped_samples,
      "selected_target_samples": selected_samples,
      "observed_dropout_probability": dropped_samples / max(selected_samples, 1),
      "torso_tilt_proxy_rms": torch.sqrt(tilt_sq / (num_envs * steps)).item(),
      "action_delta_rms": action_rms, "mean_reward_per_step": reward_sum.item() / (num_envs * steps),
      "wall_seconds": time.monotonic() - start,
      "gpu_temperature_c_start": budget_before["temperature_c"],
      "gpu_temperature_c_end": after["temperature_c"],
      "gpu_memory_mib_peak_sampled": max(budget_before["memory_used_mib"], after["memory_used_mib"]),
      **enhanced,
    }
  finally:
    env.close()


def main():
  import argparse
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint-root", type=Path, default=ROOT / "logs/rsl_rl/g1_dropout_style")
  parser.add_argument("--reference-file", type=Path, default=ROOT / "data/style_dropout/walking_reference.npz")
  parser.add_argument("--num-envs", type=int, default=128)
  parser.add_argument("--report-root", type=Path, default=ROOT / "results/style_dropout_training/evaluation/seed42_pilot")
  args = parser.parse_args()
  if not 1 <= args.num_envs <= 512:
    parser.error("Evaluation is limited to 1..512 environments to preserve workstation headroom.")
  limits = ResourceLimits(cpu_threads=2, duty_cycle=0.5, minimum_rest_s=1.0)
  torch.set_num_threads(2)
  os.nice(10)
  control = ROOT / "results/style_dropout_training"
  control.mkdir(parents=True, exist_ok=True)
  lock = (control / "training.lock").open("a+")
  try:
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
  except BlockingIOError:
    parser.error("Another training/evaluation queue owns the single-GPU lock.")
  reference = load_gait_reference(args.reference_file)
  if reference.provenance.get("split") != "development":
    parser.error("The pilot evaluator cannot consume held-out test references.")
  expected_ref_hash = reference.sha256
  runs = {}
  summaries = {}
  records = []
  args.report_root.mkdir(parents=True, exist_ok=True)
  report_path = args.report_root / "evaluation_status.json"
  shared_status_path = control / "resource_state.json"
  csv_path = args.report_root / "evaluation.csv"
  if report_path.exists() or csv_path.exists():
    parser.error("Evaluation outputs already exist; choose a new report root to preserve prior results.")
  for group, (task, _) in STYLE_TASKS.items():
    matching = [p for p in args.checkpoint_root.glob(f"*_style_pilot_{group}_seed42") if (p / "model_1999.pt").is_file()]
    if len(matching) != 1:
      parser.error(f"Expected one completed seed-42 {group} run; found {len(matching)}.")
    run = matching[0]
    provenance_path = run / "reference_provenance.json"
    if not provenance_path.exists():
      parser.error(f"Reference provenance is missing from {run}.")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance.get("sha256") != expected_ref_hash or provenance.get("purpose") != "DEVELOPMENT_TRAINING":
      parser.error(f"Reference provenance is incompatible with {run}.")
    runs[group] = run / "model_1999.pt"
  report = {
    "status": "evaluating", "started_at_utc": datetime.now(timezone.utc).isoformat(),
    "purpose": "PILOT_EVALUATION_ONE_TRAIN_SEED_CONDITIONAL_ON_ONE_DEVELOPMENT_REFERENCE",
    "training_seed": 42, "evaluation_seeds": EVAL_SEEDS,
    "num_envs_per_evaluation": args.num_envs, "duration_s": 20.0,
    "fault_start_s": 6.0, "reference_sha256": expected_ref_hash,
    "scenarios": list(SCENARIOS), "results": [], "resource_limits": asdict(limits),
    "panel_seed_checkpoint_root": str(args.checkpoint_root.resolve()),
  }
  write_state(report_path, report)
  write_state(shared_status_path, {
    "updated_at_utc": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(),
    "status": "evaluation_preparing", "iteration": 0,
    "total_iterations": len(EVAL_SEEDS) * len(SCENARIOS) * len(STYLE_TASKS),
    "evaluation_group": "A", "evaluation_seed": EVAL_SEEDS[0], "scenario": "nominal",
    "gpu": "cuda:0", "limits": asdict(limits),
  })
  columns = [
    "group", "mode", "train_seed", "evaluation_seed", "scenario", "num_envs",
    "duration_s", "reference_sha256", "checkpoint", "xy_speed_rmse_mps",
    "fault_window_xy_rmse_mps", "post_fault_xy_rmse_mps", "reference_joint_rmse_scaled",
    "fault_reference_rmse_scaled", "post_fault_reference_rmse_scaled", "survival_rate",
    "fall_rate_per_episode", "episodes", "falls", "recovery_time_s", "recovery_rate",
    "observed_target_dropouts", "selected_target_samples", "observed_dropout_probability",
    "torso_tilt_proxy_rms", "action_delta_rms", "mean_reward_per_step",
    "wall_seconds", "gpu_temperature_c_start", "gpu_temperature_c_end",
    "gpu_memory_mib_peak_sampled",
  ]
  try:
    for eval_seed in EVAL_SEEDS:
      for scenario in SCENARIOS:
        for group in STYLE_TASKS:
          if (control / "stop.request").exists():
            raise TrainingStopRequested("Manual stop request received before the next evaluation.")
          report.update({"active_group": group, "active_evaluation_seed": eval_seed,
                         "active_scenario": scenario})
          write_state(report_path, report)
          result = evaluate_one(group, runs[group], scenario, eval_seed,
                                reference.path, args.num_envs, limits)
          records.append(result)
          with csv_path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            if stream.tell() == 0:
              writer.writeheader()
            writer.writerow(result)
    for group in STYLE_TASKS:
      summaries[group] = {}
      for scenario in SCENARIOS:
        subset = [x for x in records if x["group"] == group and x["scenario"] == scenario]
        metrics = ["xy_speed_rmse_mps", "reference_joint_rmse_scaled",
                   "survival_rate", "fall_rate_per_episode", "action_delta_rms"]
        if scenario != "nominal":
          metrics.extend(("fault_window_xy_rmse_mps", "post_fault_xy_rmse_mps",
                          "fault_reference_rmse_scaled", "post_fault_reference_rmse_scaled",
                          "recovery_time_s", "recovery_rate"))
        summaries[group][scenario] = {
          metric: {"mean": statistics.mean(x[metric] for x in subset),
                   "sd_across_evaluation_seeds": statistics.stdev(x[metric] for x in subset)}
          for metric in metrics
        }
    report.update({"status": "completed", "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                   "completed_scenarios": len(records), "summary": summaries})
    write_state(report_path, report)
    write_state(shared_status_path, {
      "updated_at_utc": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(),
      "status": "evaluation_completed", "iteration": len(records),
      "total_iterations": len(EVAL_SEEDS) * len(SCENARIOS) * len(STYLE_TASKS),
      "evaluation_file": str(csv_path), "limits": asdict(limits),
    })
    print(json.dumps(report, indent=2))
    return 0
  except TrainingStopRequested as error:
    report.update({"status": "stopped", "stop_reason": str(error), "completed_scenarios": len(records)})
    write_state(report_path, report)
    write_state(shared_status_path, {
      "updated_at_utc": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(),
      "status": "evaluation_stopped", "iteration": len(records),
      "total_iterations": len(EVAL_SEEDS) * len(SCENARIOS) * len(STYLE_TASKS),
      "limits": asdict(limits), "reason": str(error),
    })
    return 3
  except Exception as error:
    report.update({"status": "failed", "error": f"{type(error).__name__}: {error}",
                   "completed_scenarios": len(records)})
    write_state(report_path, report)
    write_state(shared_status_path, {
      "updated_at_utc": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(),
      "status": "evaluation_failed", "iteration": len(records),
      "total_iterations": len(EVAL_SEEDS) * len(SCENARIOS) * len(STYLE_TASKS),
      "limits": asdict(limits), "reason": str(error),
    })
    raise


if __name__ == "__main__":
  raise SystemExit(main())
