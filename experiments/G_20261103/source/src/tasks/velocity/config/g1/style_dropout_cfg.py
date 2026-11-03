"""Isolated A/B/C/D tasks for local relaxation of a gait reference."""

from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg

from src.tasks.velocity.mdp.style_reference import (
  GaitReferenceCommandCfg, StyleMode, reference_feet_gait, reference_phase,
  style_imitation_cost, style_unmasked_error, style_weight_mass,
)

from .env_cfgs import unitree_g1_flat_joint_dropout_robust_env_cfg
from .rl_cfg import unitree_g1_joint_dropout_ppo_runner_cfg


STYLE_TASKS = {
  "A": ("Unitree-G1-Flat-Dropout-Style-None", "none"),
  "B": ("Unitree-G1-Flat-Dropout-Style-Fixed", "fixed"),
  "C": ("Unitree-G1-Flat-Dropout-Style-Global", "global"),
  "D": ("Unitree-G1-Flat-Dropout-Style-Selective", "selective"),
}


def style_dropout_env_cfg(mode: StyleMode, play: bool = False):
  cfg = unitree_g1_flat_joint_dropout_robust_env_cfg(play=play)
  cfg.commands["gait_reference"] = GaitReferenceCommandCfg(
    mode=mode, resampling_time_range=(1.0e9, 1.0e9)
  )
  # All arms use the same single-speed pilot protocol. The study launcher
  # replaces this with the supplied reference's declared velocity.
  twist = cfg.commands["twist"]
  twist.heading_command = False
  twist.ranges.heading = None
  twist.rel_standing_envs = 0.0
  twist.resampling_time_range = (1.0e9, 1.0e9)
  twist.ranges.lin_vel_x = (0.7, 0.7)
  twist.ranges.lin_vel_y = (0.0, 0.0)
  twist.ranges.ang_vel_z = (0.0, 0.0)
  # Freeze command-difficulty changes in this new protocol for all four arms.
  cfg.curriculum = {}
  for group in ("actor", "critic"):
    cfg.observations[group].terms["phase"] = ObservationTermCfg(func=reference_phase)
  gait = cfg.rewards["foot_gait"]
  gait.func = reference_feet_gait
  gait.params = {k: v for k, v in gait.params.items() if k != "period"}
  cfg.rewards["style_imitation"] = RewardTermCfg(
    func=style_imitation_cost, weight=0.0 if mode == "none" else -1.0
  )
  cfg.metrics["style_unmasked_error"] = MetricsTermCfg(func=style_unmasked_error)
  cfg.metrics["style_weight_mass"] = MetricsTermCfg(func=style_weight_mass)
  return cfg


def style_dropout_rl_cfg():
  cfg = unitree_g1_joint_dropout_ppo_runner_cfg()
  cfg.experiment_name = "g1_dropout_style"
  cfg.wandb_project = "unitree-g1-dropout-style"
  return cfg
