from mjlab.tasks.registry import register_mjlab_task
from src.tasks.velocity.rl import (
  DualStreamVelocityOnPolicyRunner,
  VelocityOnPolicyRunner,
)

from .env_cfgs import (
  unitree_g1_flat_env_cfg,
  unitree_g1_flat_energy_balanced_env_cfg,
  unitree_g1_flat_energy_conditioned_env_cfg,
  unitree_g1_flat_energy_eco_env_cfg,
  unitree_g1_flat_energy_performance_env_cfg,
  unitree_g1_flat_energy_unaware_env_cfg,
  unitree_g1_flat_full_dr_reset_env_cfg,
  unitree_g1_flat_joint_dropout_robust_env_cfg,
  unitree_g1_flat_joint_dropout_standard_env_cfg,
  unitree_g1_flat_joint_dropout_v2_stage1_env_cfg,
  unitree_g1_flat_joint_dropout_v2_stage2_env_cfg,
  unitree_g1_flat_joint_dropout_v2_stage3_env_cfg,
  unitree_g1_flat_joint_dropout_v2_stage4_env_cfg,
  unitree_g1_flat_overnight_dr_env_cfg,
  unitree_g1_flat_progressive_dr_env_cfg,
  unitree_g1_rough_env_cfg,
)
from .rl_cfg import unitree_g1_ppo_runner_cfg
from .rl_cfg import unitree_g1_dual_stream_ppo_runner_cfg
from .rl_cfg import unitree_g1_energy_morl_ppo_runner_cfg
from .rl_cfg import unitree_g1_joint_dropout_ppo_runner_cfg
from .style_dropout_cfg import STYLE_TASKS, style_dropout_env_cfg, style_dropout_rl_cfg
from src.tasks.velocity.rl.resource_limited_runner import ResourceLimitedVelocityRunner

for _style_task_id, _style_mode in STYLE_TASKS.values():
  register_mjlab_task(
    task_id=_style_task_id,
    env_cfg=style_dropout_env_cfg(_style_mode),
    play_env_cfg=style_dropout_env_cfg(_style_mode, play=True),
    rl_cfg=style_dropout_rl_cfg(),
    runner_cls=ResourceLimitedVelocityRunner,
  )

register_mjlab_task(
  task_id="Unitree-G1-Rough",
  env_cfg=unitree_g1_rough_env_cfg(),
  play_env_cfg=unitree_g1_rough_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Flat",
  env_cfg=unitree_g1_flat_env_cfg(),
  play_env_cfg=unitree_g1_flat_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Flat-Joint-Dropout-Standard",
  env_cfg=unitree_g1_flat_joint_dropout_standard_env_cfg(),
  play_env_cfg=unitree_g1_flat_joint_dropout_standard_env_cfg(play=True),
  rl_cfg=unitree_g1_joint_dropout_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Flat-Joint-Dropout-Robust",
  env_cfg=unitree_g1_flat_joint_dropout_robust_env_cfg(),
  play_env_cfg=unitree_g1_flat_joint_dropout_robust_env_cfg(play=True),
  rl_cfg=unitree_g1_joint_dropout_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Flat-Dual-Stream",
  env_cfg=unitree_g1_flat_env_cfg(),
  play_env_cfg=unitree_g1_flat_env_cfg(play=True),
  rl_cfg=unitree_g1_dual_stream_ppo_runner_cfg(),
  runner_cls=DualStreamVelocityOnPolicyRunner,
)

for stage, env_cfg_factory in (
  (1, unitree_g1_flat_joint_dropout_v2_stage1_env_cfg),
  (2, unitree_g1_flat_joint_dropout_v2_stage2_env_cfg),
  (3, unitree_g1_flat_joint_dropout_v2_stage3_env_cfg),
  (4, unitree_g1_flat_joint_dropout_v2_stage4_env_cfg),
):
  register_mjlab_task(
    task_id=f"Unitree-G1-Flat-Joint-Dropout-Robust-V2-Stage{stage}",
    env_cfg=env_cfg_factory(),
    play_env_cfg=env_cfg_factory(play=True),
    rl_cfg=unitree_g1_joint_dropout_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
  )

register_mjlab_task(
  task_id="Unitree-G1-Flat-Progressive-DR",
  env_cfg=unitree_g1_flat_progressive_dr_env_cfg(),
  play_env_cfg=unitree_g1_flat_progressive_dr_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Flat-Full-DR-Reset",
  env_cfg=unitree_g1_flat_full_dr_reset_env_cfg(),
  play_env_cfg=unitree_g1_flat_full_dr_reset_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Flat-Overnight-DR",
  env_cfg=unitree_g1_flat_overnight_dr_env_cfg(),
  play_env_cfg=unitree_g1_flat_overnight_dr_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)


register_mjlab_task(
  task_id="Unitree-G1-Flat-Energy-Performance",
  env_cfg=unitree_g1_flat_energy_performance_env_cfg(),
  play_env_cfg=unitree_g1_flat_energy_performance_env_cfg(play=True),
  rl_cfg=unitree_g1_energy_morl_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Flat-Energy-Balanced",
  env_cfg=unitree_g1_flat_energy_balanced_env_cfg(),
  play_env_cfg=unitree_g1_flat_energy_balanced_env_cfg(play=True),
  rl_cfg=unitree_g1_energy_morl_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Flat-Energy-Eco",
  env_cfg=unitree_g1_flat_energy_eco_env_cfg(),
  play_env_cfg=unitree_g1_flat_energy_eco_env_cfg(play=True),
  rl_cfg=unitree_g1_energy_morl_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Flat-Energy-Unaware",
  env_cfg=unitree_g1_flat_energy_unaware_env_cfg(),
  play_env_cfg=unitree_g1_flat_energy_unaware_env_cfg(play=True),
  rl_cfg=unitree_g1_energy_morl_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Flat-Energy-Conditioned",
  env_cfg=unitree_g1_flat_energy_conditioned_env_cfg(),
  play_env_cfg=unitree_g1_flat_energy_conditioned_env_cfg(play=True),
  rl_cfg=unitree_g1_energy_morl_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
