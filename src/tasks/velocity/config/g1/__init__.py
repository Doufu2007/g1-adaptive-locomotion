from mjlab.tasks.registry import register_mjlab_task
from src.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  unitree_g1_flat_env_cfg,
  unitree_g1_rough_env_cfg,
  unitree_g1_flat_phasor_env_cfg,
  unitree_g1_flat_no_phase_env_cfg,
  unitree_g1_flat_phasor_obs_env_cfg,
  unitree_g1_flat_phasor_reward_env_cfg,
  unitree_g1_flat_fixed_clock_env_cfg,
  unitree_g1_flat_actor_ablation_env_cfg,
  unitree_g1_flat_reward_ablation_env_cfg,
  unitree_g1_flat_joint_phase_env_cfg,
  unitree_g1_flat_random_vector_env_cfg,
  unitree_g1_rough_phasor_env_cfg,
)
from .rl_cfg import unitree_g1_ppo_runner_cfg

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
  task_id="Unitree-G1-Flat-No-Phase",
  env_cfg=unitree_g1_flat_no_phase_env_cfg(),
  play_env_cfg=unitree_g1_flat_no_phase_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Flat-PHASOR",
  env_cfg=unitree_g1_flat_phasor_env_cfg(),
  play_env_cfg=unitree_g1_flat_phasor_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Flat-PHASOR-Obs",
  env_cfg=unitree_g1_flat_phasor_obs_env_cfg(),
  play_env_cfg=unitree_g1_flat_phasor_obs_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Flat-PHASOR-Reward",
  env_cfg=unitree_g1_flat_phasor_reward_env_cfg(),
  play_env_cfg=unitree_g1_flat_phasor_reward_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Flat-Phase-FixedClock",
  env_cfg=unitree_g1_flat_fixed_clock_env_cfg(),
  play_env_cfg=unitree_g1_flat_fixed_clock_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Flat-Phase-ActorAblation",
  env_cfg=unitree_g1_flat_actor_ablation_env_cfg(),
  play_env_cfg=unitree_g1_flat_actor_ablation_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Flat-Phase-RewardAblation",
  env_cfg=unitree_g1_flat_reward_ablation_env_cfg(),
  play_env_cfg=unitree_g1_flat_reward_ablation_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Flat-Phase-Joint",
  env_cfg=unitree_g1_flat_joint_phase_env_cfg(),
  play_env_cfg=unitree_g1_flat_joint_phase_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Flat-Phase-RandomVector",
  env_cfg=unitree_g1_flat_random_vector_env_cfg(),
  play_env_cfg=unitree_g1_flat_random_vector_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Rough-PHASOR",
  env_cfg=unitree_g1_rough_phasor_env_cfg(),
  play_env_cfg=unitree_g1_rough_phasor_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
