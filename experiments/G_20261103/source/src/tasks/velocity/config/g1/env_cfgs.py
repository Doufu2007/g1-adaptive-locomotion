"""Unitree G1 velocity environment configurations."""


import math
from src.assets.robots import (
  G1_ACTION_SCALE,
  get_g1_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, RayCastSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from src.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
import src.tasks.velocity.mdp as unitree_mdp


def unitree_g1_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 rough terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500
  cfg.sim.nconmax = 48

  cfg.scene.entities = {"robot": get_g1_robot_cfg()}

  # Set raycast sensor frame to G1 pelvis.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      sensor.frame.name = "pelvis"

  site_names = ("left_foot", "right_foot")
  geom_names = tuple(
    f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
  )

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    self_collision_cfg,
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = G1_ACTION_SCALE

  cfg.viewer.body_name = "torso_link"

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 1.15

  cfg.observations["critic"].terms["foot_height"].params[
    "asset_cfg"
  ].site_names = site_names

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

  # Rationale for std values:
  # - Knees/hip_pitch get the loosest std to allow natural leg bending during stride.
  # - Hip roll/yaw stay tighter to prevent excessive lateral sway and keep gait stable.
  # - Ankle roll is very tight for balance; ankle pitch looser for foot clearance.
  # - Waist roll/pitch stay tight to keep the torso upright and stable.
  # - Shoulders/elbows get moderate freedom for natural arm swing during walking.
  # - Wrists are loose (0.3) since they don't affect balance much.
  # Running values are ~1.5-2x walking values to accommodate larger motion range.
  cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
  cfg.rewards["pose"].params["std_walking"] = {
    # Lower body.
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.15,
    r".*hip_yaw.*": 0.15,
    r".*knee.*": 0.5,
    r".*ankle_pitch.*": 0.15,
    r".*ankle_roll.*": 0.1,
    # Waist.
    r".*waist_yaw.*": 0.15,
    r".*waist_roll.*": 0.1,
    r".*waist_pitch.*": 0.1,
    # Arms.
    r".*shoulder_pitch.*": 0.15,
    r".*shoulder_roll.*": 0.1,
    r".*shoulder_yaw.*": 0.1,
    r".*elbow.*": 0.1,
    r".*wrist.*": 0.1,
  }
  cfg.rewards["pose"].params["std_running"] = {
    # Lower body.
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.25,
    r".*hip_yaw.*": 0.25,
    r".*knee.*": 0.5,
    r".*ankle_pitch.*": 0.25,
    r".*ankle_roll.*": 0.1,
    # Waist.
    r".*waist_yaw.*": 0.25,
    r".*waist_roll.*": 0.1,
    r".*waist_pitch.*": 0.1,
    # Arms.
    r".*shoulder_pitch.*": 0.25,
    r".*shoulder_roll.*": 0.1,
    r".*shoulder_yaw.*": 0.1,
    r".*elbow.*": 0.1,
    r".*wrist.*": 0.1,
  }

  cfg.rewards["body_orientation_l2"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.rewards["foot_clearance"].params["asset_cfg"].site_names = site_names
  cfg.rewards["foot_slip"].params["asset_cfg"].site_names = site_names
  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
  )

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def unitree_g1_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 flat terrain velocity configuration."""
  cfg = unitree_g1_rough_env_cfg(play=play)

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  # Switch to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Remove raycast sensor and height scan (no terrain to scan).
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]

  # Disable terrain curriculum (not present in play mode since rough clears all).
  cfg.curriculum.pop("terrain_levels", None)

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-0.5, 1.0)
    twist_cmd.ranges.lin_vel_y = (-0.5, 0.5)
    twist_cmd.ranges.ang_vel_z = (-0.5, 0.5)

  return cfg


def unitree_g1_flat_joint_dropout_standard_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create the reward-locked standard arm for the dropout study."""
  return unitree_g1_flat_env_cfg(play=play)


def unitree_g1_flat_joint_dropout_robust_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create the reward-locked arm trained with joint target update loss."""
  cfg = unitree_g1_flat_env_cfg(play=play)
  base_action = cfg.actions["joint_pos"]
  assert isinstance(base_action, JointPositionActionCfg)
  cfg.actions["joint_pos"] = unitree_mdp.make_joint_target_dropout_action_cfg(
    base_action,
    enabled=not play,
  )

  metric_params = {"action_term_name": "joint_pos"}
  cfg.metrics["dropout_event_active"] = MetricsTermCfg(
    func=unitree_mdp.dropout_event_active,
    params=metric_params,
  )
  cfg.metrics["dropout_active_fraction"] = MetricsTermCfg(
    func=unitree_mdp.dropout_active_fraction,
    params=metric_params,
  )
  cfg.metrics["dropout_selected_joint_count"] = MetricsTermCfg(
    func=unitree_mdp.dropout_selected_joint_count,
    params=metric_params,
  )
  cfg.metrics["dropout_dropped_joint_count"] = MetricsTermCfg(
    func=unitree_mdp.dropout_dropped_joint_count,
    params=metric_params,
  )
  cfg.metrics["target_hold_error"] = MetricsTermCfg(
    func=unitree_mdp.target_hold_error,
    params=metric_params,
  )
  return cfg


def _unitree_g1_flat_joint_dropout_v2_env_cfg(
  *,
  play: bool,
  event_probability: float,
  dropout_probability_range: tuple[float, float],
  warmup_updates: int,
  ramp_updates: int,
) -> ManagerBasedRlEnvCfg:
  """Create a reward-locked Robust-v2 curriculum stage."""
  cfg = unitree_g1_flat_joint_dropout_robust_env_cfg(play=play)
  action_cfg = cfg.actions["joint_pos"]
  if not isinstance(action_cfg, unitree_mdp.JointTargetDropoutActionCfg):
    raise TypeError("Robust-v2 requires the joint target dropout action.")
  action_cfg.event_probability = event_probability
  action_cfg.dropout_probability_range = dropout_probability_range
  action_cfg.duration_steps_range = (5, 20)
  action_cfg.fault_joint_count_range = (1, 2)
  action_cfg.curriculum_warmup_steps = warmup_updates * 24
  action_cfg.curriculum_ramp_steps = ramp_updates * 24
  return cfg


def unitree_g1_flat_joint_dropout_v2_stage1_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Adapt at about 9 percent fault-active time."""
  return _unitree_g1_flat_joint_dropout_v2_env_cfg(
    play=play,
    event_probability=0.008,
    dropout_probability_range=(0.25, 0.75),
    warmup_updates=50,
    ramp_updates=100,
  )


def unitree_g1_flat_joint_dropout_v2_stage2_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Strengthen at about 11 percent fault-active time."""
  return _unitree_g1_flat_joint_dropout_v2_env_cfg(
    play=play,
    event_probability=0.010,
    dropout_probability_range=(0.35, 0.85),
    warmup_updates=0,
    ramp_updates=100,
  )


def unitree_g1_flat_joint_dropout_v2_stage3_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Stress at about 14 percent fault-active time."""
  return _unitree_g1_flat_joint_dropout_v2_env_cfg(
    play=play,
    event_probability=0.0125,
    dropout_probability_range=(0.50, 1.00),
    warmup_updates=0,
    ramp_updates=100,
  )


def unitree_g1_flat_joint_dropout_v2_stage4_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Maintain the capped long-run challenge at about 16 percent active time."""
  return _unitree_g1_flat_joint_dropout_v2_env_cfg(
    play=play,
    event_probability=0.015,
    dropout_probability_range=(0.50, 1.00),
    warmup_updates=0,
    ramp_updates=50,
  )


def unitree_g1_flat_progressive_dr_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create the G1 flat task with progressive domain randomization."""
  cfg = unitree_g1_flat_env_cfg(play=play)
  if play:
    return cfg

  stages = [
    {
      "step": 0,
      "friction": (0.8, 1.2),
      "encoder_bias": (-0.003, 0.003),
      "push_velocity": {
        "x": (-0.15, 0.15),
        "y": (-0.15, 0.15),
        "z": (-0.05, 0.05),
        "roll": (-0.10, 0.10),
        "pitch": (-0.10, 0.10),
        "yaw": (-0.15, 0.15),
      },
    },
    {
      "step": 500 * 24,
      "friction": (0.5, 1.4),
      "encoder_bias": (-0.008, 0.008),
      "push_velocity": {
        "x": (-0.30, 0.30),
        "y": (-0.30, 0.30),
        "z": (-0.20, 0.20),
        "roll": (-0.30, 0.30),
        "pitch": (-0.30, 0.30),
        "yaw": (-0.45, 0.45),
      },
    },
    {
      "step": 1200 * 24,
      "friction": (0.3, 1.6),
      "encoder_bias": (-0.015, 0.015),
      "push_velocity": {
        "x": (-0.5, 0.5),
        "y": (-0.5, 0.5),
        "z": (-0.4, 0.4),
        "roll": (-0.52, 0.52),
        "pitch": (-0.52, 0.52),
        "yaw": (-0.78, 0.78),
      },
    },
  ]

  # Resample physical parameters per episode using the current curriculum range.
  for event_name in ("foot_friction", "encoder_bias"):
    cfg.events[event_name].mode = "reset"
  cfg.events["foot_friction"].params["ranges"] = stages[0]["friction"]
  cfg.events["encoder_bias"].params["bias_range"] = stages[0]["encoder_bias"]
  cfg.events["push_robot"].params["velocity_range"] = dict(
    stages[0]["push_velocity"]
  )
  cfg.curriculum["domain_randomization"] = CurriculumTermCfg(
    func=unitree_mdp.domain_randomization,
    params={"stages": stages},
  )

  return cfg


def unitree_g1_flat_full_dr_reset_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create an ablation task that resamples the full DR range every episode."""
  cfg = unitree_g1_flat_env_cfg(play=play)
  if play:
    return cfg

  for event_name in ("foot_friction", "encoder_bias"):
    cfg.events[event_name].mode = "reset"
  return cfg


def unitree_g1_flat_overnight_dr_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create the weighted cyclic DR task used for unattended refinement."""
  cfg = unitree_g1_flat_progressive_dr_env_cfg(play=play)
  if play:
    return cfg

  curriculum = cfg.curriculum["domain_randomization"]
  stages = [dict(stage) for stage in curriculum.params["stages"]]
  # One 300-update cycle spends 20%, 30%, and 50% in easy, medium, and hard DR.
  for stage, start_update in zip(stages, (0, 60, 150), strict=True):
    stage["step"] = start_update * 24
  cfg.curriculum["domain_randomization"] = CurriculumTermCfg(
    func=unitree_mdp.cyclic_domain_randomization,
    params={"stages": stages, "cycle_steps": 300 * 24},
  )
  return cfg


def _unitree_g1_flat_energy_morl_env_cfg(
  play: bool,
  preference_mode: unitree_mdp.PreferenceMode,
  fixed_preference: float,
) -> ManagerBasedRlEnvCfg:
  """Create an isolated, preference-conditioned energy study task."""
  cfg = unitree_g1_flat_env_cfg(play=play)

  if not play:
    # Keep a fixed, mild randomization distribution across every study arm.
    cfg.events["foot_friction"].mode = "reset"
    cfg.events["foot_friction"].params["ranges"] = (0.8, 1.2)
    cfg.events["encoder_bias"].mode = "reset"
    cfg.events["encoder_bias"].params["bias_range"] = (-0.003, 0.003)
    cfg.events["base_com"].params["ranges"] = {
      axis: (-0.02, 0.02) for axis in range(3)
    }
    cfg.events["push_robot"].params["velocity_range"] = {
      "x": (-0.15, 0.15),
      "y": (-0.15, 0.15),
      "z": (-0.05, 0.05),
      "roll": (-0.10, 0.10),
      "pitch": (-0.10, 0.10),
      "yaw": (-0.15, 0.15),
    }

  switch_start_step = 0 if play else 500 * 24
  multi_switch_start_step = 0 if play else 1200 * 24
  cfg.commands["energy_preference"] = unitree_mdp.EnergyPreferenceCommandCfg(
    mode=preference_mode,
    fixed_preference=fixed_preference,
    anchors=(0.0, 0.5, 1.0),
    resampling_time_range=(4.0, 8.0),
    ramp_time_s=0.5,
    switch_start_step=switch_start_step,
    multi_switch_start_step=multi_switch_start_step,
  )

  preference_observation = ObservationTermCfg(
    func=unitree_mdp.generated_commands,
    params={"command_name": "energy_preference"},
  )
  cfg.observations["actor"].terms["energy_preference"] = preference_observation
  cfg.observations["critic"].terms["energy_preference"] = ObservationTermCfg(
    func=unitree_mdp.generated_commands,
    params={"command_name": "energy_preference"},
  )

  tracking_params = {
    "command_name": "twist",
    "preference_command_name": "energy_preference",
    "minimum_tracking_scale": 0.7,
  }
  cfg.rewards["track_linear_velocity"] = RewardTermCfg(
    func=unitree_mdp.preference_scaled_track_linear_velocity,
    weight=1.0,
    params={**tracking_params, "std": 0.5},
  )
  cfg.rewards["track_angular_velocity"] = RewardTermCfg(
    func=unitree_mdp.preference_scaled_track_angular_velocity,
    weight=1.0,
    params={**tracking_params, "std": math.sqrt(0.5)},
  )
  cfg.rewards["mechanical_power"] = RewardTermCfg(
    func=unitree_mdp.preference_weighted_mechanical_power,
    weight=-0.2,
    params={
      "preference_command_name": "energy_preference",
      "reference_speed": 1.0,
      "clip_max": 5.0,
    },
  )

  cfg.metrics["energy_preference"] = MetricsTermCfg(
    func=unitree_mdp.energy_preference,
    params={"command_name": "energy_preference"},
  )
  cfg.metrics["mechanical_power_w"] = MetricsTermCfg(
    func=unitree_mdp.actuator_mechanical_power,
  )
  cfg.metrics["normalized_mechanical_power"] = MetricsTermCfg(
    func=unitree_mdp.normalized_actuator_mechanical_power,
    params={"reference_speed": 1.0, "clip_max": 5.0},
  )
  cfg.metrics["command_completion"] = MetricsTermCfg(
    func=unitree_mdp.tracking_command_completion,
    params={"command_name": "twist"},
  )

  return cfg


def unitree_g1_flat_energy_performance_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  return _unitree_g1_flat_energy_morl_env_cfg(play, "fixed", 1.0)


def unitree_g1_flat_energy_balanced_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  return _unitree_g1_flat_energy_morl_env_cfg(play, "fixed", 0.5)


def unitree_g1_flat_energy_eco_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  return _unitree_g1_flat_energy_morl_env_cfg(play, "fixed", 0.0)


def unitree_g1_flat_energy_unaware_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  return _unitree_g1_flat_energy_morl_env_cfg(play, "unaware", 0.5)


def unitree_g1_flat_energy_conditioned_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  return _unitree_g1_flat_energy_morl_env_cfg(play, "conditioned", 0.5)
