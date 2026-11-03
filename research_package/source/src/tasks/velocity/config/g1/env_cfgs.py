"""Unitree G1 velocity environment configurations."""

from functools import lru_cache
import json
import os
from pathlib import Path

from src.assets.robots import (
  G1_ACTION_SCALE,
  get_g1_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, RayCastSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from src.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg


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

    # Rough terrain creates many contacts — let mujoco_warp auto-size nconmax
    # based on actual mjd.ncon instead of the hardcoded 48 (which overflows).
    cfg.sim.nconmax = None

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


def unitree_g1_flat_phasor_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Flat terrain with PHASOR in both actor observation and gait reward."""
    cfg = unitree_g1_flat_env_cfg(play=play)
    return _configure_flat_phasor(cfg, actor_phase=True, reward_phase=True)


def unitree_g1_flat_no_phase_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Flat baseline reward with a zeroed actor phase channel."""
    from mjlab.managers.observation_manager import ObservationTermCfg
    import src.tasks.velocity.mdp as src_mdp

    cfg = unitree_g1_flat_env_cfg(play=play)
    cfg.observations["actor"].terms["phase"] = ObservationTermCfg(
        func=src_mdp.zero_phase,
        params={"command_name": "twist"},
    )
    return cfg


def unitree_g1_flat_phasor_obs_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Flat terrain with learned actor phase and the fixed-clock reward."""
    cfg = unitree_g1_flat_env_cfg(play=play)
    return _configure_flat_phasor(cfg, actor_phase=True, reward_phase=False)


def unitree_g1_flat_phasor_reward_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Flat terrain with fixed-clock actor input and learned-phase reward."""
    cfg = unitree_g1_flat_env_cfg(play=play)
    return _configure_flat_phasor(cfg, actor_phase=False, reward_phase=True)


def unitree_g1_flat_fixed_clock_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Pre-registered fixed-clock baseline (actor, reward, and critic)."""
    return unitree_g1_flat_env_cfg(play=play)


def unitree_g1_flat_actor_ablation_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Actor ablation: learned phase for actor, fixed clock for reward/critic."""
    return unitree_g1_flat_phasor_obs_env_cfg(play=play)


def unitree_g1_flat_reward_ablation_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Reward ablation: learned phase for reward, fixed clock for actor/critic."""
    return unitree_g1_flat_phasor_reward_env_cfg(play=play)


def unitree_g1_flat_joint_phase_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Joint method: learned phase drives actor and contact reward."""
    return unitree_g1_flat_phasor_env_cfg(play=play)


def unitree_g1_flat_random_vector_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Random-vector control: episode-fixed unit vector, fixed clock reward."""
    from mjlab.managers.observation_manager import ObservationTermCfg
    import src.tasks.velocity.mdp as src_mdp

    cfg = unitree_g1_flat_env_cfg(play=play)
    cfg.observations["actor"].terms["phase"] = ObservationTermCfg(
        func=src_mdp.random_episode_phase,
        params={},
    )
    cfg.events["reset_episode_random_phase"] = EventTermCfg(
        func=src_mdp.reset_random_phase,
        mode="reset",
        params={},
    )
    return cfg


def _configure_flat_phasor(
    cfg: ManagerBasedRlEnvCfg,
    *,
    actor_phase: bool,
    reward_phase: bool,
) -> ManagerBasedRlEnvCfg:
    from mjlab.managers.observation_manager import ObservationTermCfg
    from mjlab.managers.reward_manager import RewardTermCfg
    import src.tasks.velocity.mdp as src_mdp

    artifacts = _g1_phase_runtime_artifacts()
    encoder_path = artifacts["encoder_path"]
    reward_template = artifacts["reward_template"]

    cfg.events["reset_phasor_phase_history"] = EventTermCfg(
        func=src_mdp.reset_learned_phase_history,
        mode="reset",
        params={"encoder_path": encoder_path},
    )

    if actor_phase:
        cfg.observations["actor"].terms["phase"] = ObservationTermCfg(
            func=src_mdp.learned_phase,
            params={"encoder_path": encoder_path},
        )

    if reward_phase:
        cfg.rewards["foot_gait"] = RewardTermCfg(
            func=src_mdp.feet_gait_learned,
            weight=0.5,
            params={
                "offset": list(reward_template["offset_cycles"]),
                "threshold": float(reward_template["stance_threshold"]),
                "command_threshold": 0.1,
                "command_name": "twist",
                "sensor_name": "feet_ground_contact",
                "encoder_path": encoder_path,
            },
        )

    return cfg


def unitree_g1_rough_phasor_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Rough terrain with learned phase (PHASOR) replacing fixed 0.6s CPG."""
    from mjlab.managers.observation_manager import ObservationTermCfg
    from mjlab.managers.reward_manager import RewardTermCfg
    import src.tasks.velocity.mdp as src_mdp

    cfg = unitree_g1_rough_env_cfg(play=play)
    artifacts = _g1_phase_runtime_artifacts()
    encoder_path = artifacts["encoder_path"]
    reward_template = artifacts["reward_template"]

    cfg.events["reset_phasor_phase_history"] = EventTermCfg(
        func=src_mdp.reset_learned_phase_history,
        mode="reset",
        params={"encoder_path": encoder_path},
    )

    cfg.observations["actor"].terms["phase"] = ObservationTermCfg(
        func=src_mdp.learned_phase,
        params={"encoder_path": encoder_path},
    )

    cfg.rewards["foot_gait"] = RewardTermCfg(
        func=src_mdp.feet_gait_learned,
        weight=0.5,
        params={
            "offset": list(reward_template["offset_cycles"]),
            "threshold": float(reward_template["stance_threshold"]),
            "command_threshold": 0.1,
            "command_name": "twist",
            "sensor_name": "feet_ground_contact",
            "encoder_path": encoder_path,
        },
    )

    return cfg


@lru_cache(maxsize=1)
def _g1_phase_runtime_artifacts() -> dict[str, object]:
    """Resolve the frozen encoder and train-only reward template."""
    from src.tasks.velocity.phasor.data import G1_JOINT_ORDER

    root = Path(__file__).resolve().parents[5]
    artifact_dir = root / "results" / "g1_phase_dataset_20261019"
    # Default runtime encoder is the 2026-10-21 architecture-sweep TCN
    # (gate 0.4793 vs 0.4163 for the v1 CNN, both on the 603-cycle validation
    # set).  It is still below the 0.90 bar, so it does not authorise a PPO run.
    # Set G1_PHASOR_ENCODER_RUNTIME=phase_encoder_runtime_v1.pt to roll back.
    encoder_name = os.environ.get(
        "G1_PHASOR_ENCODER_RUNTIME", "phase_encoder_runtime_tcn_20261021.pt"
    )
    encoder_path = (artifact_dir / encoder_name).resolve(strict=True)
    template_path = artifact_dir / "g1_reward_template_v1.json"
    template = json.loads(template_path.read_text(encoding="utf-8"))
    if template.get("schema") != "g1.learned_phase_reward_template.v1":
        raise ValueError("unsupported G1 learned-phase reward template")
    if template.get("status") != "FROZEN" or template.get("fit_split") != "train":
        raise ValueError("G1 learned-phase reward template must be frozen from train")
    if tuple(template.get("joint_order", ())) != G1_JOINT_ORDER:
        raise ValueError("reward template joint order does not match the G1 encoder")
    offset = template.get("offset_cycles")
    threshold = template.get("stance_threshold")
    if not isinstance(offset, list) or len(offset) != 2:
        raise ValueError("reward template must define two leg phase offsets")
    if not all(isinstance(value, (int, float)) for value in offset):
        raise ValueError("reward offsets must be numeric cycle fractions")
    if not isinstance(threshold, (int, float)) or not 0.0 < float(threshold) < 1.0:
        raise ValueError("stance threshold must be a cycle fraction between zero and one")
    return {"encoder_path": str(encoder_path), "reward_template": template}
