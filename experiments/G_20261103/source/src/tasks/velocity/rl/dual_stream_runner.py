"""Runner dedicated to the isolated dual-stream G1 study."""

from copy import deepcopy

from mjlab.rl import RslRlVecEnvWrapper

from .dual_stream_model import compute_stream_config_from_env
from .runner import VelocityOnPolicyRunner


class DualStreamVelocityOnPolicyRunner(VelocityOnPolicyRunner):
  """Inject a dual-stream Actor while leaving the Critic unchanged."""

  env: RslRlVecEnvWrapper

  def __init__(
    self,
    env,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
  ):
    cfg = deepcopy(train_cfg)
    actor_cfg = cfg["actor"]
    if actor_cfg.get("class_name") not in (None, "MLPModel"):
      raise ValueError(
        "Dual-stream task refuses to replace a non-standard Actor configuration"
      )

    stream_config = compute_stream_config_from_env(
      env.unwrapped.observation_manager
    )
    if stream_config["input_dim"] != 98:
      raise ValueError(
        "Unitree-G1-Flat-Dual-Stream requires exactly 98 Actor observations, "
        f"got {stream_config['input_dim']}"
      )
    if stream_config["proprio_dim"] != 93 or stream_config["extero_dim"] != 5:
      raise ValueError(
        "Unitree-G1-Flat-Dual-Stream requires proprio=93D and extero=5D, "
        f"got proprio={stream_config['proprio_dim']}D and "
        f"extero={stream_config['extero_dim']}D"
      )

    actor_cfg["class_name"] = (
      "src.tasks.velocity.rl.dual_stream_model:DualStreamMLPModel"
    )
    actor_cfg["stream_config"] = stream_config
    super().__init__(env, cfg, log_dir, device)

