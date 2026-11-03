"""Dynamically audit the five registered G1 phase experiment configurations."""

from __future__ import annotations

import hashlib
import json
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "results" / "g1_phase_dataset_20261019"
CHECKPOINT = ARTIFACT_DIR / "phase_encoder_runtime_v1.pt"
TEMPLATE = ARTIFACT_DIR / "g1_reward_template_v1.json"
OUTPUT = ARTIFACT_DIR / "config_audit_runtime_v1.json"

GROUPS = {
    "fixed_clock": ("Unitree-G1-Flat-Phase-FixedClock", "unitree_g1_flat_fixed_clock_env_cfg"),
    "actor_ablation": ("Unitree-G1-Flat-Phase-ActorAblation", "unitree_g1_flat_actor_ablation_env_cfg"),
    "reward_ablation": ("Unitree-G1-Flat-Phase-RewardAblation", "unitree_g1_flat_reward_ablation_env_cfg"),
    "joint_phase": ("Unitree-G1-Flat-Phase-Joint", "unitree_g1_flat_joint_phase_env_cfg"),
    "random_vector": ("Unitree-G1-Flat-Phase-RandomVector", "unitree_g1_flat_random_vector_env_cfg"),
}
EXPECTED_MODES = {
    "fixed_clock": ("fixed_clock", "fixed_clock"),
    "actor_ablation": ("learned", "fixed_clock"),
    "reward_ablation": ("fixed_clock", "learned"),
    "joint_phase": ("learned", "learned"),
    "random_vector": ("random_vector", "fixed_clock"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable(value: Any) -> Any:
    if callable(value):
        return f"{value.__module__}.{value.__qualname__}"
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, Enum):
        return _stable(value.value)
    if is_dataclass(value):
        return {field.name: _stable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, dict):
        return {str(key): _stable(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (tuple, list)):
        return [_stable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        try:
            return _stable(value.item())
        except (TypeError, ValueError):
            pass
    return repr(value)


def _digest(value: Any) -> str:
    payload = json.dumps(_stable(value), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _term_signature(cfg: Any) -> dict[str, Any]:
    return {
        "func": _stable(cfg.func),
        "params": _stable(cfg.params),
        "weight": _stable(getattr(cfg, "weight", None)),
    }


def _common_signature(cfg: Any) -> dict[str, Any]:
    return {
        "actor_term_names": sorted(cfg.observations["actor"].terms),
        "critic_term_names": sorted(cfg.observations["critic"].terms),
        "reward_term_names": sorted(cfg.rewards),
        "scene_num_envs": cfg.scene.num_envs,
        "mujoco_timestep": cfg.sim.mujoco.timestep,
        "decimation": cfg.decimation,
        "commands": _stable(cfg.commands),
    }


def _config_signature(cfg: Any) -> dict[str, Any]:
    return {
        "common": _common_signature(cfg),
        "actor_phase": _term_signature(cfg.observations["actor"].terms["phase"]),
        "critic_phase": _term_signature(cfg.observations["critic"].terms["phase"]),
        "foot_gait": _term_signature(cfg.rewards["foot_gait"]),
        "events": {
            name: {
                "func": _stable(event.func),
                "mode": _stable(event.mode),
                "params": _stable(event.params),
            }
            for name, event in sorted(cfg.events.items())
            if name in {"reset_phasor_phase_history", "reset_episode_random_phase"}
        },
    }


def audit_configurations(*, write_report: bool = True) -> dict[str, Any]:
    """Inspect registered configs, their factories, and the frozen artifacts."""
    import src.tasks.velocity.config.g1  # noqa: F401 - registers task IDs
    from mjlab.tasks.registry import _REGISTRY
    from src.tasks.velocity.config.g1 import env_cfgs
    import src.tasks.velocity.mdp as src_mdp

    from scripts.prepare_g1_phase_runtime_checkpoint import sha256_file as file_hash

    checkpoint_path = CHECKPOINT.resolve(strict=True)
    template_path = TEMPLATE.resolve(strict=True)
    template = json.loads(template_path.read_text(encoding="utf-8"))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_sha = file_hash(checkpoint_path)
    template_sha = file_hash(template_path)
    contract = checkpoint.get("runtime_contract", {})

    checks: list[dict[str, Any]] = []

    def record(name: str, passed: bool, detail: Any = None) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": _stable(detail)})

    record("template_frozen_from_train", template.get("status") == "FROZEN" and template.get("fit_split") == "train")
    record("checkpoint_joint_order", checkpoint.get("joint_names") == contract.get("joint_order"))
    record("checkpoint_template_hash", contract.get("reward_template_sha256") == template_sha)
    record("checkpoint_dataset_hash", checkpoint.get("dataset_sha256") == template.get("dataset_sha256"))
    record("checkpoint_manifest_hash", checkpoint.get("data_manifest_sha256") == template.get("provenance_manifest_sha256"))
    record("checkpoint_input_shape", checkpoint.get("joint_dim") == 29 and checkpoint.get("window") == 10)

    rows: dict[str, Any] = {}
    budget_hashes: set[str] = set()
    common_hashes: set[str] = set()
    for name, (task_id, builder_name) in GROUPS.items():
        entry = _REGISTRY.get(task_id)
        registered = entry is not None
        expected_actor, expected_reward = EXPECTED_MODES[name]
        factory_cfg = getattr(env_cfgs, builder_name)()
        registered_cfg = entry.env_cfg if entry is not None else factory_cfg
        registered_rl_cfg = entry.rl_cfg if entry is not None else None
        actor_term = registered_cfg.observations["actor"].terms["phase"]
        critic_term = registered_cfg.observations["critic"].terms["phase"]
        reward_term = registered_cfg.rewards["foot_gait"]
        actor_func = actor_term.func
        reward_func = reward_term.func

        if actor_func is src_mdp.learned_phase:
            actor_mode = "learned"
        elif actor_func is src_mdp.random_episode_phase:
            actor_mode = "random_vector"
        elif actor_func is src_mdp.phase:
            actor_mode = "fixed_clock"
        else:
            actor_mode = "unexpected"
        reward_mode = "learned" if reward_func is src_mdp.feet_gait_learned else (
            "fixed_clock" if reward_func is src_mdp.feet_gait else "unexpected"
        )
        critic_fixed_clock = (
            critic_term.func is src_mdp.phase
            and float(critic_term.params.get("period", -1.0)) == 0.6
            and critic_term.params.get("command_name") == "twist"
        )

        learned = expected_actor == "learned" or expected_reward == "learned"
        reset_event = registered_cfg.events.get("reset_phasor_phase_history")
        reset_consistent = (
            reset_event is not None
            and reset_event.func is src_mdp.reset_learned_phase_history
            and Path(reset_event.params.get("encoder_path", "")).resolve() == checkpoint_path
        ) if learned else reset_event is None
        random_reset = registered_cfg.events.get("reset_episode_random_phase")
        random_reset_consistent = (
            random_reset is not None
            and random_reset.func is src_mdp.reset_random_phase
        ) if expected_actor == "random_vector" else random_reset is None

        actor_path = actor_term.params.get("encoder_path") if actor_mode == "learned" else None
        reward_path = reward_term.params.get("encoder_path") if reward_mode == "learned" else None
        paths_consistent = all(
            value is None or Path(value).resolve() == checkpoint_path
            for value in (actor_path, reward_path)
        )
        reward_template_consistent = True
        if reward_mode == "learned":
            reward_template_consistent = (
                reward_term.params.get("offset") == template.get("offset_cycles")
                and float(reward_term.params.get("threshold", -1.0)) == float(template.get("stance_threshold", -2.0))
            )

        factory_matches = _config_signature(factory_cfg) == _config_signature(registered_cfg)
        budget_hash = _digest(registered_rl_cfg)
        common_hash = _digest(_common_signature(registered_cfg))
        budget_hashes.add(budget_hash)
        common_hashes.add(common_hash)
        row = {
            "task_id": task_id,
            "factory": builder_name,
            "registered": registered,
            "factory_config_matches_registered": factory_matches,
            "actor_mode": actor_mode,
            "reward_mode": reward_mode,
            "critic_fixed_clock": critic_fixed_clock,
            "reset_consistent": reset_consistent,
            "random_reset_consistent": random_reset_consistent,
            "checkpoint_path_consistent": paths_consistent,
            "reward_template_consistent": reward_template_consistent,
            "joint_dim": checkpoint.get("joint_dim"),
            "window": checkpoint.get("window"),
            "checkpoint_sha256": checkpoint_sha,
            "reward_template_sha256": template_sha,
            "training_budget_sha256": budget_hash,
            "common_config_sha256": common_hash,
        }
        rows[name] = row
        record(f"{name}.registered", registered, task_id)
        record(f"{name}.factor_matrix", (actor_mode, reward_mode) == (expected_actor, expected_reward), row)
        record(f"{name}.factory_matches_registered", factory_matches)
        record(f"{name}.critic_fixed_clock", critic_fixed_clock)
        record(f"{name}.reset_consistent", reset_consistent)
        record(f"{name}.random_reset_consistent", random_reset_consistent)
        record(f"{name}.checkpoint_path_consistent", paths_consistent)
        record(f"{name}.reward_template_consistent", reward_template_consistent)

    record("shared_training_budget", len(budget_hashes) == 1, sorted(budget_hashes))
    record("shared_common_environment", len(common_hashes) == 1, sorted(common_hashes))

    source_files = {
        "env_cfgs_sha256": ROOT / "src" / "tasks" / "velocity" / "config" / "g1" / "env_cfgs.py",
        "observations_sha256": ROOT / "src" / "tasks" / "velocity" / "mdp" / "observations.py",
        "rewards_sha256": ROOT / "src" / "tasks" / "velocity" / "mdp" / "rewards.py",
        "audit_script_sha256": Path(__file__).resolve(),
    }
    report = {
        "schema": "g1.phase_config_audit.v2",
        "status": "PASS" if all(check["passed"] for check in checks) else "FAIL",
        "groups": rows,
        "checks": checks,
        "artifacts": {
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha,
            "reward_template_path": str(template_path),
            "reward_template_sha256": template_sha,
            "dataset_sha256": checkpoint.get("dataset_sha256"),
            "manifest_sha256": checkpoint.get("data_manifest_sha256"),
            "joint_dim": checkpoint.get("joint_dim"),
            "window": checkpoint.get("window"),
        },
        "source_sha256": {name: sha256_file(path) for name, path in source_files.items()},
        "formal_ppo_started": False,
    }
    if write_report:
        OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    report = audit_configurations()
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
