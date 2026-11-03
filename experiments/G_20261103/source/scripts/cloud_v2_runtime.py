"""GPU worker adapters for the isolated v2 robot-control study."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
import math
import os
from pathlib import Path
import time

from scripts.cloud_v2_campaign import (ROOT, CONTROL, REFERENCE, AUDIT, TRAIN_END, EVAL_END,
    stamp, epoch, publish, read, sha, manual_stop_paths, directory_for)


def severity(common_step, origin=2000 * 24, ramp=500 * 24):
    return min(1., max(0., (common_step - origin) / ramp))


def severity_bounds(common_step, cfg=None):
    origin = cfg.extension_origin_steps if cfg is not None else 2000 * 24
    ramp = cfg.extension_ramp_steps if cfg is not None else 500 * 24
    f = severity(common_step, origin, ramp)
    start_p = cfg.dropout_probability_range[1] if cfg is not None else .75
    end_p = cfg.extension_max_probability if cfg is not None else 1.
    start_d = cfg.duration_steps_range[1] if cfg is not None else 20
    end_d = cfg.extension_max_duration_steps if cfg is not None else 40
    return start_p + (end_p - start_p) * f, start_d + int(round((end_d - start_d) * f))


def limits():
    from src.tasks.velocity.rl.resource_control import ResourceLimits
    return ResourceLimits(cpu_threads=3, duty_cycle=1, minimum_rest_s=0,
                          maximum_gpu_memory_mib=28672, minimum_available_ram_mib=8192,
                          minimum_host_available_ram_mib=8192, minimum_start_host_available_ram_mib=12288)


def guard(directory, deadline):
    from src.tasks.velocity.rl.resource_control import TrainingStopRequested
    paths = manual_stop_paths(directory)
    if paths:
        raise TrainingStopRequested(f"User stop remains active: {paths}", category="manual")
    if time.time() >= epoch(deadline):
        raise TrainingStopRequested("Pre-authorized rental deadline reached; cooperative checkpoint", category="deadline")


def verify_parent_config(cfg, audited):
    import yaml
    import copy
    normalized = lambda data: yaml.load(yaml.dump(data), Loader=yaml.BaseLoader)
    current = normalized(asdict(cfg.agent))
    saved = copy.deepcopy(audited["agent"])
    # Saved configs omit consumed selectors and unused optional model fields.
    # Normalize only the exact verified defaults; retain every non-null value.
    for component, expected in (("actor", "MLPModel"), ("critic", "MLPModel"), ("algorithm", "PPO")):
        if "class_name" not in saved[component] and "class_name" in current[component]:
            assert current[component].pop("class_name") == expected
    for component in ("actor", "critic"):
        if "cnn_cfg" not in saved[component] and "cnn_cfg" in current[component]:
            assert current[component].pop("cnn_cfg") == "null"
    if "distribution_cfg" not in saved["critic"] and "distribution_cfg" in current["critic"]:
        assert current["critic"].pop("distribution_cfg") == "null"
    if "share_cnn_encoders" not in saved["algorithm"] and "share_cnn_encoders" in current["algorithm"]:
        assert current["algorithm"].pop("share_cnn_encoders") == "false"
    for key in ("rnd_cfg", "symmetry_cfg"):
        if key not in current["algorithm"] and key in saved["algorithm"]:
            assert saved["algorithm"][key] == "null"
            current["algorithm"][key] = "null"
    distribution = current["actor"]["distribution_cfg"]
    if "class_name" not in saved["actor"]["distribution_cfg"] and "class_name" in distribution:
        assert distribution.pop("class_name") == "GaussianDistribution"
    for key in ("actor", "critic", "algorithm", "num_steps_per_env", "clip_actions", "obs_groups"):
        if current[key] != saved[key]:
            raise RuntimeError(f"Saved agent config differs at {key}: current={current[key]} saved={saved[key]}")
    # Scene.__init__ synchronizes this before the original launcher dumps YAML.
    cfg.env.scene.terrain.num_envs = cfg.env.scene.num_envs
    current_env = normalized(asdict(cfg.env))
    saved_env = yaml.load((Path(audited["checkpoint"]).parent / "params/env.yaml").read_text(), Loader=yaml.BaseLoader)
    for e in (current_env, saved_env):
        e["commands"]["gait_reference"]["reference_file"] = "REFERENCE_PATH_RESOLVED_BY_FROZEN_SHA256"
    if current_env != saved_env:
        def differences(a, b, path=""):
            if isinstance(a, dict) and isinstance(b, dict):
                return [d for k in sorted(set(a) | set(b)) for d in differences(a.get(k), b.get(k), path + "/" + k)]
            return [] if a == b else [{"path": path, "current": a, "saved": b}]
        raise RuntimeError("Saved environment differs: " + json.dumps(differences(current_env, saved_env)[:12], ensure_ascii=False))


def sample_expanded_events(cfg, common_step, num_envs, device):
    import torch
    high, maximum = severity_bounds(common_step, cfg)
    low = cfg.dropout_probability_range[0]
    probabilities = low + (high - low) * torch.rand(num_envs, device=device)
    full = torch.rand(num_envs, device=device) < cfg.extension_full_loss_long_event_mass * severity(
        common_step, cfg.extension_origin_steps, cfg.extension_ramp_steps)
    probabilities = torch.where(full, torch.ones_like(probabilities), probabilities)
    durations = torch.randint(cfg.duration_steps_range[0], maximum + 1, (num_envs,), device=device)
    durations = torch.where(full, torch.full_like(durations, maximum), durations)
    return probabilities, durations, full


def extend_action(base):
    from src.tasks.velocity.mdp.action_dropout import JointTargetDropoutActionCfg, JointTargetDropoutAction
    import torch

    class ExtendedAction(JointTargetDropoutAction):
        def _sample_fault_probability(self, start_mask, factor):
            low = self.cfg.dropout_probability_range[0]
            values, durations, full = sample_expanded_events(self.cfg, self._env.common_step_counter, self.num_envs, self.device)
            if not hasattr(self, "_full_loss_event"):
                self._full_loss_event = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
            self._full_loss_event[start_mask] = full[start_mask]
            self._extended_durations = durations
            self._fault_probability[start_mask] = (low + (values - low) * factor)[start_mask]

        def _sample_fault_duration(self, start_mask, factor):
            # v2 always resumes after the original curriculum has saturated.
            assert factor == 1., "Expanded distribution requires the verified resumed curriculum"
            self._fault_remaining[start_mask] = self._extended_durations[start_mask] - 1

    @dataclass(kw_only=True)
    class ExtendedCfg(JointTargetDropoutActionCfg):
        extension_origin_steps: int = 2000 * 24
        extension_ramp_steps: int = 500 * 24
        extension_max_probability: float = 1.
        extension_max_duration_steps: int = 40
        extension_full_loss_long_event_mass: float = .25

        def build(self, env):
            return ExtendedAction(self, env)

    return ExtendedCfg(**{f.name: getattr(base, f.name) for f in fields(base)})


def train_worker(value, branch, index):
    import torch
    import yaml
    import mjlab.tasks  # noqa: F401
    import src.tasks  # noqa: F401
    from src.tasks.velocity.config.g1.style_dropout_cfg import STYLE_TASKS
    from src.tasks.velocity.mdp.style_reference import load_gait_reference
    from src.tasks.velocity.rl.resource_control import ResourceController, TrainingStopRequested
    from src.tasks.velocity.rl.resource_limited_runner import ResourceLimitedVelocityRunner
    from scripts.style_resume import stage_private_checkpoint, saved_iteration
    from scripts.run_style_dropout import configure_reference
    from scripts import train

    parent = value["parents"][index]
    directory = directory_for(branch, "train", parent)
    log_dir = ROOT / "logs/rsl_rl" / f"g1_dropout_style_v2_{branch}" / f"{parent['group']}_seed{parent['seed']}"
    directory.mkdir(parents=True, exist_ok=True)
    result = {"at_utc": stamp(), "branch": branch, **parent, "protocol_sha256": value["sha256"], "status": "preparing"}
    publish(directory / "result.json", result)
    try:
        guard(directory, TRAIN_END)
        if log_dir.exists():
            raise FileExistsError(f"No automatic overwrite/resume: {log_dir}")
        assert torch.cuda.is_available()
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        resource_limits = limits()
        os.environ.update(G1_RESOURCE_LIMITS_JSON=json.dumps(asdict(resource_limits)), G1_RESOURCE_DIRECTORY=str(directory),
                          G1_ACTIVE_STYLE_GROUP=parent["group"], G1_ACTIVE_STYLE_SEED=str(parent["seed"]),
                          G1_RUN_PURPOSE="DEVELOPMENT_TRAINING_V2_INDEPENDENT_CONTINUATION")
        torch.set_num_threads(3)
        original_ensure = ResourceController.ensure_continue

        def ensure(self):
            original_ensure(self)
            guard(directory, TRAIN_END)

        ResourceController.ensure_continue = ensure
        gate = ResourceController(resource_limits, directory, directory)
        gate.before_allocation(4000)
        checkpoint = Path(parent["checkpoint"])
        assert sha(checkpoint) == parent["sha256"] and saved_iteration(checkpoint) == 1999
        snapshot = stage_private_checkpoint(checkpoint)
        task = STYLE_TASKS[parent["group"]][0]
        cfg = train.TrainConfig.from_task(task)
        reference = load_gait_reference(REFERENCE)
        configure_reference(cfg.env, reference)
        cfg.env.scene.num_envs = 1024
        cfg.agent.seed = cfg.env.seed = parent["seed"]
        cfg.env.sim.nan_guard.enabled = True
        audited = next(m for m in read(AUDIT)["models"] if (m["group"], m["seed"]) == (parent["group"], parent["seed"]))
        verify_parent_config(cfg, audited)
        if branch == "F":
            cfg.env.actions["joint_pos"] = extend_action(cfg.env.actions["joint_pos"])
        cfg.agent.max_iterations = 4000
        cfg.agent.save_interval = 100
        cfg.agent.experiment_name = f"g1_dropout_style_v2_{branch}"
        cfg.agent.run_name = f"v2_{branch}_{parent['group']}_seed{parent['seed']}"
        cfg.agent.logger = "tensorboard"
        cfg.agent.upload_model = False
        cfg.agent.resume = True

        class AuditedRunner(ResourceLimitedVelocityRunner):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                original_load = self.alg.load

                def checked_load(checkpoint_dict, **load_args):
                    if int(checkpoint_dict["iter"]) != 1999:
                        raise RuntimeError("Expected precisely 2000 completed parent updates")
                    loaded = original_load(checkpoint_dict, **load_args)
                    actual = self.alg.optimizer.state_dict()
                    expected = checkpoint_dict["optimizer_state_dict"]
                    assert len(actual["state"]) == len(expected["state"]) > 0
                    assert actual["param_groups"] == expected["param_groups"]
                    for component in ("actor", "critic"):
                        model_state = getattr(self.alg, component).state_dict()
                        for key, tensor in checkpoint_dict[f"{component}_state_dict"].items():
                            assert torch.equal(model_state[key].to("cpu"), tensor.to("cpu")), f"{component}/{key} failed exact restore"
                    for key, state in expected["state"].items():
                        for name, tensor in state.items():
                            candidate = actual["state"][key][name]
                            assert torch.equal(candidate.to("cpu"), tensor.to("cpu")) if torch.is_tensor(tensor) else candidate == tensor
                    rates = {float(g["lr"]) for g in actual["param_groups"]}
                    if len(rates) != 1:
                        raise RuntimeError("Unexpected unequal optimizer learning rates")
                    self.alg.learning_rate = rates.pop()
                    self.resume_audit = {"optimizer_entries": len(actual["state"]), "optimizer_exactly_restored": True,
                                         "actor_critic_and_normalization_exactly_restored": True,
                                         "adaptive_learning_rate_restored": self.alg.learning_rate,
                                         "parent_iteration": int(checkpoint_dict["iter"]), "checkpoint_keys": list(checkpoint_dict)}
                    return loaded

                self.alg.load = checked_load
                original_log = self.logger.log
                action_term = self.env.unwrapped.action_manager.get_term("joint_pos")
                process_action = action_term.process_actions
                # A fixed 32-environment sample records actual mask sequences.
                # No extra simulator steps, RNG draws, or reference updates.
                mask_buffer = torch.zeros((600, 32, action_term.action_dim), device=action_term.device, dtype=torch.bool)
                episode_start_buffer = torch.zeros((600, 32), device=action_term.device, dtype=torch.bool)
                mask_cursor = 0
                sampled_run = None

                def capture_masks(actions):
                    nonlocal mask_cursor
                    episode_start_buffer[mask_cursor % 600].copy_(self.env.unwrapped.episode_length_buf[:32] == 0)
                    process_action(actions)
                    mask_buffer[mask_cursor % 600].copy_(action_term.dropout_mask[:32])
                    mask_cursor += 1

                action_term.process_actions = capture_masks

                def log_with_audit(**values):
                    nonlocal mask_cursor, sampled_run
                    original_log(**values)
                    if (values["it"] + 1) % 25 == 0 or values["it"] == 2000:
                        import numpy as np
                        env = self.env.unwrapped
                        term = env.command_manager.get_term("gait_reference")
                        action = env.action_manager.get_term("joint_pos")
                        high, maximum = severity_bounds(env.common_step_counter) if branch == "F" else (.75, 20)
                        masks = mask_buffer[:min(600, mask_cursor)].cpu().numpy()
                        episode_starts = episode_start_buffer[:min(600, mask_cursor)].cpu().numpy()
                        if sampled_run is None:
                            sampled_run = np.zeros(masks.shape[1:], dtype=np.int32)
                        run_lengths = []
                        reset_censored_runs = 0
                        per_joint_runs = [[] for _ in action.target_names]
                        for mask, episode_start in zip(masks, episode_starts):
                            reset_censored_runs += int((sampled_run[episode_start] > 0).sum())
                            sampled_run[episode_start] = 0
                            ended = sampled_run[(~mask) & (sampled_run > 0)]
                            run_lengths.extend(ended.tolist())
                            for j in range(len(per_joint_runs)):
                                per_joint_runs[j].extend(sampled_run[:, j][(~mask[:, j]) & (sampled_run[:, j] > 0)].tolist())
                            sampled_run = np.where(mask, sampled_run + 1, 0)
                        mask_cursor = 0
                        histogram = np.histogram(run_lengths, bins=[1, 2, 3, 5, 10, 20, 30, 40, np.inf])[0].tolist()
                        publish(directory / "training_audit.json", {"at_utc": stamp(), "completed_updates": values["it"] + 1,
                                "common_step_counter": int(env.common_step_counter), "control_dt": env.step_dt,
                                "branch": branch, "mode": term.cfg.mode, "max_dropout_probability": high,
                                "max_duration_steps": maximum, "mean_loss_trace": float(term.loss_trace.mean()),
                                "mean_cached_weight": float(term._weights.mean()),
                                "selected_joint_fraction": float(action.selected_fault_mask.float().mean()),
                                "dropped_joint_fraction": float(action.dropout_mask.float().mean()),
                                "right_ankle_action_index": list(action.target_names).index("right_ankle_pitch_joint"),
                                "fault_joint_mean_weight": float(term._weights[term.loss_trace > .1].mean()) if bool((term.loss_trace > .1).any()) else None,
                                "unaffected_joint_mean_weight": float(term._weights[term.loss_trace == 0].mean()) if bool((term.loss_trace == 0).any()) else None,
                                "sampled_mask_sequences": {"num_envs": 32, "control_steps": len(masks),
                                    "completed_run_bins": ["1", "2", "3-4", "5-9", "10-19", "20-29", "30-39", "40+"],
                                    "completed_run_counts": histogram, "completed_runs": len(run_lengths),
                                    "episode_reset_censored_runs": reset_censored_runs,
                                    "per_joint_completed_runs": {name: {"count": len(runs), "max_steps": max(runs, default=0),
                                        "bins": np.histogram(runs, bins=[1, 2, 3, 5, 10, 20, 30, 40, np.inf])[0].tolist()}
                                        for name, runs in zip(action.target_names, per_joint_runs)},
                                    "max_completed_run_steps": max(run_lengths, default=0),
                                    "max_censored_ongoing_steps": int(sampled_run.max())},
                                "reference_sha256": term.reference.sha256})
                        with (directory / "training_audit_history.jsonl").open("a") as history:
                            history.write(json.dumps(read(directory / "training_audit.json")) + "\n")

                self.logger.log = log_with_audit

            def learn(self, *args, **kwargs):
                assert self.current_learning_iteration == 2000
                assert self.env.unwrapped.common_step_counter == 2000 * 24
                publish(directory / "resume_receipt.json", {"at_utc": stamp(), **self.resume_audit,
                        "next_ppo_iteration": self.current_learning_iteration, "env_common_step_counter": self.env.unwrapped.common_step_counter,
                        "target_updates": 4000, "added_updates": 2000, "private_snapshot": snapshot,
                        "protocol_sha256": value["sha256"], "device": str(self.device)})
                return super().learn(*args, **kwargs)

        train.load_runner_cls = lambda _: AuditedRunner
        configured = train.TrainConfig(env=cfg.env, agent=cfg.agent, gpu_ids=[0], enable_nan_guard=True,
                                      resume_checkpoint_file=snapshot["private_snapshot"])
        publish(directory / "configuration_receipt.json", {"at_utc": stamp(), "branch": branch, "parent": parent,
                "log_directory": str(log_dir), "reference_sha256": reference.sha256, "agent": asdict(cfg.agent),
                "action": asdict(cfg.env.actions["joint_pos"]), "reference": asdict(cfg.env.commands["gait_reference"]),
                "environment_config_checked_against_parent": True, "protocol_sha256": value["sha256"]})
        train.run_train(task, configured, log_dir)
        final = log_dir / "model_3999.pt"
        assert saved_iteration(final) == 3999
        assert sha(Path(parent["checkpoint"])) == parent["sha256"]
        result.update(status="completed", final_checkpoint=str(final), final_sha256=sha(final), completed_updates=4000, added_updates=2000)
        return_code = 0
    except TrainingStopRequested as ex:
        result.update(status="stopped", reason=str(ex), stop_category=ex.category)
        return_code = 3
    except Exception as ex:
        result.update(status="failed", reason=f"{type(ex).__name__}: {ex}")
        raise
    finally:
        result["updated_at_utc"] = stamp()
        publish(directory / "result.json", result)
    return return_code


def evaluate_worker(value, branch, index):
    import torch
    from scripts import evaluate_style_dropout as evaluator
    from scripts import style_evaluation_metrics as metrics
    from scripts.style_resume import stage_private_checkpoint
    from src.tasks.velocity.rl.resource_control import TrainingStopRequested, read_host_memory
    torch.set_num_threads(3)
    parent = value["parents"][index]
    directory = directory_for(branch, "eval", parent)
    directory.mkdir(parents=True, exist_ok=True)
    finished = read(directory_for(branch, "train", parent) / "result.json")
    assert finished["status"] == "completed" and finished["completed_updates"] == 4000
    assert sha(Path(finished["final_checkpoint"])) == finished["final_sha256"]
    checkpoint = stage_private_checkpoint(Path(finished["final_checkpoint"]))["private_snapshot"]
    original_budget = evaluator.check_budget

    def budget(resource_limits):
        guard(directory, EVAL_END)
        sample = original_budget(resource_limits)
        if read_host_memory()["host_available_ram_mib"] < 8192:
            raise TrainingStopRequested("Host memory reserve breached", category="resource")
        return sample

    evaluator.check_budget = budget
    original_rmse = metrics.alive_rmse
    counts = []

    def counted_rmse(values, alive, start, end):
        selected = values[start:end][alive[start:end]]
        counts.append({"frame_env_samples": selected.numel(), "finite_frame_env_samples": int(torch.isfinite(selected).sum()),
                       "window_start": start, "window_end": min(end, len(values))})
        return original_rmse(values, alive, start, end)

    metrics.alive_rmse = counted_rmse
    report = {"status": "evaluating", "branch": branch, "metric_schema_version": 3,
              "schema2_metric_values_unchanged": True, "extra_fields": "valid frame/environment counts",
              "protocol_sha256": value["sha256"], "planned_cases": 57, "completed_cases": 0, "records": []}
    names = ["first_life_xy", "first_life_reference", "fault_xy", "fault_reference", "fault_unaffected_reference", "post_fault_reference"]
    try:
        for scenario, config in value["evaluation"]["scenarios"].items():
            for seed in value["evaluation"]["evaluation_seeds"]:
                guard(directory, EVAL_END)
                counts.clear()
                result = evaluator.evaluate_one(parent["group"], Path(checkpoint), scenario, seed, REFERENCE, 128, limits(),
                    training_seed=parent["seed"], control_directory=directory, scenario_config=config)
                labels = names if config["joints"] else names[:2]
                if len(counts) != len(labels):
                    raise RuntimeError("Metric counting hook contract changed")
                result.update(branch=branch, protocol_sha256=value["sha256"], metric_schema_version=3,
                              valid_samples=dict(zip(labels, counts)), training_completed_updates=4000)
                report["records"].append(result)
                report["completed_cases"] += 1
                report["updated_at_utc"] = stamp()
                publish(directory / "evaluation.json", report)
                import gc
                gc.collect()
                torch.cuda.empty_cache()
        report["status"] = "completed"
        return_code = 0
    except TrainingStopRequested as ex:
        report.update(status="stopped", reason=str(ex), stop_category=ex.category)
        return_code = 3
    except Exception as ex:
        report.update(status="failed", reason=f"{type(ex).__name__}: {ex}")
        raise
    finally:
        report["updated_at_utc"] = stamp()
        publish(directory / "evaluation.json", report)
    return return_code


def validate():
    """CPU-only algebra/config checks; deliberately no policy training or GPU allocation."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    from scripts.cloud_v2_campaign import admission, TRAIN_END, EVAL_END, verify_protocol
    import mjlab.tasks  # noqa: F401
    import src.tasks  # noqa: F401
    from scripts.train import TrainConfig
    from scripts.run_style_dropout import configure_reference
    from src.tasks.velocity.mdp.style_reference import load_gait_reference
    from src.tasks.velocity.config.g1.style_dropout_cfg import STYLE_TASKS
    from src.tasks.velocity.mdp.style_reference import update_loss_trace, imitation_weights
    from src.tasks.velocity.mdp.action_dropout import JointTargetDropoutActionCfg
    assert severity_bounds(48000) == (.75, 20)
    assert severity_bounds(54000) == (.875, 30)
    assert severity_bounds(60000) == (1., 40)
    assert severity_bounds(96000) == (1., 40)
    extended = extend_action(JointTargetDropoutActionCfg(entity_name="robot", actuator_names=(".*",)))
    assert extended.extension_origin_steps == 48000 and extended.event_probability == .005
    torch.manual_seed(97042)
    probabilities, durations, full = sample_expanded_events(extended, 60000, 20000, "cpu")
    assert .23 < float(full.float().mean()) < .27
    assert torch.all(probabilities[full] == 1.) and torch.all(durations[full] == 40)
    p0, d0, full0 = sample_expanded_events(extended, 48000, 20000, "cpu")
    assert not bool(full0.any()) and float(p0.max()) <= .75 and int(d0.max()) <= 20
    trace = torch.zeros(3, 29)
    mask = torch.zeros_like(trace, dtype=torch.bool)
    mask[:, 7] = True
    for _ in range(20):
        trace = update_loss_trace(trace, mask, .02, .08, .25)
    d = imitation_weights(trace, torch.ones(29), .75, "selective")
    c = imitation_weights(trace, torch.ones(29), .75, "global")
    assert torch.allclose(c.sum(-1), d.sum(-1), atol=1e-5)
    assert torch.all(d[:, 7] < .26) and torch.all(d[:, 8] == 1.)
    for _ in range(70):
        trace = update_loss_trace(trace, torch.zeros_like(mask), .02, .08, .25)
    recovered = imitation_weights(trace, torch.ones(29), .75, "selective")
    assert torch.all(recovered[:, 7] > .99)
    assert not admission(epoch(TRAIN_END), "training", 1, 1)
    assert not admission(epoch(EVAL_END), "evaluation", 0, 1)
    audit = read(AUDIT)
    reference = load_gait_reference(REFERENCE)
    for model in audit["models"]:
        cfg = TrainConfig.from_task(STYLE_TASKS[model["group"]][0])
        configure_reference(cfg.env, reference)
        cfg.env.scene.num_envs = 1024
        cfg.env.seed = cfg.agent.seed = model["seed"]
        cfg.env.sim.nan_guard.enabled = True
        verify_parent_config(cfg, model)
    value = verify_protocol() if (CONTROL / "protocol.json").exists() else {"parents": audit["models"]}
    for branch in "EF":
        for parent in value["parents"]:
            assert directory_for(branch, "train", parent).is_relative_to(CONTROL)
    report = {"at_utc": stamp(), "status": "passed", "purpose": "SOFTWARE_CHECKS_NOT_POLICY_PERFORMANCE", "cpu_only": True,
              "checks": ["curriculum_origin_2000_updates", "500_update_ramp", "sampled_exact_full_loss_long_events", "matched_C_D_weight_mass", "locality", "smooth_recovery", "deadline_admission", "20_parent_full_environment_and_agent_configs", "isolated_output_paths"],
              "failed_joint_weight": float(d[0, 7]), "recovered_weight": float(recovered[0, 7])}
    publish(CONTROL / "software_validation.json", report)
    print("V2_VALIDATION_PASSED", json.dumps(report))
