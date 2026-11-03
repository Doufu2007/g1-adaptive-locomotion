"""G-only worker adapters; frozen E/F sources and outputs remain immutable."""
from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import time

from scripts.cloud_stage_today_20261103 import (
    ROOT, CONTROL, REFERENCE, EVAL_END, stamp, epoch, publish, read, sha,
    directory_for, require, safety,
)
from scripts.cloud_v2_runtime import severity, severity_bounds, sample_expanded_events, extend_action, verify_parent_config, limits
from scripts.cloud_v2_campaign import AUDIT, manual_stop_paths

TRAIN_END = EVAL_END


def guard(directory, deadline):
    from src.tasks.velocity.rl.resource_control import TrainingStopRequested
    paths = manual_stop_paths(directory) + ([str(CONTROL / "stop.request")] if (CONTROL / "stop.request").exists() else [])
    if paths:
        raise TrainingStopRequested(f"User stop remains active: {paths}", category="manual")
    if time.time() >= epoch(deadline):
        raise TrainingStopRequested("Independent today's deadline reached", category="deadline")


class MechanismAudit:
    """Read-only hooks over 32 environments; no extra RNG or simulator steps.

    Target and recurrence residuals stay on GPU until the 25-update logger.
    Trace/weights are captured immediately after their actual diagnostics update,
    before a later reset can clear trace while leaving a cached weight behind.
    """
    def __init__(self, runner, directory):
        import torch
        self.torch, self.runner, self.directory = torch, runner, directory
        self.env = runner.env.unwrapped
        self.action = self.env.action_manager.get_term("joint_pos")
        self.term = self.env.command_manager.get_term("gait_reference")
        self.n, self.capacity, self.cursor = 32, 600, 0
        self.ids = [self.term.joint_names.index(n) for n in self.action.target_names]
        shape = (600, 32, self.action.action_dim)
        self.mask = torch.zeros(shape, device=self.action.device, dtype=torch.bool)
        self.trace = torch.zeros(shape, device=self.action.device)
        self.weight = torch.zeros_like(self.trace)
        self.residuals = torch.zeros((600, 7), device=self.action.device)
        self.counts = torch.zeros((600, 5), device=self.action.device, dtype=torch.long)
        self.reset = torch.zeros((600, 32), device=self.action.device, dtype=torch.bool)
        self.life = torch.zeros((600, 32), device=self.action.device, dtype=torch.long)
        self.life_id = torch.zeros(32, device=self.action.device, dtype=torch.long)
        self.pending_reset = torch.zeros(32, device=self.action.device, dtype=torch.bool)
        self.steps = [None] * 600
        self.sampled_run = None
        self.cumulative_full_runs = 0
        self.any_violation = False
        self.last_slot = None
        self.full_event_counts = torch.zeros((600, 2), device=self.action.device, dtype=torch.long)
        original_action, original_reset = self.action.process_actions, self.action.reset
        original_diag, original_log = self.term.diagnostics, runner.logger.log

        def process(actions):
            slot = self.cursor % self.capacity
            self.last_slot = slot
            previous = self.action._previous_applied_target[:32].clone()
            self.reset[slot].copy_(self.pending_reset)
            self.life[slot].copy_(self.life_id)
            self.pending_reset.zero_()
            original_action(actions)
            mask = self.action.dropout_mask[:32]
            applied = self.action._processed_actions[:32]
            desired = self.action._last_desired_target[:32]
            self.mask[slot].copy_(mask)
            self.residuals[slot].zero_()
            self.counts[slot].zero_()
            self.counts[slot, 0] = mask.sum()
            self.counts[slot, 1] = (~mask).sum()
            self.residuals[slot, 0] = torch.where(mask, (applied - previous).abs(), 0.).max()
            self.residuals[slot, 1] = torch.where(~mask, (applied - desired).abs(), 0.).max()
            self.residuals[slot, 6] = (~torch.isfinite(applied)).sum()
            self.steps[slot] = None
            self.full_event_counts[slot].zero_()
            if hasattr(self.action, "_full_loss_event"):
                # Explicit chosen event exposure; contiguous-mask runs are reported separately.
                selected = self.action.selected_fault_mask[:32].any(-1)
                full = self.action._full_loss_event[:32] & selected
                self.full_event_counts[slot, 0] = full.sum()
                self.full_event_counts[slot, 1] = (full & (self.action._fault_probability[:32] == 1.)).sum()
            self.cursor += 1

        def reset(env_ids=None):
            original_reset(env_ids)
            selected = torch.zeros(self.env.num_envs, device=self.action.device, dtype=torch.bool)
            if env_ids is None:
                selected[:] = True
            else:
                selected[env_ids] = True
            self.life_id.add_(selected[:32].long())
            self.pending_reset.logical_or_(selected[:32])

        def diagnostics():
            from src.tasks.velocity.mdp.style_reference import update_loss_trace, imitation_weights
            step = int(self.env.common_step_counter)
            updating = self.term._last_trace_step != step
            before = self.term.loss_trace[:32].clone() if updating else None
            result = original_diag()
            if updating and self.last_slot is not None:
                slot = self.last_slot
                mask = torch.zeros_like(before, dtype=torch.bool)
                mask[:, self.ids] = self.action.dropout_mask[:32]
                actual = self.term.loss_trace[:32]
                expected = update_loss_trace(before, mask, self.env.step_dt, self.term.cfg.attack_tau_s, self.term.cfg.recovery_tau_s)
                expected_weights = imitation_weights(actual, self.term.base_weights, self.term.cfg.alpha, self.term.cfg.mode)
                d = imitation_weights(actual, self.term.base_weights, self.term.cfg.alpha, "selective")
                c = imitation_weights(actual, self.term.base_weights, self.term.cfg.alpha, "global")
                self.trace[slot].copy_(actual[:, self.ids])
                self.weight[slot].copy_(self.term._weights[:32, self.ids])
                self.residuals[slot, 2] = (actual - expected).abs().max()
                self.residuals[slot, 3] = (self.term._weights[:32] - expected_weights).abs().max()
                self.residuals[slot, 4] = (((c-d) * self.term.base_weights).sum(-1) / self.term.base_weights.sum()).abs().max()
                self.residuals[slot, 5] = (~torch.isfinite(actual)).sum() + (~torch.isfinite(self.term._weights[:32])).sum()
                self.counts[slot, 2] = actual.numel()
                self.counts[slot, 3] = (~mask).sum()
                self.counts[slot, 4] = mask.sum()
                self.steps[slot] = step
            return result

        def log(**values):
            original_log(**values)
            if (values["it"] + 1) % 25 == 0 or values["it"] == 2000:
                self.flush(values["it"] + 1)

        self.action.process_actions, self.action.reset = process, reset
        self.term.diagnostics, runner.logger.log = diagnostics, log

    def flush(self, updates):
        import numpy as np
        count = min(self.cursor, self.capacity)
        if not count:
            return
        require(self.cursor <= self.capacity, "Mechanism sample ring overflow; logging contract changed")
        masks = self.mask[:count].cpu().numpy()
        resets = self.reset[:count].cpu().numpy()
        totals = self.counts[:count].sum(0).cpu().tolist()
        residuals = self.residuals[:count].amax(0).cpu().tolist()
        eligible = np.array([step is not None for step in self.steps[:count]])
        if self.sampled_run is None:
            self.sampled_run = np.zeros(masks.shape[1:], dtype=np.int32)
        ended, reset_censored = [], 0
        for mask, reset in zip(masks, resets):
            reset_censored += int((self.sampled_run[reset] > 0).sum())
            self.sampled_run[reset] = 0
            ended.extend(self.sampled_run[(~mask) & (self.sampled_run > 0)].tolist())
            self.sampled_run = np.where(mask, self.sampled_run + 1, 0)
        self.cumulative_full_runs += sum(run >= 40 for run in ended)
        lane = np.unravel_index(np.argmax(masks.sum(0)), masks.shape[1:])
        env_id, joint = map(int, lane)
        trace = self.trace[:count, env_id, joint].cpu().tolist()
        weights = self.weight[:count, env_id, joint].cpu().tolist()
        lives = self.life[:count, env_id].cpu().tolist()
        sequence = [{"step": self.steps[i], "life": lives[i], "reset_before_action": bool(resets[i, env_id]),
            "dropout": bool(masks[i, env_id, joint]), "trace": trace[i], "weight": weights[i]}
            for i in range(count) if eligible[i]]
        violation = any(x > 1e-5 for x in residuals[:4]) or residuals[4] > 1e-5 or any(x > 0 for x in residuals[5:])
        self.any_violation |= violation
        high, maximum = severity_bounds(self.env.common_step_counter)
        report = {"at_utc": stamp(), "completed_updates": updates, "branch": "G", "mode": self.term.cfg.mode,
            "common_step_counter": int(self.env.common_step_counter), "control_dt": self.env.step_dt,
            "max_dropout_probability": high, "max_duration_steps": maximum, "num_sampled_envs": 32,
            "observed_control_steps": count, "synchronized_diagnostics_steps": int(eligible.sum()),
            "hold_samples": totals[0], "received_samples": totals[1], "trace_channel_samples": totals[2],
            "hold_max_abs_residual": residuals[0], "receive_max_abs_residual": residuals[1],
            "trace_recurrence_max_abs_residual": residuals[2], "weight_formula_max_abs_residual": residuals[3],
            "same_trace_C_D_normalized_mass_max_abs_residual": residuals[4],
            "nonfinite_trace_weight_count": residuals[5], "nonfinite_applied_count": residuals[6],
            "mechanism_violation": self.any_violation, "violation_this_window": violation,
            "target_scope": "Application target channel; physical joint angles need not freeze",
            "sampled_mask_sequences": {"completed_run_max_steps": max(ended, default=0),
                "completed_40plus_runs": sum(run >= 40 for run in ended), "cumulative_completed_40plus_runs": self.cumulative_full_runs,
                "reset_censored_runs": reset_censored, "ongoing_max_steps": int(self.sampled_run.max()),
                "completed_run_counts": np.histogram(ended, bins=[1,2,3,5,10,20,30,40,np.inf])[0].tolist()},
            "full_loss_event_frame_env_counts": self.full_event_counts[:count].sum(0).cpu().tolist(),
            "event_aligned_lane": {"environment": env_id, "joint": self.action.target_names[joint], "sequence": sequence,
                "selection": "most observed dropout samples in this window; mechanism diagnostic, not policy selection"},
            "reference_sha256": self.term.reference.sha256}
        publish(self.directory / "training_audit.json", report)
        with (self.directory / "training_audit_history.jsonl").open("a") as history:
            history.write(json.dumps(report) + "\n")
        self.cursor = 0
        self.last_slot = None


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
    directory = directory_for("train", parent)
    log_dir = ROOT / "logs/rsl_rl" / "g1_dropout_style_G_today_20261103" / f"{parent['group']}_seed{parent['seed']}"
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
                          G1_RUN_PURPOSE="DEVELOPMENT_TRAINING_G_TODAY_20261103")
        torch.set_num_threads(3)
        original_ensure = ResourceController.ensure_continue

        def ensure(self):
            original_ensure(self)
            guard(directory, TRAIN_END)

        ResourceController.ensure_continue = ensure
        gate = ResourceController(resource_limits, directory, directory)
        gate.before_allocation(3000)
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
        if branch == "G":
            cfg.env.actions["joint_pos"] = extend_action(cfg.env.actions["joint_pos"])
        cfg.agent.max_iterations = 3000
        cfg.agent.save_interval = 100
        cfg.agent.experiment_name = "g1_dropout_style_G_today_20261103"
        cfg.agent.run_name = f"G_today_{parent['group']}_seed{parent['seed']}"
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
                        require(set(model_state) == set(checkpoint_dict[f"{component}_state_dict"]),
                                f"{component} state/normalization key set changed")
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
                self.mechanism_audit = MechanismAudit(self, directory)

            def learn(self, *args, **kwargs):
                assert self.current_learning_iteration == 2000
                assert self.env.unwrapped.common_step_counter == 2000 * 24
                publish(directory / "resume_receipt.json", {"at_utc": stamp(), **self.resume_audit,
                        "next_ppo_iteration": self.current_learning_iteration, "env_common_step_counter": self.env.unwrapped.common_step_counter,
                        "target_updates": 3000, "added_updates": 1000, "private_snapshot": snapshot,
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
        final = log_dir / "model_2999.pt"
        assert saved_iteration(final) == 2999
        assert sha(Path(parent["checkpoint"])) == parent["sha256"]
        result.update(status="completed", final_checkpoint=str(final), final_sha256=sha(final), completed_updates=3000, added_updates=1000)
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


def evaluate_worker(value, index, eval_seed):
    import gc
    import torch
    from scripts import evaluate_style_dropout as evaluator
    from scripts import style_evaluation_metrics as metrics
    from scripts.style_resume import saved_iteration
    from src.tasks.velocity.rl.resource_control import TrainingStopRequested, read_host_memory
    parent = value["parents"][index]
    directory = directory_for("eval", parent) / f"eval_seed{eval_seed}"
    directory.mkdir(parents=True, exist_ok=True)
    safety()
    manifest = read(CONTROL / "checkpoints.json")
    require(manifest.get("protocol_sha256") == value["sha256"], "Checkpoint manifest protocol mismatch")
    checkpoint = next(m for m in manifest["models"] if m["index"] == index)
    path = Path(checkpoint["private_snapshot"])
    require(sha(path) == checkpoint["sha256"] and saved_iteration(path) == 2999, "Evaluation checkpoint mismatch")
    resource_limits = limits()
    os.environ["G1_RESOURCE_LIMITS_JSON"] = json.dumps(asdict(resource_limits))
    torch.set_num_threads(3)
    torch.set_num_interop_threads(1)
    original_budget, original_rmse = evaluator.check_budget, metrics.alive_rmse
    original_runner = evaluator.load_runner_cls
    load_receipts, counts = [], []

    def budget(resource_limits):
        guard(directory, EVAL_END)
        sample = original_budget(resource_limits)
        if read_host_memory()["host_available_ram_mib"] < 8192:
            raise TrainingStopRequested("Host memory reserve breached", category="resource")
        return sample

    def counted_rmse(values, alive, start, end):
        selected = values[start:end][alive[start:end]]
        counts.append({"frame_environment_samples": selected.numel(),
            "finite_frame_environment_samples": int(torch.isfinite(selected).sum()),
            "window_start": start, "window_end": min(end, len(values))})
        return original_rmse(values, alive, start, end)

    def runner_for(task):
        base = original_runner(task)
        require(base is not None, "Inference runner registry missing")

        class InferenceAudit(base):
            def load(self, checkpoint_path, *args, **kwargs):
                loaded = super().load(checkpoint_path, *args, **kwargs)
                expected = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                actual = self.alg.actor.state_dict()
                require(set(actual) == set(expected["actor_state_dict"]), "Inference actor/normalization keys differ")
                for name, tensor in expected["actor_state_dict"].items():
                    require(torch.equal(actual[name].detach().cpu(), tensor.cpu()), f"Inference restore failed at {name}")
                normalization = [name for name in actual if "norm" in name.lower()]
                require(normalization, "Expected checkpoint-contained observation normalization buffers")
                self._normalization_snapshot = {name: actual[name].detach().clone() for name in normalization}
                load_receipts.append({"at_utc": stamp(), "actor_and_normalization_exact": True,
                    "normalization_keys": normalization, "checkpoint_sha256": checkpoint["sha256"], "iteration": expected["iter"]})
                return loaded

            def get_inference_policy(self, *args, **kwargs):
                policy = super().get_inference_policy(*args, **kwargs)
                require(not self.alg.actor.training, "Inference actor must be in eval mode")

                def audited_policy(observation):
                    result = policy(observation)
                    if self._normalization_snapshot is not None:
                        actual = self.alg.actor.state_dict()
                        for name, tensor in self._normalization_snapshot.items():
                            require(torch.equal(actual[name], tensor), f"Inference updated normalization: {name}")
                        self._normalization_snapshot = None
                    return result
                return audited_policy

        return InferenceAudit

    evaluator.check_budget, metrics.alive_rmse, evaluator.load_runner_cls = budget, counted_rmse, runner_for
    report = {"status": "evaluating", "at_utc": stamp(), "branch": "G", "group": parent["group"],
        "train_seed": parent["seed"], "evaluation_seed": eval_seed, "protocol_sha256": value["sha256"],
        "planned_cases": len(value["evaluation"]["scenarios"]), "completed_cases": 0, "records": [], "load_receipts": load_receipts}
    names = ["first_life_xy", "first_life_reference", "fault_xy", "fault_reference", "fault_unaffected_reference", "post_fault_reference"]
    try:
        for scenario, config in value["evaluation"]["scenarios"].items():
            guard(directory, EVAL_END)
            counts.clear()
            record = evaluator.evaluate_one(parent["group"], path, scenario, eval_seed, REFERENCE, 128, resource_limits,
                training_seed=parent["seed"], control_directory=directory, scenario_config=config, simulation_seconds=20.)
            labels = names if config["joints"] else names[:2]
            require(len(counts) == len(labels), "Metric count hook contract changed")
            record.update(branch="G", training_completed_updates=3000, metric_schema_version=3,
                valid_samples=dict(zip(labels, counts)), protocol_sha256=value["sha256"],
                checkpoint_sha256=checkpoint["sha256"], reference_sha256=value["reference_sha256"])
            report["records"].append(record)
            report["completed_cases"] += 1
            report["updated_at_utc"] = stamp()
            publish(directory / "evaluation.json", report)
            gc.collect()
            torch.cuda.empty_cache()
        report["status"] = "completed"
        return_code = 0
    except TrainingStopRequested as error:
        report.update(status="stopped", reason=str(error), stop_category=error.category)
        return_code = 3
    except Exception as error:
        report.update(status="failed", reason=f"{type(error).__name__}: {error}")
        return_code = 2
    finally:
        evaluator.check_budget, metrics.alive_rmse, evaluator.load_runner_cls = original_budget, original_rmse, original_runner
        report["updated_at_utc"] = stamp()
        publish(directory / "evaluation.json", report)
    return return_code


def validate_cpu():
    """Actual G sampler/config and hook tests; CUDA hidden, no environment build."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import tempfile
    from types import SimpleNamespace
    import torch
    import mjlab.tasks
    import src.tasks
    from scripts.train import TrainConfig
    from scripts.run_style_dropout import configure_reference
    from scripts.cloud_stage_today_20261103 import SOURCES
    from src.tasks.velocity.config.g1.style_dropout_cfg import STYLE_TASKS
    from src.tasks.velocity.mdp.style_reference import load_gait_reference, update_loss_trace, imitation_weights
    from src.tasks.velocity.mdp.action_dropout import JointTargetDropoutActionCfg
    torch.set_num_threads(1)
    extended = extend_action(JointTargetDropoutActionCfg(entity_name="robot", actuator_names=(".*",)))
    require(extended.extension_origin_steps == 48000 and extended.extension_ramp_steps == 12000, "G ramp origin/length")
    torch.manual_seed(97042)
    probabilities, durations, full = sample_expanded_events(extended, 60000, 20000, "cpu")
    require(.23 < float(full.float().mean()) < .27, "Full-loss event mass")
    require(bool(torch.all(probabilities[full] == 1.)) and bool(torch.all(durations[full] == 40)), "Exact full long events")
    p0, d0, full0 = sample_expanded_events(extended, 48000, 20000, "cpu")
    require(not bool(full0.any()) and float(p0.max()) <= .75 and int(d0.max()) <= 20, "No prematurely saturated extension")
    require(severity_bounds(72000, extended) == (1., 40), "Stable last 500 updates")
    reference = load_gait_reference(REFERENCE)
    audited = read(AUDIT)["models"]
    require(len(audited) == 20, "Need twenty parent configurations")
    for model in audited:
        cfg = TrainConfig.from_task(STYLE_TASKS[model["group"]][0])
        configure_reference(cfg.env, reference)
        cfg.env.scene.num_envs = 1024
        cfg.env.seed = cfg.agent.seed = model["seed"]
        cfg.env.sim.nan_guard.enabled = True
        verify_parent_config(cfg, model)
        cfg.env.actions["joint_pos"] = extend_action(cfg.env.actions["joint_pos"])
        cfg.agent.max_iterations = 3000
        require(cfg.agent.num_steps_per_env == 24 and cfg.agent.max_iterations - 2000 == 1000, "G actual agent budget")
    # Isolated synthetic CPU objects test instrumentation contracts, not policy performance.
    action = SimpleNamespace(action_dim=2, target_names=("test_joint0", "test_joint1"), device="cpu",
        _previous_applied_target=torch.zeros(32,2), _last_desired_target=torch.zeros(32,2),
        _processed_actions=torch.zeros(32,2), dropout_mask=torch.zeros(32,2,dtype=torch.bool))
    env = SimpleNamespace(num_envs=32, common_step_counter=48000, step_dt=.02)
    term = SimpleNamespace(joint_names=action.target_names, loss_trace=torch.zeros(32,2), _weights=torch.ones(32,2),
        base_weights=torch.ones(2), _last_trace_step=-1,
        cfg=SimpleNamespace(attack_tau_s=.08,recovery_tau_s=.25,alpha=.75,mode="selective"),
        reference=SimpleNamespace(sha256="ISOLATED_SYNTHETIC_CPU_HOOK_TEST"))
    def process(desired):
        action._last_desired_target.copy_(desired)
        action._processed_actions.copy_(torch.where(action.dropout_mask, action._previous_applied_target, desired))
        action._previous_applied_target.copy_(action._processed_actions)
    def reset(ids=None):
        selection = slice(None) if ids is None else ids
        action._previous_applied_target[selection] = 0
        action.dropout_mask[selection] = False
    def diagnostics():
        if term._last_trace_step != env.common_step_counter:
            term.loss_trace.copy_(update_loss_trace(term.loss_trace, action.dropout_mask, .02,.08,.25))
            term._last_trace_step = env.common_step_counter
        term._weights = imitation_weights(term.loss_trace, term.base_weights, .75,"selective")
        return None
    action.process_actions, action.reset, term.diagnostics = process, reset, diagnostics
    env.action_manager = SimpleNamespace(get_term=lambda _:action)
    env.command_manager = SimpleNamespace(get_term=lambda _:term)
    runner = SimpleNamespace(env=SimpleNamespace(unwrapped=env), logger=SimpleNamespace(log=lambda **_:None))
    with tempfile.TemporaryDirectory(prefix="G_isolated_synthetic_cpu_") as temporary:
        hook = MechanismAudit(runner, Path(temporary))
        low = None
        for step in range(90):
            action.dropout_mask.zero_()
            if step < 20:
                action.dropout_mask[:,0] = True
            action.process_actions(torch.full((32,2),float(step + 1)))
            env.common_step_counter += 1
            term.diagnostics()
            term.diagnostics()  # no duplicate trace update or instrumentation sample
            if step == 19:
                low = float(term._weights[0,0])
        hook.flush(3000)
        report = read(Path(temporary) / "training_audit.json")
        require(not report["mechanism_violation"] and report["synchronized_diagnostics_steps"] == 90, "Hook synchronization/residuals")
        require(report["hold_samples"] == 640 and report["received_samples"] == 5120, "Hold/receive counts")
        require(low < .26 and float(term._weights[0,0]) > .99, "Event-aligned software recovery")
        require(report["sampled_mask_sequences"]["completed_run_max_steps"] == 20, "Contiguous run accounting")
        action.reset(torch.tensor([0]))
        require(int(hook.life_id[0]) == 1 and bool(hook.pending_reset[0]), "Reset/lifecycle hook")
    receipt = {"at_utc": stamp(), "status": "passed", "cpu_only": True,
        "purpose": "SOFTWARE_CHECKS_NOT_POLICY_PERFORMANCE", "no_environment_or_gpu_allocation": True,
        "checks": ["20_exact_parent_configurations", "3000_target_1000_added", "500_ramp_500_stable", "exact_full_long_event_mass",
            "hold_receive_residual_hooks", "synchronized_trace_weight_hooks", "same_trace_C_D_mass", "event_aligned_recovery", "reset_lifecycle"],
        "sources": {source:sha(ROOT/source) for source in SOURCES}}
    publish(CONTROL / "software_validation.json", receipt)
    print("G_CPU_RUNTIME_VALIDATION_PASSED", json.dumps(receipt))

