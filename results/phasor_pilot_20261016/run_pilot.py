"""Run the preregistered PHASOR PPO pilot and fixed-seed evaluation."""

from __future__ import annotations

import argparse
import contextlib
import ctypes
from ctypes import wintypes
import hashlib
import importlib.util
import json
import math
import os
import platform
import random
import re
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PILOT = ROOT / "results" / "phasor_pilot_20261016"
PREREG = PILOT / "pilot_preregistration.json"
ENCODER = (
    ROOT
    / "results"
    / "phasor_contact_supervised_v3"
    / "run_20261016"
    / "encoder_training_v3_attempt3"
    / "phasor_encoder_g1_contact_v3.pt"
)
ENCODER_SHA256 = "1770615e14e0a610bfb569ffc3c7969adc1dd6413fdb5322f4a7ebb655045e6f"
PARTIAL_BASELINE_DIR = (
    ROOT
    / "logs"
    / "rsl_rl"
    / "g1_velocity"
    / "2026-10-16_18-36-51_phasor_pilot_baseline_seed_20261031"
)
PARTIAL_BASELINE_CKPT = PARTIAL_BASELINE_DIR / "model_400.pt"
SOURCE_FILES = (
    "scripts/train.py",
    "src/tasks/velocity/config/g1/env_cfgs.py",
    "src/tasks/velocity/config/g1/rl_cfg.py",
    "src/tasks/velocity/mdp/observations.py",
    "src/tasks/velocity/phasor/runtime.py",
    "src/tasks/velocity/velocity_env_cfg.py",
)

_BOOTED = False
_PATH_ALIASES: list[tuple[str, str]] = []
_DEVICE_MAPPINGS: list[str] = []


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=json_default) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def git_provenance() -> dict[str, Any]:
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=ROOT, text=True
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    return {"git_revision": revision, "git_dirty_at_run": dirty}


def _package_dir(name: str) -> Path:
    spec = importlib.util.find_spec(name)
    if spec is None:
        raise ModuleNotFoundError(name)
    if spec.submodule_search_locations:
        return Path(next(iter(spec.submodule_search_locations))).resolve()
    if spec.origin is None:
        raise RuntimeError(f"Cannot resolve package path for {name}")
    return Path(spec.origin).resolve().parent


def _define_dos_device(letter: str, target: Path) -> None:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    define = kernel32.DefineDosDeviceW
    define.argtypes = (wintypes.DWORD, wintypes.LPCWSTR, wintypes.LPCWSTR)
    define.restype = wintypes.BOOL
    target_nt = "\\??\\" + str(target.resolve())
    flags = 0x00000001 | 0x00000008  # RAW_TARGET_PATH | NO_BROADCAST_SYSTEM
    if not define(flags, f"{letter}:", target_nt):
        raise ctypes.WinError(ctypes.get_last_error())
    _DEVICE_MAPPINGS.append(letter)


def _remove_dos_devices() -> None:
    if os.name != "nt":
        return
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    define = kernel32.DefineDosDeviceW
    define.argtypes = (wintypes.DWORD, wintypes.LPCWSTR, wintypes.LPCWSTR)
    define.restype = wintypes.BOOL
    for letter in reversed(_DEVICE_MAPPINGS):
        define(0x00000002 | 0x00000001 | 0x00000008, f"{letter}:", None)
    _DEVICE_MAPPINGS.clear()


def _mapped_path(path: Any) -> str:
    raw = str(path)
    normalized = raw.replace("/", "\\")
    if len(normalized) >= 2 and normalized[1] == ":":
        candidate = normalized.casefold()
    else:
        candidate = str((ROOT / raw).resolve()).replace("/", "\\").casefold()
        normalized = str((ROOT / raw).resolve()).replace("/", "\\")
    for source, alias in _PATH_ALIASES:
        source_norm = source.replace("/", "\\").rstrip("\\")
        source_fold = source_norm.casefold()
        if candidate == source_fold:
            return alias + "\\"
        if candidate.startswith(source_fold + "\\"):
            relative = normalized[len(source_norm) :].lstrip("\\")
            return alias + "\\" + relative
    return raw


def bootstrap_runtime() -> None:
    """Install ASCII path aliases before MuJoCo and Warp initialize."""
    global _BOOTED, _PATH_ALIASES
    if _BOOTED:
        return
    if os.name != "nt":
        raise RuntimeError("The recorded MuJoCo/Warp path workaround requires Windows")

    ascii_temp = Path(r"<user-profile>\phasor_pilot_temp")
    ascii_cache = Path(r"<user-profile>\phasor_pilot_warp_cache")
    ascii_temp.mkdir(parents=True, exist_ok=True)
    ascii_cache.mkdir(parents=True, exist_ok=True)
    os.environ["TEMP"] = str(ascii_temp)
    os.environ["TMP"] = str(ascii_temp)
    os.environ["TMPDIR"] = str(ascii_temp)
    os.environ["WARPKIT_CACHE_DIR"] = str(ascii_cache)
    # EGL is a Linux-only MuJoCo backend. On Windows, keep MuJoCo's default
    # GLFW backend; this experiment does not request rendered frames.
    os.environ["MUJOCO_GL"] = ""
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    sys.path.insert(0, str(ROOT))

    roots = (_package_dir("warp"), _package_dir("mjlab"), ROOT.resolve())
    letters = ("X", "Y", "Z")
    for letter, root in zip(letters, roots, strict=True):
        _define_dos_device(letter, root)
        _PATH_ALIASES.append((str(root), letter + ":"))

    import warp
    import warp._src.build as warp_build
    import warp._src.context as warp_context

    warp.config.kernel_cache_dir = str(ascii_cache)
    warp_build.warp_home = "X:\\"
    warp_context.warp_home = "X:\\"
    if not hasattr(warp, "context"):
        warp.context = warp_context

    import mujoco

    original_from_file = mujoco.MjSpec.from_file

    def from_file_with_ascii_alias(path: Any, *args: Any, **kwargs: Any):
        return original_from_file(_mapped_path(path), *args, **kwargs)

    mujoco.MjSpec.from_file = from_file_with_ascii_alias
    _BOOTED = True


def encoder_hash_check() -> str:
    observed = sha256_file(ENCODER)
    if observed != ENCODER_SHA256:
        raise RuntimeError(
            f"Frozen encoder SHA-256 mismatch: expected {ENCODER_SHA256}, got {observed}"
        )
    return observed


def import_project():
    bootstrap_runtime()
    import numpy as np
    import torch
    import mjlab.tasks  # noqa: F401
    import src.tasks  # noqa: F401

    return np, torch


def task_for(condition: str) -> str:
    return "Unitree-G1-Flat" if condition == "baseline" else "Unitree-G1-Flat-PHASOR"


def configure_phase_encoder(env_cfg: Any, condition: str) -> None:
    if condition != "joint_phasor":
        return
    actor_term = env_cfg.observations["actor"].terms["phase"]
    reward_term = env_cfg.rewards["foot_gait"]
    actor_term.params["encoder_path"] = str(ENCODER)
    reward_term.params["encoder_path"] = str(ENCODER)


def seed_all(seed: int, torch: Any, np: Any) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class _FilteredTee:
    def __init__(self, terminal: Any, file_handle: Any, every: int = 50):
        self.terminal = terminal
        self.file_handle = file_handle
        self.every = every
        self.pending = ""
        self.last_progress = -1

    def write(self, value: str) -> int:
        self.file_handle.write(value)
        self.file_handle.flush()
        self.pending += value
        while "\n" in self.pending:
            line, self.pending = self.pending.split("\n", 1)
            match = re.search(r"Learning iteration\s+(\d+)/(\d+)", line)
            show = bool(
                "Traceback" in line
                or "RuntimeError" in line
                or "OutOfMemory" in line
                or "RUN_" in line
                or "[INFO]" in line
            )
            if match:
                iteration, total = map(int, match.groups())
                show = (
                    iteration == 1
                    or iteration == total
                    or iteration - self.last_progress >= self.every
                )
                if show:
                    self.last_progress = iteration
            if show:
                self.terminal.write(line + "\n")
                self.terminal.flush()
        return len(value)

    def flush(self) -> None:
        self.file_handle.flush()
        self.terminal.flush()

    def isatty(self) -> bool:
        return bool(self.terminal.isatty())


def train_one(
    condition: str,
    seed: int,
    *,
    resume_checkpoint: Path | None = None,
    prior_completed_iterations: int = 0,
) -> dict[str, Any]:
    np, torch = import_project()
    encoder_hash = encoder_hash_check()
    seed_all(seed, torch, np)

    from dataclasses import asdict
    from scripts import train as train_module
    from scripts.train import TrainConfig

    task_id = task_for(condition)
    cfg = TrainConfig.from_task(task_id)
    configure_phase_encoder(cfg.env, condition)
    cfg.env.scene.num_envs = 128
    cfg.env.seed = seed
    cfg.agent.seed = seed
    cfg.agent.max_iterations = 1001 - prior_completed_iterations
    cfg.agent.resume = resume_checkpoint is not None
    # Keep all six runs local and identical at the algorithm level. W&B needs
    # an external account; TensorBoard writes the same training metrics locally.
    cfg.agent.logger = "tensorboard"

    prefix = "pilot_" + condition + "_seed_" + str(seed)
    checkpoint_state = None
    if resume_checkpoint is not None:
        resume_checkpoint = resume_checkpoint.resolve()
        if not resume_checkpoint.is_file():
            raise FileNotFoundError(resume_checkpoint)
        checkpoint_state = torch.load(
            resume_checkpoint, map_location="cpu", weights_only=False
        )
        checkpoint_iteration = int(checkpoint_state.get("iter", -1))
        if prior_completed_iterations != checkpoint_iteration + 1:
            raise RuntimeError(
                "Resume iteration accounting mismatch: "
                f"checkpoint iter={checkpoint_iteration}, "
                f"prior_completed_iterations={prior_completed_iterations}"
            )
        if not 0 < prior_completed_iterations < 1001:
            raise RuntimeError("Resume checkpoint must be partial and within the frozen budget")
        if cfg.agent.max_iterations <= 0:
            raise RuntimeError("Resume checkpoint already meets the frozen iteration budget")
        cfg.agent.load_run = resume_checkpoint.parent.name
        cfg.agent.load_checkpoint = resume_checkpoint.name
        run_name = f"{prefix}_continued_from_{checkpoint_iteration}"
    else:
        run_name = prefix
    cfg.agent.run_name = run_name
    original_get_checkpoint_path = None
    original_runner_load = None
    if resume_checkpoint is not None:
        original_get_checkpoint_path = train_module.get_checkpoint_path
        train_module.get_checkpoint_path = lambda *_args, **_kwargs: resume_checkpoint
        from src.tasks.velocity.rl import VelocityOnPolicyRunner

        original_runner_load = VelocityOnPolicyRunner.load

        def load_and_advance_one_iteration(self: Any, path: str, *args: Any, **kwargs: Any):
            loaded = original_runner_load(self, path, *args, **kwargs)
            self.current_learning_iteration = int(checkpoint_state["iter"]) + 1
            return loaded

        VelocityOnPolicyRunner.load = load_and_advance_one_iteration

    if cfg.agent.max_iterations <= 0:
        raise ValueError("Training iteration count must be positive")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    log_dir = ROOT / "logs" / "rsl_rl" / "g1_velocity" / f"{timestamp}_{run_name}"
    log_dir.mkdir(parents=True, exist_ok=False)
    runs_dir = PILOT / "runs"
    legacy_attempt = runs_dir / prefix
    attempt_numbers = [1] if legacy_attempt.exists() else []
    for existing in runs_dir.glob(prefix + "_attempt_*"):
        match = re.search(r"_attempt_(\d+)$", existing.name)
        if match:
            attempt_numbers.append(int(match.group(1)))
    next_attempt = max(attempt_numbers, default=0) + 1
    artifact_dir = runs_dir / f"{prefix}_attempt_{next_attempt:02d}"
    artifact_dir.mkdir(parents=True, exist_ok=False)
    config = {
        "condition": condition,
        "task_id": task_id,
        "seed": seed,
        "num_envs": 128,
        "steps_per_env_per_iteration": 24,
        "training_segment_iterations": int(cfg.agent.max_iterations),
        "prior_completed_iterations": prior_completed_iterations,
        "total_completed_iterations_target": 1001,
        "total_transitions_target": 1001 * 128 * 24,
        "resume_replayed_updates_from_stale_attempt": 79
        if condition == "joint_phasor" and seed == 20261032 and prior_completed_iterations == 901
        else 0,
        "terrain": "flat plane",
        "encoder_path": str(ENCODER) if condition == "joint_phasor" else None,
        "encoder_sha256": encoder_hash if condition == "joint_phasor" else None,
        "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint is not None else None,
        "resume_checkpoint_sha256": sha256_file(resume_checkpoint)
        if resume_checkpoint is not None
        else None,
        "agent_config": asdict(cfg.agent),
        "environment_config_source": "scripts.train TrainConfig.from_task + frozen PHASOR v3 path override",
        "source_sha256": {name: sha256_file(ROOT / name) for name in SOURCE_FILES},
        "argv": [sys.executable, str(Path(__file__).resolve()), "train-one", condition, str(seed)],
        **git_provenance(),
    }
    write_json(artifact_dir / "config.json", config)
    manifest = {
        "schema": "phasor.ppo_pilot_run.v1",
        **config,
        "status": "running",
        "started_at": now(),
        "process_id": os.getpid(),
        "artifact_dir": str(artifact_dir),
        "log_dir": str(log_dir),
        "stdout": str(artifact_dir / "stdout.log"),
        "stderr": str(artifact_dir / "stderr.log"),
        "runtime_workaround": "ASCII DOS-device path mapping for MuJoCo XML and Warp NVRTC; TEMP/TMP and Warp kernel cache use ASCII paths.",
    }
    write_json(artifact_dir / "manifest.json", manifest)

    import mjlab.tasks  # noqa: F401
    import src.tasks  # noqa: F401

    from mjlab.utils.torch import configure_torch_backends

    configure_torch_backends()
    os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"
    from scripts.train import run_train

    original_stdout, original_stderr = sys.stdout, sys.stderr
    try:
        with (artifact_dir / "stdout.log").open("w", encoding="utf-8") as out_file, (
            artifact_dir / "stderr.log"
        ).open("w", encoding="utf-8") as err_file:
            sys.stdout = _FilteredTee(original_stdout, out_file)
            sys.stderr = _FilteredTee(original_stderr, err_file)
            print(
                f"RUN_START condition={condition} seed={seed} segment_iterations={cfg.agent.max_iterations} "
                f"total_target=1001 log_dir={log_dir}",
                flush=True,
            )
            try:
                run_train(task_id, cfg, log_dir)
            finally:
                sys.stdout.flush()
                sys.stderr.flush()
    except BaseException as exc:
        manifest.update(
            {
                "status": "failed",
                "ended_at": now(),
                "exception": repr(exc),
                "traceback": traceback.format_exc(),
                "log_dir": str(log_dir),
            }
        )
        write_json(artifact_dir / "manifest.json", manifest)
        raise
    finally:
        sys.stdout, sys.stderr = original_stdout, original_stderr
        if resume_checkpoint is not None:
            from src.tasks.velocity.rl import VelocityOnPolicyRunner

            VelocityOnPolicyRunner.load = original_runner_load
            train_module.get_checkpoint_path = original_get_checkpoint_path
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    checkpoints = sorted(log_dir.glob("model_*.pt"))
    expected = log_dir / "model_1000.pt"
    if not expected.is_file():
        raise RuntimeError(f"Final training checkpoint was not produced: {expected}")
    final_state = torch.load(expected, map_location="cpu", weights_only=False)
    if int(final_state.get("iter", -1)) != 1000:
        raise RuntimeError(f"Expected checkpoint iteration 1000, got {final_state.get('iter')}")
    manifest.update(
        {
            "status": "completed",
            "ended_at": now(),
            "final_checkpoint": str(expected),
            "final_checkpoint_sha256": sha256_file(expected),
            "final_checkpoint_iter": int(final_state["iter"]),
            "checkpoint_files": [str(path) for path in checkpoints],
            "completed_training_iterations_total": 1001,
            "completed_transitions_total": 1001 * 128 * 24,
        }
    )
    write_json(artifact_dir / "manifest.json", manifest)
    print(f"RUN_COMPLETED condition={condition} seed={seed} checkpoint={expected}", flush=True)
    return manifest


def train_all() -> None:
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    expected_order = [(name, int(seed)) for name, seed in prereg["run_order"]]
    manifests = []
    manifest_by_condition_seed: dict[str, str] = {}
    errors = []
    previous_summary_path = PILOT / "training_summary.json"
    attempt_number = 1
    if previous_summary_path.is_file():
        previous = json.loads(previous_summary_path.read_text(encoding="utf-8"))
        attempt_number = int(previous.get("attempt", 1)) + 1
        archived_summary = PILOT / f"training_summary_attempt_{attempt_number - 1:02d}.json"
        if not archived_summary.exists():
            archived_summary.write_bytes(previous_summary_path.read_bytes())
    started_at = now()

    def persist_summary(status: str = "running") -> None:
        summary = {
            "schema": "phasor.ppo_pilot_training_summary.v1",
            "attempt": attempt_number,
            "started_at": started_at,
            "updated_at": now(),
            "run_order": [list(item) for item in expected_order],
            "completed_runs": len(manifest_by_condition_seed),
            "failed_runs": errors,
            "status": status,
            "manifest_by_condition_seed": manifest_by_condition_seed,
            "manifests": [
                manifest_by_condition_seed.get(f"{condition}:{seed}")
                for condition, seed in expected_order
            ],
        }
        write_json(PILOT / "training_summary.json", summary)

    def training_process_is_active(pid: int | None) -> bool:
        if pid is None or pid == os.getpid():
            return False
        try:
            import psutil

            process = psutil.Process(pid)
            command = " ".join(process.cmdline()).lower()
            return process.is_running() and "run_pilot.py" in command and "train" in command
        except Exception:
            return False

    def find_prior_attempt(condition: str, seed: int) -> tuple[Path | None, dict[str, Any] | None]:
        prefix = f"pilot_{condition}_seed_{seed}"
        candidates = sorted(
            (PILOT / "runs").glob(prefix + "*/manifest.json"),
            key=lambda path: path.parent.name,
        )
        for path in reversed(candidates):
            try:
                manifest = json.loads(path.read_text(encoding="utf-8"))
                return path, manifest
            except (OSError, json.JSONDecodeError):
                continue
        return None, None

    persist_summary()
    for index, (condition, seed) in enumerate(expected_order, start=1):
        print(f"PILOT_RUN {index}/{len(expected_order)}: {condition}, seed={seed}", flush=True)
        resume_checkpoint = None
        prior_completed_iterations = 0
        try:
            prior_path, prior_manifest = find_prior_attempt(condition, seed)
            if prior_path is not None and prior_manifest is not None:
                prior_status = prior_manifest.get("status")
                if prior_status == "completed":
                    checkpoint_path = Path(prior_manifest["final_checkpoint"])
                    if checkpoint_path.is_file() and sha256_file(checkpoint_path) == prior_manifest.get(
                        "final_checkpoint_sha256"
                    ):
                        manifests.append(prior_manifest)
                        manifest_by_condition_seed[f"{condition}:{seed}"] = str(prior_path)
                        print(f"RUN_ALREADY_COMPLETED verified={checkpoint_path}", flush=True)
                        persist_summary()
                        continue
                    raise RuntimeError(f"Completed run has a missing or changed checkpoint: {prior_path}")

                if prior_status == "running":
                    prior_pid = prior_manifest.get("process_id")
                    if training_process_is_active(int(prior_pid) if prior_pid is not None else None):
                        raise RuntimeError(
                            f"Prior training process {prior_pid} is still active; refusing duplicate PPO run"
                        )
                    stale_log_dir = Path(prior_manifest.get("log_dir", ""))
                    stale_checkpoints = list(stale_log_dir.glob("model_*.pt")) if stale_log_dir.is_dir() else []
                    if stale_checkpoints:
                        import torch

                        checkpoint_records = []
                        for checkpoint_path in stale_checkpoints:
                            try:
                                state = torch.load(
                                    checkpoint_path, map_location="cpu", weights_only=False
                                )
                                checkpoint_records.append(
                                    (int(state.get("iter", -1)), checkpoint_path)
                                )
                            except Exception:
                                continue
                        if checkpoint_records:
                            last_iteration, resume_checkpoint = max(checkpoint_records)
                            prior_completed_iterations = last_iteration + 1
                    prior_manifest.update(
                        {
                            "status": "interrupted",
                            "ended_at": now(),
                            "interruption_reason": "Training process was absent on restart; preserved as an incomplete attempt.",
                            "last_recoverable_checkpoint": str(resume_checkpoint)
                            if resume_checkpoint is not None
                            else None,
                            "last_recoverable_checkpoint_iter": prior_completed_iterations - 1
                            if resume_checkpoint is not None
                            else None,
                        }
                    )
                    write_json(prior_path, prior_manifest)

            if resume_checkpoint is None and condition == "baseline" and seed == 20261031:
                resume_checkpoint = PARTIAL_BASELINE_CKPT
                checkpoint_state = __import__("torch").load(
                    resume_checkpoint, map_location="cpu", weights_only=False
                )
                if int(checkpoint_state.get("iter", -1)) != 400:
                    raise RuntimeError("The saved partial baseline is not model_400.pt")
                prior_completed_iterations = 401

            manifest = train_one(
                condition,
                seed,
                resume_checkpoint=resume_checkpoint,
                prior_completed_iterations=prior_completed_iterations,
            )
            manifests.append(manifest)
            manifest_by_condition_seed[f"{condition}:{seed}"] = str(
                Path(manifest["artifact_dir"]) / "manifest.json"
            )
            persist_summary()
        except BaseException as exc:
            errors.append({"condition": condition, "seed": seed, "error": repr(exc)})
            print(f"RUN_FAILED condition={condition} seed={seed}: {exc!r}", flush=True)
            traceback.print_exc()
            persist_summary()
            if "torch" in sys.modules and sys.modules["torch"].cuda.is_available():
                sys.modules["torch"].cuda.empty_cache()
    persist_summary("completed" if not errors and len(manifest_by_condition_seed) == len(expected_order) else "partial_failed")
    if errors:
        raise RuntimeError(f"Training had {len(errors)} failed run(s)")


def eval_all() -> None:
    np, torch = import_project()
    encoder_hash = encoder_hash_check()
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    episode_seeds = [int(seed) for seed in prereg["evaluation"]["episode_seeds"]]
    commands = [float(value) for value in prereg["evaluation"]["forward_velocity_commands_mps"]]
    episodes_per_policy = len(episode_seeds) * len(commands)

    from dataclasses import asdict
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

    result_rows: list[dict[str, Any]] = []
    for condition in ("baseline", "joint_phasor"):
        for seed in (20261031, 20261032, 20261033):
            run_prefix = f"pilot_{condition}_seed_{seed}"
            training_summary = json.loads(
                (PILOT / "training_summary.json").read_text(encoding="utf-8")
            )
            manifest_path = Path(
                training_summary["manifest_by_condition_seed"][f"{condition}:{seed}"]
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            checkpoint_path = Path(manifest["final_checkpoint"])
            if not checkpoint_path.is_file():
                raise FileNotFoundError(checkpoint_path)
            if sha256_file(checkpoint_path) != manifest["final_checkpoint_sha256"]:
                raise RuntimeError(f"Checkpoint hash changed after training: {checkpoint_path}")
            output_dir = PILOT / "evaluations" / run_prefix
            existing_summary = output_dir / "summary.json"
            if existing_summary.is_file():
                try:
                    cached = json.loads(existing_summary.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    cached = {}
                if (
                    cached.get("schema") == "phasor.ppo_pilot_evaluation.v1"
                    and cached.get("checkpoint_sha256") == manifest["final_checkpoint_sha256"]
                    and len(cached.get("episodes", [])) == episodes_per_policy
                ):
                    result_rows.append(
                        {
                            "condition": condition,
                            "ppo_seed": seed,
                            "primary_outcome_mae_mps": float(cached["primary_outcome_mae_mps"]),
                            "secondary_rmse_mps": float(cached["secondary_rmse_mps"]),
                            "fall_fraction": float(cached["fall_fraction"]),
                            "summary": str(existing_summary),
                        }
                    )
                    print(
                        f"EVAL_ALREADY_COMPLETED condition={condition} ppo_seed={seed} "
                        f"mae={cached['primary_outcome_mae_mps']:.5f}",
                        flush=True,
                    )
                    write_json(
                        PILOT / "evaluation_progress.json",
                        {
                            "schema": "phasor.ppo_pilot_evaluation_progress.v1",
                            "status": "running",
                            "updated_at": now(),
                            "completed_policies": result_rows,
                        },
                    )
                    continue
            if output_dir.exists():
                archive_dir = output_dir.with_name(
                    output_dir.name + "_interrupted_" + datetime.now().strftime("%Y%m%d_%H%M%S")
                )
                output_dir.rename(archive_dir)
            output_dir.mkdir(parents=True, exist_ok=False)
            task_id = task_for(condition)
            env_cfg = load_env_cfg(task_id, play=True)
            configure_phase_encoder(env_cfg, condition)
            env_cfg.scene.num_envs = 1
            env_cfg.episode_length_s = float(prereg["evaluation"]["episode_horizon_seconds"])
            command_cfg = env_cfg.commands["twist"]
            command_cfg.heading_command = False
            command_cfg.ranges.heading = None
            command_cfg.rel_standing_envs = 0.0
            command_cfg.resampling_time_range = (
                env_cfg.episode_length_s + 1.0,
                env_cfg.episode_length_s + 1.0,
            )
            command_cfg.ranges.lin_vel_y = (0.0, 0.0)
            command_cfg.ranges.ang_vel_z = (0.0, 0.0)
            env_cfg.seed = seed

            agent_cfg = load_rl_cfg(task_id)
            agent_cfg.seed = seed
            raw_env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0")
            wrapped_env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
            runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
            runner = runner_cls(wrapped_env, asdict(agent_cfg), device="cuda:0")
            runner.load(
                str(checkpoint_path),
                load_cfg={"actor": True},
                strict=True,
                map_location="cuda:0",
            )
            policy = runner.get_inference_policy(device="cuda:0")
            dt = float(raw_env.step_dt)
            expected_dt = 1.0 / float(prereg["evaluation"]["control_hz"])
            if not math.isclose(dt, expected_dt, rel_tol=0.0, abs_tol=1e-9):
                raise RuntimeError(f"Evaluation requires 50 Hz, got step_dt={dt}")
            horizon_steps = int(round(env_cfg.episode_length_s / dt))
            if horizon_steps != 1000:
                raise RuntimeError(f"Expected 1000 steps for a 20 s episode, got {horizon_steps}")

            episode_rows = []
            raw_errors = []
            raw_velocities = []
            raw_commands = []
            terminal_flags = []
            terminal_steps = []
            robot = raw_env.scene["robot"]
            try:
                for episode_seed in episode_seeds:
                    for target_vx in commands:
                        command_cfg.ranges.lin_vel_x = (target_vx, target_vx)
                        random.seed(episode_seed)
                        np.random.seed(episode_seed)
                        torch.manual_seed(episode_seed)
                        torch.cuda.manual_seed_all(episode_seed)
                        raw_env.reset(seed=episode_seed)
                        active_command = raw_env.command_manager.get_command("twist")
                        if active_command is None:
                            raise RuntimeError("Evaluation command 'twist' is missing")
                        if not torch.allclose(
                            active_command,
                            torch.tensor([[target_vx, 0.0, 0.0]], device=active_command.device),
                            rtol=0.0,
                            atol=1e-6,
                        ):
                            raise RuntimeError(f"Fixed command mismatch: {active_command}")

                        velocities = []
                        errors = []
                        actual_commands = []
                        fell = False
                        fall_step = None
                        for step in range(horizon_steps):
                            command = raw_env.command_manager.get_command("twist")
                            if command is None:
                                raise RuntimeError("Evaluation command disappeared")
                            current_command = float(command[0, 0].item())
                            current_vx = float(robot.data.root_link_lin_vel_b[0, 0].item())
                            actual_commands.append(current_command)
                            velocities.append(current_vx)
                            errors.append(abs(current_vx - current_command))
                            obs = wrapped_env.get_observations()
                            with torch.inference_mode():
                                action = policy(obs)
                            _, _, terminated, truncated, _ = raw_env.step(action)
                            if bool(terminated[0].item()):
                                fell = True
                                fall_step = step + 1
                                # A terminated robot contributes zero forward mobility for
                                # the remainder of the fixed 20-second scoring horizon.
                                remaining = horizon_steps - len(errors)
                                if remaining > 0:
                                    errors.extend([abs(target_vx)] * remaining)
                                    velocities.extend([0.0] * remaining)
                                    actual_commands.extend([target_vx] * remaining)
                                break
                            if bool(truncated[0].item()) and step + 1 < horizon_steps:
                                raise RuntimeError(
                                    f"Unexpected time truncation at step {step + 1}/{horizon_steps}"
                                )

                        errors_np = np.asarray(errors, dtype=np.float64)
                        velocities_np = np.asarray(velocities, dtype=np.float64)
                        commands_np = np.asarray(actual_commands, dtype=np.float64)
                        if len(errors_np) != horizon_steps:
                            raise RuntimeError("An evaluation cell did not cover 20 seconds")
                        episode_row = {
                            "condition": condition,
                            "ppo_seed": seed,
                            "episode_seed": episode_seed,
                            "command_vx_mps": target_vx,
                            "steps": horizon_steps,
                            "duration_seconds": horizon_steps * dt,
                            "mae_forward_velocity_mps": float(errors_np.mean()),
                            "rmse_forward_velocity_mps": float(np.sqrt(np.mean(errors_np**2))),
                            "fall": fell,
                            "fall_step": fall_step,
                            "fall_mobility_scoring": "zero forward velocity for remaining horizon after termination",
                            "mean_measured_forward_velocity_mps": float(velocities_np.mean()),
                            "max_command_deviation_mps": float(
                                np.max(np.abs(commands_np - target_vx))
                            ),
                        }
                        episode_rows.append(episode_row)
                        raw_errors.append(errors_np.astype(np.float32))
                        raw_velocities.append(velocities_np.astype(np.float32))
                        raw_commands.append(commands_np.astype(np.float32))
                        terminal_flags.append(fell)
                        terminal_steps.append(-1 if fall_step is None else fall_step)
                        print(
                            f"EVAL_CELL condition={condition} ppo_seed={seed} episode_seed={episode_seed} "
                            f"vx={target_vx:.2f} mae={episode_row['mae_forward_velocity_mps']:.5f} "
                            f"fall={fell}",
                            flush=True,
                        )
            finally:
                wrapped_env.close()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()

            if len(episode_rows) != episodes_per_policy:
                raise RuntimeError(f"Expected {episodes_per_policy} evaluation cells")
            condition_score = float(
                np.mean([row["mae_forward_velocity_mps"] for row in episode_rows])
            )
            summary = {
                "schema": "phasor.ppo_pilot_evaluation.v1",
                "condition": condition,
                "ppo_seed": seed,
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": manifest["final_checkpoint_sha256"],
                "encoder_sha256": encoder_hash if condition == "joint_phasor" else None,
                "evaluation": prereg["evaluation"],
                "episodes": episode_rows,
                "primary_outcome_mae_mps": condition_score,
                "secondary_rmse_mps": float(
                    np.mean([row["rmse_forward_velocity_mps"] for row in episode_rows])
                ),
                "fall_fraction": float(np.mean(terminal_flags)),
                "raw_data": "episode_raw.npz",
                "evaluated_at": now(),
                "evaluation_script_sha256": sha256_file(Path(__file__).resolve()),
            }
            np.savez_compressed(
                output_dir / "episode_raw.npz",
                episode_seed=np.asarray([r["episode_seed"] for r in episode_rows], dtype=np.int64),
                command_vx_mps=np.asarray([r["command_vx_mps"] for r in episode_rows], dtype=np.float32),
                measured_forward_velocity_mps=np.stack(raw_velocities),
                absolute_forward_velocity_error_mps=np.stack(raw_errors),
                command_forward_velocity_mps=np.stack(raw_commands),
                fall=np.asarray(terminal_flags, dtype=bool),
                fall_step=np.asarray(terminal_steps, dtype=np.int32),
            )
            write_json(output_dir / "summary.json", summary)
            result_rows.append(
                {
                    "condition": condition,
                    "ppo_seed": seed,
                    "primary_outcome_mae_mps": condition_score,
                    "secondary_rmse_mps": summary["secondary_rmse_mps"],
                    "fall_fraction": summary["fall_fraction"],
                    "summary": str(output_dir / "summary.json"),
                }
            )
            write_json(
                PILOT / "evaluation_progress.json",
                {
                    "schema": "phasor.ppo_pilot_evaluation_progress.v1",
                    "status": "running",
                    "updated_at": now(),
                    "completed_policies": result_rows,
                },
            )

    write_json(
        PILOT / "evaluation_summary.json",
        {
            "schema": "phasor.ppo_pilot_evaluation_set.v1",
            "status": "completed",
            "evaluated_at": now(),
            "cells_per_policy": episodes_per_policy,
            "policy_results": result_rows,
            "encoder_scope": prereg["encoder_scope"],
        },
    )


def _power_two_sided(n: int, sd: float, effect: float) -> float:
    from scipy.stats import nct, t

    if sd == 0.0:
        return 1.0
    df = n - 1
    critical = float(t.ppf(0.975, df))
    ncp = effect * math.sqrt(n) / sd
    return float(nct.cdf(-critical, df, -ncp) + nct.sf(critical, df, ncp))


def analyze() -> None:
    import numpy as np
    from scipy.optimize import brentq
    from scipy.stats import t

    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    evaluation = json.loads((PILOT / "evaluation_summary.json").read_text(encoding="utf-8"))
    validation_path = (
        ROOT
        / "results/phasor_contact_supervised_v3/run_20261016/encoder_training_v3_attempt3/validation_summary.json"
    )
    locked_test_path = (
        ROOT
        / "results/phasor_contact_supervised_v3/run_20261016/encoder_training_v3_attempt3/locked_test_summary.json"
    )
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    locked_test = json.loads(locked_test_path.read_text(encoding="utf-8"))
    validation_metrics = validation["metrics"]
    locked_test_metrics = locked_test["metrics"]
    rows = evaluation["policy_results"]
    scores = {(r["condition"], int(r["ppo_seed"])): float(r["primary_outcome_mae_mps"]) for r in rows}
    seeds = [int(s) for s in prereg["paired_ppo_seeds"]]
    differences = np.asarray(
        [scores[("joint_phasor", s)] - scores[("baseline", s)] for s in seeds], dtype=float
    )
    n = len(differences)
    mean = float(np.mean(differences))
    sd = float(np.std(differences, ddof=1)) if n > 1 else float("nan")
    sem = sd / math.sqrt(n)
    ci_critical = float(t.ppf(0.975, n - 1))
    ci = [mean - ci_critical * sem, mean + ci_critical * sem]
    minimum_meaningful = float(prereg["analysis"]["minimum_meaningful_improvement_mps"])

    if sd == 0.0:
        lambda_80 = 0.0
    else:
        from scipy.stats import nct

        critical = float(t.ppf(0.975, n - 1))

        def power_for_lambda(value: float) -> float:
            return float(nct.cdf(-critical, n - 1, -value) + nct.sf(critical, n - 1, value))

        lambda_80 = float(brentq(lambda value: power_for_lambda(value) - 0.8, 0.0, 100.0))
    mde_80 = float(lambda_80 * sd / math.sqrt(n)) if sd > 0.0 else 0.0

    required_n = None
    if sd == 0.0:
        required_n = 2
    else:
        for candidate in range(2, 10001):
            if _power_two_sided(candidate, sd, minimum_meaningful) >= 0.8:
                required_n = candidate
                break

    all_improved = bool(np.all(differences < 0.0))
    expansion_rule = bool(
        mde_80 <= minimum_meaningful and mean <= -minimum_meaningful and all_improved
    )
    report = {
        "schema": "phasor.ppo_pilot_analysis.v1",
        "analysis_time": now(),
        "primary_outcome": prereg["evaluation"]["primary_outcome"],
        "paired_seed_differences_joint_minus_baseline_mps": {
            str(seed): float(diff) for seed, diff in zip(seeds, differences, strict=True)
        },
        "mean_paired_difference_mps": mean,
        "sample_sd_of_paired_differences_mps": sd,
        "paired_95_percent_t_interval_mps": ci,
        "n_paired_seeds": n,
        "80_percent_power_two_sided_mde_at_n3_mps": mde_80,
        "minimum_meaningful_improvement_mps": minimum_meaningful,
        "paired_seed_count_for_80_percent_power_at_target": required_n,
        "all_three_seed_differences_favor_joint": all_improved,
        "expansion_rule_passed": expansion_rule,
        "formal_matrix_recommendation": "proceed" if expansion_rule else "stop after pilot",
        "condition_scores": [
            {
                **r,
                "ppo_seed": int(r["ppo_seed"]),
            }
            for r in rows
        ],
        "statistical_scope": "Exploratory paired-seed pilot; evaluation episodes are repeated measures, not independent PPO replicates.",
        "encoder_scope": prereg["encoder_scope"],
        "encoder_checkpoint": str(ENCODER),
        "encoder_sha256": encoder_hash_check(),
        "gate_evidence": {
            "validation_summary": str(validation_path),
            "validation_summary_sha256": sha256_file(validation_path),
            "validation_split_seed": validation["validation_simulator_seed"],
            "validation_contact_agreement_macro": validation_metrics["contact_agreement_macro_by_trajectory"],
            "validation_reconstruction_mse": validation_metrics["validation_reconstruction_mse"],
            "validation_full_cycle_coverage": validation_metrics["all_trajectories_have_full_time_ordered_cycle"],
            "locked_test_summary": str(locked_test_path),
            "locked_test_summary_sha256": sha256_file(locked_test_path),
            "locked_test_split_seed": locked_test["simulator_seed"],
            "locked_test_contact_agreement_macro": locked_test_metrics["contact_agreement_macro_by_trajectory"],
            "locked_test_reconstruction_mse": locked_test_metrics["validation_reconstruction_mse"],
            "locked_test_full_cycle_coverage": locked_test_metrics["all_trajectories_have_full_time_ordered_cycle"],
            "encoder_training_simulator_seeds": validation["training_simulator_seeds"],
            "encoder_training_seed": validation["training_seed"],
        },
        **git_provenance(),
    }
    write_json(PILOT / "analysis.json", report)
    lines = [
        "# PHASOR PPO pilot 结果分析",
        "",
        f"分析时间：{report['analysis_time']}",
        "",
        "这是一个 3 对 PPO 种子的探索性 pilot。每个训练策略在 3 个固定速度命令 × 3 个 episode seed 上评估；统计单位仍是配对 PPO 训练 seed。",
        "",
        f"主指标为 9 个评估单元等权平均绝对前向速度误差（m/s），越低越好。joint − baseline 的逐 seed 差值为：{', '.join(f'{seed}: {diff:+.5f}' for seed, diff in zip(seeds, differences, strict=True))}。",
        "",
        f"平均配对差值 = {mean:+.5f} m/s；配对差值样本标准差 = {sd:.5f} m/s；95% 配对 t 区间 = [{ci[0]:+.5f}, {ci[1]:+.5f}] m/s。负值有利于 joint PHASOR。",
        f"n=3、双侧 80% power 的可检测效应约为 {mde_80:.5f} m/s；按观察到的配对差值标准差，检测 0.05 m/s 目标所需约 {required_n} 个配对 PPO 种子。",
        "",
        f"预注册扩展规则结果：{'通过' if expansion_rule else '未通过'}。建议：{'进入正式矩阵' if expansion_rule else '停止在 pilot，不启动正式矩阵'}。",
        "",
        "## 结果解释",
        "",
    ]
    if expansion_rule:
        lines.append("三个 seed 的差值方向一致、平均改善达到 0.05 m/s，且 pilot 的 80% power 可检测效应不超过 0.05 m/s；按预注册规则可开展正式矩阵。该规则只决定是否扩展，不构成确认性有效性结论。")
    else:
        lines.append("当前 pilot 没有满足预注册的全部扩展条件。即使点估计方向有利或达到阈值，n=3 的不确定性仍按配对差值标准差和置信区间报告；本结果不支持确认性疗效结论。")
    lines.extend(
        [
            "",
            "## 冻结 encoder 的证据边界",
            "",
            f"PHASOR 结果以单个冻结 encoder checkpoint 为条件：`{ENCODER_SHA256}`。3 个 PPO seeds 只覆盖 PPO 训练随机性，不覆盖 encoder 预训练随机性；因此不得把这 3 对结果描述成 3 个完整方法重复。",
            f"phase/contact 门使用独立划分：encoder 训练 simulator seeds={validation['training_simulator_seeds']}，validation seed={validation['validation_simulator_seed']}，locked-test seed={locked_test['simulator_seed']}。validation contact agreement={validation_metrics['contact_agreement_macro_by_trajectory']:.4f}、重构 MSE={validation_metrics['validation_reconstruction_mse']:.4f}；locked test agreement={locked_test_metrics['contact_agreement_macro_by_trajectory']:.4f}、重构 MSE={locked_test_metrics['validation_reconstruction_mse']:.4f}；两边都完整覆盖时间有序周期。该门通过只验证 latent/contact 对应关系，不代表策略性能提升。",
            "",
            "## 评估说明",
            "",
            "评估按预注册固定命令、episode seeds、20 秒 horizon 和 50 Hz 执行。终止摔倒后，为保持 20 秒固定评分窗口，余下时段按零前向移动计入主指标，并在 `fall_fraction` 单独报告。",
            "训练预算是 1,001 次 PPO 更新、每次 128×24 transitions；不能与历史 10,001 iteration 结果直接比较。",
            "",
            "原始逐 episode 结果和时序数组分别保存在 `evaluations/` 下的 `summary.json` 与 `episode_raw.npz`；本次分析不修改论文文件。",
        ]
    )
    (PILOT / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((PILOT / "analysis.md").read_text(encoding="utf-8"), flush=True)


def preflight() -> None:
    np, torch = import_project()
    encoder_hash = encoder_hash_check()
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg

    observed = {}
    for condition in ("baseline", "joint_phasor"):
        task_id = task_for(condition)
        cfg = load_env_cfg(task_id)
        configure_phase_encoder(cfg, condition)
        cfg.scene.num_envs = 1
        cfg.seed = 20261031
        raw_env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
        wrapped = RslRlVecEnvWrapper(raw_env, clip_actions=0.0)
        try:
            raw_env.reset(seed=20261031)
            obs = wrapped.get_observations()
            actor = obs["actor"] if hasattr(obs, "keys") and "actor" in obs.keys() else obs
            action = torch.zeros((1, wrapped.action_space.shape[-1]), device="cuda:0")
            raw_env.step(action)
            observed[condition] = {
                "task_id": task_id,
                "step_dt": float(raw_env.step_dt),
                "actor_observation_shape": list(actor.shape),
                "action_shape": list(action.shape),
            }
        finally:
            wrapped.close()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    if observed["baseline"]["actor_observation_shape"] != observed["joint_phasor"]["actor_observation_shape"]:
        raise RuntimeError(f"Actor observation shapes differ: {observed}")
    if not math.isclose(observed["baseline"]["step_dt"], 0.02, rel_tol=0.0, abs_tol=1e-9):
        raise RuntimeError(f"Training control rate is not 50 Hz: {observed}")
    report = {
        "schema": "phasor.pilot_runtime_preflight.v1",
        "status": "passed",
        "completed_at": now(),
        "conditions": observed,
        "encoder_sha256": encoder_hash,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "runtime_workaround": "ASCII DOS-device path mapping for MuJoCo XML and Warp NVRTC; TEMP/TMP and Warp kernel cache use ASCII paths.",
    }
    write_json(PILOT / "runtime_preflight.json", report)
    print(json.dumps(report, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("preflight", "train-all", "train-one", "eval-all", "analyze"))
    parser.add_argument("condition", nargs="?", choices=("baseline", "joint_phasor"))
    parser.add_argument("seed", nargs="?", type=int)
    args = parser.parse_args()
    try:
        if args.action == "preflight":
            preflight()
        elif args.action == "train-all":
            train_all()
        elif args.action == "train-one":
            if args.condition is None or args.seed is None:
                parser.error("train-one requires condition and seed")
            train_one(
                args.condition,
                args.seed,
            )
        elif args.action == "eval-all":
            eval_all()
        else:
            analyze()
    finally:
        _remove_dos_devices()


if __name__ == "__main__":
    main()
