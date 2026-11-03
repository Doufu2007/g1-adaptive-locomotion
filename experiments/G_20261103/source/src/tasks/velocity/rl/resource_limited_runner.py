"""The style study's PPO runner with cooperative low-load operation."""

from pathlib import Path
import os
import time

import torch

from .runner import VelocityOnPolicyRunner
from .resource_control import ResourceController, ResourceLimits, TrainingStopRequested


class ResourceLimitedVelocityRunner(VelocityOnPolicyRunner):
  def __init__(self, env, train_cfg, log_dir=None, device="cpu", **kwargs):
    limits = ResourceLimits.from_environment()
    torch.set_num_threads(limits.cpu_threads)
    if os.environ.get("G1_CLOUD_MODE") != "1" and hasattr(os, "nice") and os.getpriority(os.PRIO_PROCESS, 0) < 10:
      os.nice(10 - os.getpriority(os.PRIO_PROCESS, 0))
    super().__init__(env, train_cfg, log_dir, device, **kwargs)
    if log_dir is None:
      self.resource_controller = None
      return
    directory = Path(os.environ.get("G1_RESOURCE_DIRECTORY", "results/style_dropout_training"))
    self.resource_controller = ResourceController(limits, Path(log_dir), directory)
    self.resource_controller.total_iterations = train_cfg.get("max_iterations", 0)
    self.resource_controller.snapshot("initializing_policy")
    from src.tasks.velocity.mdp.style_reference import reference_term
    from .resource_control import write_state
    reference = reference_term(env.unwrapped).reference
    write_state(Path(log_dir) / "reference_provenance.json", {
      "reference": str(reference.path), "sha256": reference.sha256,
      "provenance": reference.provenance,
      "purpose": os.environ.get("G1_RUN_PURPOSE", "SOFTWARE_TEST_ONLY" if reference.provenance["kind"] == "synthetic_test" else "DEVELOPMENT_TRAINING"),
    })
    original_log = self.logger.log

    def limited_log(**values):
      original_log(**values)
      self.resource_controller.after_iteration(
        values["collect_time"] + values["learn_time"], values["it"], values["total_it"]
      )

    self.logger.log = limited_log

  def learn(self, *args, **kwargs):
    try:
      ready = os.environ.get("G1_BENCH_READY_FILE")
      release = os.environ.get("G1_BENCH_RELEASE_FILE")
      if ready and release:
        Path(ready).write_text("ready\n")
        started = time.monotonic()
        while not Path(release).exists():
          if self.resource_controller:
            self.resource_controller.ensure_continue()
          if time.monotonic() - started > 600:
            raise RuntimeError("Benchmark readiness barrier timed out.")
          time.sleep(0.2)
      result = super().learn(*args, **kwargs)
      if self.resource_controller is not None:
        self.resource_controller.snapshot("completed")
      return result
    except TrainingStopRequested as error:
      path = Path(self.logger.log_dir) / f"model_{self.current_learning_iteration}.pt"
      self.save(str(path))
      self.logger.stop_logging_writer()
      self.resource_controller.snapshot("stopped", reason=str(error), stop_category=error.category, checkpoint=str(path))
      raise
    except Exception as error:
      # A failed CUDA context must not be used to write new model tensors.
      if self.resource_controller is not None:
        checkpoints = list(Path(self.logger.log_dir).glob("model_*.pt"))
        checkpoint = max(checkpoints, key=lambda p: int(p.stem.split("_")[-1])) if checkpoints else None
        self.resource_controller.snapshot(
          "failed", reason=f"{type(error).__name__}: {error}",
          checkpoint=str(checkpoint) if checkpoint else None,
        )
      raise
