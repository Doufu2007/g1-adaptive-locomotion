"""Linux transport adapter around the byte-identical frozen stateful runner.

Only path representation, simulator bootstrap, and child launch identity change.
Scientific model, losses, seeds, sampling, budgets and test lock remain sealed.
"""
from pathlib import Path
import json
import os
import subprocess
import sys
import time

SOURCE_ROOT = r"<local-project-root>/phasor"
ROOT = Path(__file__).resolve().parents[1]
MODULE = "scripts.run_g1_stateful_remote_20261025"


class PortableRoot(type(Path())):
    def __truediv__(self, value):
        text = str(value).replace("\\", "/")
        source = SOURCE_ROOT.replace("\\", "/")
        if text == source:
            text = "."
        elif text.startswith(source + "/"):
            text = text[len(source) + 1:]
        return super().__truediv__(text)


def bootstrap():
    """POSIX equivalent of the recorded pilot import_project/bootstrap calls."""
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import numpy
    import torch
    import warp
    if not hasattr(warp, "context"):
        import warp._src.context as warp_context
        warp.context = warp_context
    from mjlab.utils.torch import configure_torch_backends
    configure_torch_backends()
    torch.set_num_threads(2)
    import src.tasks  # registers the same task ids, imports MuJoCo before EGL
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["MUJOCO_EGL_DEVICE_ID"] = os.environ.get("LOCAL_RANK", "0")
    return numpy, torch


def matching_process(pid, script):
    import psutil
    try:
        identity = Path(script).stem
        accepted = {identity}
        if identity in ("run_g1_stateful_phase_20261025", "run_g1_stateful_remote_20261025"):
            accepted |= {"run_g1_stateful_phase_20261025", "run_g1_stateful_remote_20261025"}
        return any(Path(token).stem in accepted or token.rsplit(".", 1)[-1] in accepted
                   for token in psutil.Process(int(pid)).cmdline())
    except (psutil.NoSuchProcess, psutil.AccessDenied, TypeError, ValueError):
        return False


def resume_command(row, seed):
    return [sys.executable, "-m", MODULE, "--stage", "train", "--row", row, "--seed", str(seed)]


def native_frame_count(make_batch):
    """Keep NumPy cycle indices from leaking into JSON frame counters."""
    def wrapped(*args, **kwargs):
        *tensors, frames = make_batch(*args, **kwargs)
        return (*tensors, int(frames))
    return wrapped


def child(stage, **kwargs):
    from scripts import run_g1_stateful_phase_20261025 as runner
    command = [sys.executable, "-u", "-m", MODULE, "--stage", stage]
    for key, value in kwargs.items():
        command += ["--" + key.replace("_", "-"), str(value)]
    name = "_".join([stage] + [str(v) for v in kwargs.values()])
    log = runner.OUT / "logs" / f"{name}_{time.time_ns()}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as stream:
        os.environ["PYTHONHASHSEED"] = str(kwargs.get("seed", 20261025))
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        runner.write(runner.OUT / "queue_status.json", dict(status="RUNNING", stage=stage, kwargs=kwargs,
            child_pid=process.pid, argv=command, log_path=str(log), updated_at=runner.now(),
            execution_host="<ssh-host>:23827", adapter=MODULE))
        exit_code = process.wait()
    if exit_code:
        raise RuntimeError(f"child failed ({exit_code}): {log}")


def install():
    from scripts import run_g1_stateful_phase_20261025 as runner
    from scripts import g1_phase_adaptation_20261024 as adaptation
    from scripts import run_g1_best_phase_control as base
    from scripts import g1_best_phase_batched as batch
    # Existing serialized Windows paths are translated at the root join only.
    for module in (runner, adaptation, base, batch):
        module.ROOT = PortableRoot(ROOT)
    runner.matching_process = matching_process
    runner.training_resume_command = resume_command
    runner.child = child
    runner.core.make_batch = native_frame_count(runner.core.make_batch)
    base.bootstrap = adaptation.bootstrap = batch.bootstrap = bootstrap
    return runner


def main():
    runner = install()
    # A migration seal covers this additive adapter; the two original seals stay intact.
    seal_path = runner.OUT / "remote_migration/adapter_seal.json"
    seal = runner.read(seal_path)
    if runner.core.sha(Path(__file__)) != seal["adapter_sha256"]:
        raise ValueError("remote adapter changed after migration seal")
    runner.main()


if __name__ == "__main__":
    main()
