"""Discover compatible study checkpoints and stage private resume snapshots."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import pickletools
import shutil
import zipfile

import yaml


def saved_iteration(path: Path) -> int:
  """Read integer metadata without executing pickle deserialization."""
  with zipfile.ZipFile(path) as archive:
    key = next(n for n in archive.namelist() if n.endswith("/data.pkl") or n == "data.pkl")
    if archive.getinfo(key).file_size > 10 * 1024**2:
      raise ValueError("Unexpected checkpoint metadata size.")
    operations = pickletools.genops(archive.read(key))
    for operation, value, _ in operations:
      if value == "iter" and operation.name in ("BINUNICODE", "SHORT_BINUNICODE", "UNICODE"):
        for next_operation, next_value, _ in operations:
          if next_operation.name in ("BINPUT", "LONG_BINPUT", "MEMOIZE"):
            continue
          if next_operation.name not in ("BININT", "BININT1", "BININT2", "LONG1", "LONG4", "INT"):
            raise ValueError("Checkpoint iteration is not an integer.")
          return int(next_value)
  raise ValueError("Checkpoint has no iteration metadata.")


def file_sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for block in iter(lambda: stream.read(1024**2), b""):
      digest.update(block)
  return digest.hexdigest()


def discover_job(root: Path, group: str, seed: int, reference, target: int, num_envs: int) -> dict:
  modes = {"A": "none", "B": "fixed", "C": "global", "D": "selective"}
  log_root = root / "logs/rsl_rl/g1_dropout_style"
  candidates = []
  rejected = []
  for run in log_root.glob(f"*_style_pilot_{group}_seed{seed}"):
    try:
      env = yaml.load((run / "params/env.yaml").read_text(), Loader=yaml.BaseLoader)
      agent = yaml.load((run / "params/agent.yaml").read_text(), Loader=yaml.BaseLoader)
      command = env["commands"]["gait_reference"]
      if command["mode"] != modes[group] or int(agent["seed"]) != seed:
        raise ValueError("Group/seed contract differs.")
      if int(env["scene"]["num_envs"]) != num_envs or int(agent["num_steps_per_env"]) != 24:
        raise ValueError("Parallel environments or rollout length differs from this study.")
      if int(agent["max_iterations"]) != target:
        raise ValueError("Target budget differs from the saved study configuration.")
      provenance = run / "reference_provenance.json"
      if provenance.exists():
        expected_hash = json.loads(provenance.read_text())["sha256"]
      else:
        # The original interrupted queue stored its provenance once for all
        # four arms. Require both its frozen hash and the saved reference path.
        manifest = root / "results/style_dropout_training/launch_source_manifest.json"
        expected_hash = json.loads(manifest.read_text())["reference_sha256"]
        if Path(command["reference_file"]).resolve() != reference.path.resolve():
          raise ValueError("Reference path differs in an interrupted legacy run.")
      if expected_hash != reference.sha256:
        raise ValueError("Reference hash differs; do not mix datasets when resuming.")
      checkpoints = sorted(run.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[-1]))
      if not checkpoints:
        continue
      checkpoint = checkpoints[-1]
      iteration = saved_iteration(checkpoint)
      if iteration != int(checkpoint.stem.split("_")[-1]):
        raise ValueError("Checkpoint filename and internal iteration disagree.")
      candidates.append((iteration, checkpoint))
    except (OSError, ValueError, KeyError, StopIteration, zipfile.BadZipFile) as error:
      rejected.append({"run": str(run), "reason": str(error)})
  if candidates:
    iteration, checkpoint = max(candidates, key=lambda item: item[0])
    return {"group": group, "seed": seed, "action": "skip" if iteration + 1 >= target else "resume",
            "completed_iterations": iteration + 1, "target_iterations": target,
            "checkpoint": str(checkpoint), "rejected_runs": rejected}
  if rejected:
    raise ValueError(f"Existing {group}/seed{seed} checkpoints are incompatible: {rejected}")
  return {"group": group, "seed": seed, "action": "new", "completed_iterations": 0,
          "target_iterations": target, "checkpoint": None, "rejected_runs": []}


def stage_private_checkpoint(source: Path) -> dict:
  """Create an owned mode-0600 copy in a mode-0700 Linux directory."""
  private = Path.home() / ".local/share/unitree_style_training/checkpoints"
  private.mkdir(parents=True, exist_ok=True)
  if private.is_symlink():
    raise PermissionError("Private checkpoint storage cannot be a symlink.")
  private.chmod(0o700)
  digest = file_sha256(source)
  target = private / f"{digest}.pt"
  if target.exists():
    if target.is_symlink() or target.stat().st_uid != os.getuid() or file_sha256(target) != digest:
      raise PermissionError("Existing private checkpoint snapshot failed integrity checks.")
    target.chmod(0o600)
  else:
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as output, source.open("rb") as stream:
      shutil.copyfileobj(stream, output)
    if file_sha256(target) != digest:
      raise ValueError("Checkpoint snapshot hash mismatch.")
  return {"source": str(source), "private_snapshot": str(target), "sha256": digest,
          "iteration": saved_iteration(target)}
