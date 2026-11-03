"""Cooperative training throttling, telemetry and bounded thermal pauses."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import time


class TrainingStopRequested(RuntimeError):
  """A checkpoint should be saved before leaving the training process."""

  def __init__(self, message: str, *, category: str = "manual"):
    super().__init__(message)
    self.category = category


@dataclass(frozen=True)
class ResourceLimits:
  cpu_threads: int = 2
  duty_cycle: float = 0.5
  minimum_rest_s: float = 1.0
  pause_temperature_c: float = 75.0
  resume_temperature_c: float = 68.0
  maximum_gpu_memory_mib: int = 4096
  minimum_available_ram_mib: int = 3072
  maximum_pause_s: float = 900.0
  critical_temperature_c: float = 83.0
  monitor_interval_s: float = 3.0
  minimum_host_available_ram_mib: int = 4096
  minimum_start_host_available_ram_mib: int = 7168

  def __post_init__(self):
    numeric = [self.duty_cycle, self.minimum_rest_s, self.pause_temperature_c,
               self.resume_temperature_c, self.maximum_pause_s,
               self.critical_temperature_c, self.monitor_interval_s]
    if not all(math.isfinite(value) for value in numeric):
      raise ValueError("Resource limits must be finite.")
    if not 1 <= self.cpu_threads <= 8 or not 0.1 <= self.duty_cycle <= 1.0:
      raise ValueError("Use 1..8 CPU threads and 0.1..1.0 active duty cycle.")
    if not 45 <= self.resume_temperature_c < self.pause_temperature_c <= 85:
      raise ValueError("Thermal resume must be below pause, between 45 and 85 C.")
    if self.minimum_rest_s < 0 or self.maximum_pause_s <= 0:
      raise ValueError("Invalid pause duration.")
    if not self.pause_temperature_c < self.critical_temperature_c <= 87:
      raise ValueError("Critical temperature must exceed pause and be at most 87 C.")
    if not 0.25 <= self.monitor_interval_s <= 5:
      raise ValueError("Resource monitoring interval must be 0.25..5 seconds.")
    if self.maximum_gpu_memory_mib < 1024 or self.minimum_available_ram_mib < 1024:
      raise ValueError("Memory thresholds must reserve a meaningful amount of memory.")
    if self.minimum_host_available_ram_mib < 2048:
      raise ValueError("Keep at least 2 GiB of Windows host memory available.")
    if self.minimum_start_host_available_ram_mib < self.minimum_host_available_ram_mib + 1024:
      raise ValueError("Startup memory must include at least 1 GiB above the runtime reserve.")

  @classmethod
  def from_environment(cls):
    return cls(**json.loads(os.environ.get("G1_RESOURCE_LIMITS_JSON", "{}")))


def rest_duration(active_s: float, limits: ResourceLimits) -> float:
  return max(limits.minimum_rest_s, max(0.0, active_s) * (1 / limits.duty_cycle - 1))


def read_gpu() -> dict:
  result = subprocess.run(
    ["nvidia-smi", "--query-gpu=temperature.gpu,memory.used,memory.total,utilization.gpu",
     "--format=csv,noheader,nounits"], capture_output=True, text=True,
    timeout=5, check=True,
  )
  values = [float(x.strip()) for x in result.stdout.strip().splitlines()[0].split(",")]
  return dict(zip(("temperature_c", "memory_used_mib", "memory_total_mib", "utilization_percent"), values))


def available_ram_mib() -> float:
  available = None
  for line in Path("/proc/meminfo").read_text().splitlines():
    if line.startswith("MemAvailable:"):
      available = int(line.split()[1]) / 1024
      break
  if available is None:
    raise RuntimeError("Cannot determine available Linux memory.")
  # A container's /proc/meminfo can describe the entire physical server.
  # Honor the effective allocation rather than spending another tenant's RAM.
  for limit_name, usage_name in (
    ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
    ("/sys/fs/cgroup/memory/memory.limit_in_bytes", "/sys/fs/cgroup/memory/memory.usage_in_bytes"),
  ):
    limit_path, usage_path = Path(limit_name), Path(usage_name)
    if limit_path.is_file() and usage_path.is_file():
      value = limit_path.read_text().strip()
      if value != "max":
        limit = int(value)
        if limit < 2**60:
          remaining = max(0, limit - int(usage_path.read_text().strip())) / 1024**2
          available = min(available, remaining)
  return available


def read_host_memory() -> dict:
  release = Path("/proc/sys/kernel/osrelease").read_text().lower()
  if "microsoft" not in release and "wsl" not in release:
    return {"host_available_ram_mib": available_ram_mib(), "memory_backend": "linux_cgroup"}
  probe = Path(__file__).resolve().parents[4] / "tools/StyleTrainingControl/HostMemoryProbe.exe"
  if not probe.is_file():
    raise OSError(f"Windows memory probe is missing: {probe}")
  result = subprocess.run([str(probe)], capture_output=True, text=True, timeout=5, check=True)
  data = json.loads(result.stdout)
  available = float(data["host_available_ram_mib"])
  if not math.isfinite(available) or available < 0:
    raise ValueError("Invalid host memory sample.")
  return data


def write_state(path: Path, data: dict) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  payload = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
  descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
  temporary = Path(name)
  try:
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
      stream.write(payload)
    # Windows readers or scanners can briefly deny rename on a DrvFS file.
    # Retain atomic publication and give those handles a bounded time to close.
    for attempt in range(15):
      try:
        temporary.replace(path)
        return
      except PermissionError:
        if attempt == 14:
          raise
        time.sleep(min(0.025 * (attempt + 1), 0.2))
  finally:
    temporary.unlink(missing_ok=True)


class ResourceController:
  def __init__(self, limits: ResourceLimits, log_directory: Path, shared_directory: Path):
    self.limits = limits
    self.log_directory = log_directory.resolve()
    self.directory = shared_directory.resolve()
    self.directory.mkdir(parents=True, exist_ok=True)
    self.stop_path = self.directory / "stop.request"
    self.cooling = False
    self.failures = 0
    self.iteration = -1
    self.total_iterations = 0
    self._last_sample = None
    self._last_sample_time = float("-inf")

  def snapshot(self, status: str, **extra) -> dict:
    if self._last_sample is not None and "telemetry" not in extra:
      extra["telemetry"] = self._last_sample
    state = {
      "updated_at_utc": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(),
      "status": status, "iteration": self.iteration, "total_iterations": self.total_iterations,
      "group": os.environ.get("G1_ACTIVE_STYLE_GROUP"),
      "seed": os.environ.get("G1_ACTIVE_STYLE_SEED"),
      "log_directory": str(self.log_directory), "limits": asdict(self.limits), **extra,
    }
    write_state(self.directory / "resource_state.json", state)
    write_state(self.log_directory / "resource_state.json", state)
    return state

  def ensure_continue(self):
    if self.stop_path.exists():
      raise TrainingStopRequested("Manual stop request received.")

  def sample(self, *, force: bool = False) -> dict:
    if (not force and self._last_sample is not None
        and time.monotonic() - self._last_sample_time < self.limits.monitor_interval_s):
      return self._last_sample
    try:
      sample = {**read_gpu(), **read_host_memory(), "available_ram_mib": available_ram_mib(),
                "sampled_at_utc": datetime.now(timezone.utc).isoformat()}
      self.failures = 0
      self._last_sample = sample
      self._last_sample_time = time.monotonic()
      return sample
    except (OSError, ValueError, IndexError, subprocess.SubprocessError) as error:
      self.failures += 1
      if self.failures >= 3:
        raise TrainingStopRequested(f"Resource monitoring failed three times: {error}", category="monitor") from error
      return {"monitor_error": str(error)}

  def pause_reason(self, sample: dict) -> str | None:
    if "monitor_error" in sample:
      return "monitor_retry"
    if sample["memory_used_mib"] > self.limits.maximum_gpu_memory_mib:
      # Pausing cannot release model allocations; leave room for the user's app.
      raise TrainingStopRequested("GPU memory budget exceeded; saving and releasing the GPU.", category="resource")
    if sample["available_ram_mib"] < self.limits.minimum_available_ram_mib:
      raise TrainingStopRequested("Available RAM below reserve; saving and releasing resources.", category="resource")
    if sample.get("host_available_ram_mib", float("inf")) < self.limits.minimum_host_available_ram_mib:
      host = "Linux container" if sample.get("memory_backend") == "linux_cgroup" else "Windows host"
      raise TrainingStopRequested(f"{host} RAM below reserve; saving and releasing resources.", category="resource")
    temperature = sample["temperature_c"]
    if temperature >= self.limits.critical_temperature_c:
      raise TrainingStopRequested("Critical GPU temperature reached; saving and releasing resources.", category="thermal")
    if temperature >= self.limits.pause_temperature_c:
      self.cooling = True
    if self.cooling and temperature <= self.limits.resume_temperature_c:
      self.cooling = False
    return "cooling" if self.cooling else None

  def before_allocation(self, total_iterations: int):
    """Wait for simulator headroom before allocating its host/CUDA resources."""
    self.total_iterations = total_iterations
    started = time.monotonic()
    while True:
      self.ensure_continue()
      sample = self.sample(force=True)
      reason = self.pause_reason(sample)
      if reason is None:
        if (sample.get("host_available_ram_mib", float("inf")) < self.limits.minimum_start_host_available_ram_mib
            or sample["available_ram_mib"] < self.limits.minimum_available_ram_mib + 1024):
          reason = "waiting_memory"
        elif sample["memory_used_mib"] > self.limits.maximum_gpu_memory_mib - 1024:
          reason = "waiting_gpu_memory"
      if reason is None:
        self.snapshot("building_environment", telemetry=sample)
        return
      if time.monotonic() - started > self.limits.maximum_pause_s:
        raise TrainingStopRequested("Startup resource wait exceeded its time limit; no simulator was started.", category="resource")
      self.snapshot(reason, telemetry=sample, waiting_seconds=time.monotonic() - started)
      time.sleep(self.limits.monitor_interval_s)

  def after_iteration(self, active_s: float, iteration: int, total_iterations: int):
    self.iteration = iteration
    self.total_iterations = total_iterations
    self.ensure_continue()
    delay = rest_duration(active_s, self.limits)
    started = time.monotonic()
    deadline = started + delay
    sample = self.sample()
    while True:
      self.ensure_continue()
      reason = self.pause_reason(sample)
      remaining = deadline - time.monotonic()
      if reason is None and remaining <= 0:
        break
      if reason is not None and time.monotonic() - started > self.limits.maximum_pause_s:
        raise TrainingStopRequested("Thermal/monitor pause exceeded its bounded waiting period.", category="thermal")
      self.snapshot(reason or "duty_pause", telemetry=sample,
                    active_seconds=active_s, requested_rest_seconds=delay)
      time.sleep(min(5.0 if reason else 1.0, max(0.1, remaining)) if not reason else 5.0)
      if reason or time.monotonic() >= deadline:
        sample = self.sample(force=bool(reason))
    state = self.snapshot("training", telemetry=sample,
                          active_seconds=active_s, actual_rest_seconds=time.monotonic() - started)
    path = self.log_directory / "resource_telemetry.csv"
    fields = ["updated_at_utc", "iteration", "active_seconds", "actual_rest_seconds",
              "temperature_c", "memory_used_mib", "utilization_percent", "available_ram_mib",
              "host_available_ram_mib"]
    row = {**state, **sample}
    fresh = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as stream:
      writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
      if fresh:
        writer.writeheader()
      writer.writerow(row)
