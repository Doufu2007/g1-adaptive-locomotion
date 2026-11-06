#!/usr/bin/env python3
from __future__ import annotations
import json, os, random, shutil, subprocess, sys, time, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
NS = "formal_recovery_gpu_01"
CONTROL_ROOT = ROOT / "g1_phase_supplement_20261031/formal_recovery_01"
CONTROL = CONTROL_ROOT / NS
OUT = ROOT / "remote_output" / NS
SOURCE_CONTROL = CONTROL_ROOT / "formal_recovery_06"
SOURCE_OUT = ROOT / "remote_output/formal_recovery_06"
SELF = Path(__file__).resolve()
GPU_SOURCE = ROOT / "scripts/g1_phase_supplement_20261031_gpu.py"
OLD_PROTOCOL = SOURCE_CONTROL / "protocol.json"

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("PHASOR_EVAL_DEVICE", "cuda:0")
os.environ.setdefault("MUJOCO_GL", "glfw")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")
os.environ.setdefault("MUJOCO_GLFW_VISIBLE", "0")
os.environ.pop("PYOPENGL_PLATFORM", None)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf8")
    tmp.replace(path)


def sha(path):
    import hashlib
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def setup():
    import remote_phase_supplement_adapter_20261104_04 as adapter
    adapter.install_runtime_hooks()
    from scripts import g1_phase_supplement_20261031_gpu as supplement
    supplement.CONTROL = CONTROL
    supplement.OUT = OUT
    supplement.ROOT = ROOT
    supplement.a.sha = adapter.remote_sha

    def verify_seal_gpu():
        value = read(CONTROL / "protocol.json")
        for entry in value["files"]:
            if adapter.remote_sha(entry["path"]) != entry["sha256"]:
                raise ValueError("sealed input changed: " + entry["path"])
        return value

    def check_disk_gpu():
        import shutil as _shutil
        OUT.mkdir(parents=True, exist_ok=True)
        for path, minimum in ((ROOT, 10), (OUT, 15)):
            if _shutil.disk_usage(path).free < minimum * 1024 ** 3:
                raise ValueError(f"{path}: below locked {minimum}GiB free disk floor")

    def child_args(command, **kwargs):
        args = [sys.executable, "-X", "utf8", str(SELF), command]
        for key, value in kwargs.items():
            if value is True:
                args.append("--" + key.replace("_", "-"))
            elif value is not None and value is not False:
                args.extend(["--" + key.replace("_", "-"), str(value)])
        return args

    def run_child(command, log, **kwargs):
        check_disk_gpu()
        env = os.environ.copy()
        env.update(PYTHONHASHSEED="20271700", PYTHONUTF8="1", PYTHONIOENCODING="utf-8",
                   OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", PHASOR_EVAL_DEVICE="cuda:0",
                   CUDA_VISIBLE_DEVICES="0", MUJOCO_GL="glfw", MUJOCO_EGL_DEVICE_ID="0")
        env.pop("PYOPENGL_PLATFORM", None)
        log = Path(log); log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("x", encoding="utf8") as stream:
            proc = subprocess.Popen(child_args(command, **kwargs), cwd=ROOT, env=env,
                                    stdin=subprocess.DEVNULL, stdout=stream,
                                    stderr=subprocess.STDOUT, start_new_session=True)
            write(CONTROL / "child_process.json", {"pid": proc.pid, "command": command,
                "at": time.time(), "log": str(log), "device": "cuda:0"})
            rc = proc.wait()
        write(CONTROL / "child_exit.json", {"pid": proc.pid, "command": command,
            "exit_code": rc, "at": time.time(), "device": "cuda:0"})
        if rc:
            raise RuntimeError(f"{command} failed with exit code {rc}; preserved {log}")

    supplement.verify_seal = verify_seal_gpu
    supplement.check_disk = check_disk_gpu
    supplement.child_args = child_args
    supplement.run_child = run_child
    return supplement


def prepare():
    if CONTROL.exists() or OUT.exists():
        raise RuntimeError(f"{NS} already exists")
    if not SOURCE_CONTROL.exists() or not SOURCE_OUT.exists():
        raise FileNotFoundError("source formal_recovery_06 missing")
    CONTROL.mkdir(parents=True)
    OUT.mkdir(parents=True)
    # Reuse only the completed GPU-trained policies; all evaluation outputs are fresh.
    shutil.copytree(SOURCE_OUT / "training", OUT / "training", copy_function=os.link)
    old = read(OLD_PROTOCOL)
    old_sha = sha(OLD_PROTOCOL)
    entries = []
    for item in old["files"]:
        if item["path"].replace("\\", "/").endswith("/scripts/g1_phase_supplement_20261031.py"):
            continue
        entries.append(item)
    for path in (GPU_SOURCE, SELF):
        entries.append({"path": str(path), "sha256": sha(path)})
    protocol = dict(old)
    protocol.update({
        "status": "LOCKED_BEFORE_GPU_EVALUATION",
        "source_protocol_sha256": old_sha,
        "files": entries,
        "environment": dict(old.get("environment", {}), evaluation_device="cuda:0",
                              policy_inference_device="cpu:0",
                              training_device="cuda:0",
                              reproducibility="GPU environment evaluation with CPU ONNX policy inference; independent GPU rerun required"),
        "gpu_runtime": {"torch_cuda": True, "onnxruntime_provider": "CPUExecutionProvider",
                        "cuda_visible_devices": "0", "mujoco_gl": "egl"},
        "scientific_parameters_changed": False,
    })
    write(CONTROL / "protocol.json", protocol)
    write(CONTROL / "gpu_plan.json", {
        "status": "PREPARED", "namespace": NS, "source_namespace": "formal_recovery_06",
        "evaluation_device": "cuda:0", "policy_inference_device": "cpu:0",
        "reuse": "GPU-trained policies only; calibration/validation/test rerun fresh",
        "automatic_retry": False, "source_protocol_sha256": old_sha,
    })
    print(json.dumps({"status": "PREPARED", "control": str(CONTROL), "output": str(OUT)}, ensure_ascii=False))


def preflight():
    S = setup()
    if not (CONTROL / "gpu_plan.json").exists():
        raise RuntimeError("prepare must run first")
    S.check_disk(); S.verify_seal(); S.verify_training()
    import torch, onnxruntime as ort
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    device_name = torch.cuda.get_device_name(0)
    write(CONTROL / "gpu_preflight.json", {
        "status": "PASS", "at": time.time(), "device": "cuda:0",
        "device_name": device_name, "onnxruntime_providers": ort.get_available_providers(),
        "formal_sampling_started": False, "training_reused": 24,
    })
    print(json.dumps({"status": "PASS", "device": "cuda:0", "device_name": device_name}, ensure_ascii=False))


def queue():
    S = setup(); preflight(); C = S.c
    status = {"status": "RUNNING", "started_at": time.time(), "completed_jobs": 0,
              "stage": "calibration", "device": "cuda:0", "automatic_retry": False}
    write(CONTROL / "queue_status.json", status)
    def run_cell(split, mode, terrain, seed):
        name = f"{split}_{mode}_{terrain}_{seed}"
        folder = S.rollout_folder(split, mode, terrain, seed)
        manifest = folder / "manifest.json"
        if manifest.exists() and read(manifest).get("status") == "COMPLETED":
            return False
        if folder.exists():
            raise RuntimeError(f"unfinished folder requires preservation: {folder}")
        status.update(stage=split, current=name, at=time.time())
        write(CONTROL / "queue_status.json", status)
        S.verify_seal()
        S.run_child("rollout", OUT / "logs" / (name + ".log"), split=split, mode=mode,
                    terrain=terrain, seed=seed)
        status["completed_jobs"] += 1
        return True
    try:
        for t in C.MAPS["calibration"]:
            for s in C.NEW_SEEDS:
                run_cell("calibration", "live", t, s)
        if not (OUT / "phase_bank/manifest.json").exists():
            status.update(stage="bank", current=None, at=time.time()); write(CONTROL / "queue_status.json", status)
            S.build_bank()
        else:
            S.verify_bank()
        for t in C.MAPS["validation"]:
            for s in C.NEW_SEEDS:
                for mode in ("clock55", "clock60"):
                    run_cell("validation", mode, t, s)
        selected = S.select_clock() if not (OUT / "clock_selection.json").exists() else S.selection()
        write(OUT / "test_inputs_locked.json", {"at": time.time(),
            "bank_manifest_sha256": sha(OUT / "phase_bank/manifest.json"),
            "selection_sha256": sha(OUT / "clock_selection.json")})
        for t in C.MAPS["test"]:
            for s in C.NEW_SEEDS:
                modes = ["live", "constant", "replay", selected]
                random.Random(t * 100 + s).shuffle(modes)
                for mode in modes:
                    run_cell("test", mode, t, s)
        status.update(stage="analyze", current=None, at=time.time()); write(CONTROL / "queue_status.json", status)
        S.analyze()
        status.update(stage="audit", at=time.time()); write(CONTROL / "queue_status.json", status)
        S.audit()
        status.update(status="COMPLETED", ended_at=time.time(), current=None)
        write(CONTROL / "queue_status.json", status)
        write(CONTROL / "gpu_complete.json", {"status": "COMPLETED", "at": time.time(),
            "device": "cuda:0", "policy_inference_device": "cpu:0", "analysis": str(CONTROL / "analysis.json"),
            "independent_audit": str(CONTROL / "independent_audit.json")})
    except BaseException as exc:
        status.update(status="FAILED", ended_at=time.time(), error=repr(exc))
        write(CONTROL / "queue_status.json", status)
        write(CONTROL / "gpu_failure.json", {"status": "FAILED", "at": time.time(),
            "error": repr(exc), "traceback": traceback.format_exc(), "automatic_retry": False})
        raise


def launch():
    setup(); preflight()
    if (CONTROL / "launch.json").exists(): raise RuntimeError("duplicate launch forbidden")
    write(CONTROL / "launch_reserved.json", {"at": time.time(), "device": "cuda:0"})
    log = CONTROL / "queue_console.log"
    env = os.environ.copy(); env.update(PHASOR_EVAL_DEVICE="cuda:0", CUDA_VISIBLE_DEVICES="0", MUJOCO_GL="glfw", MUJOCO_EGL_DEVICE_ID="0")
    with log.open("x", encoding="utf8") as stream:
        proc = subprocess.Popen([sys.executable, "-u", str(SELF), "queue"], cwd=ROOT, env=env,
                                stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    write(CONTROL / "launch.json", {"at": time.time(), "pid": proc.pid, "device": "cuda:0", "output": str(OUT)})
    print(json.dumps({"pid": proc.pid, "output": str(OUT), "device": "cuda:0"}, ensure_ascii=False), flush=True)


def main():
    if len(sys.argv) < 2: raise SystemExit("usage: prepare|preflight|queue|launch|rollout")
    cmd = sys.argv[1]
    if cmd in {"prepare", "preflight", "queue", "launch"}:
        globals()[cmd](); return
    S = setup(); S.main()

if __name__ == "__main__":
    main()
