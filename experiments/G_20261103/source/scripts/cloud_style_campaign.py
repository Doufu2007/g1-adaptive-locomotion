"""Bounded cloud deployment, synchronized throughput probes, and resumable jobs."""
from __future__ import annotations

import argparse
import atexit
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "results/cloud_campaign"
PYTHON = sys.executable
TRAINING_DEADLINE = "2026-11-04T08:00:00Z"
EVALUATION_DEADLINE = "2026-11-04T10:00:00Z"
EXPORT_DEADLINE = "2026-11-04T10:30:00Z"


def publish(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def stamp():
    return datetime.now(timezone.utc).isoformat()


def environment(threads=4):
    env = os.environ.copy()
    env.update(G1_CLOUD_MODE="1", PYTHONUNBUFFERED="1", OMP_NUM_THREADS=str(threads),
               OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS=str(threads), MUJOCO_GL="egl",
               WANDB_MODE="offline", PYTHONPATH=str(ROOT),
               XDG_CACHE_HOME=str(ROOT.parent/'cache'),
               TORCHINDUCTOR_CACHE_DIR=str(ROOT.parent/'cache/torchinductor'))
    return env


def resource_arguments(directory, threads=4):
    return ["--cpu-threads", str(threads), "--duty-cycle", "1", "--minimum-rest-s", "0",
            "--gpu-memory-budget-mib", "28672", "--minimum-host-available-ram-mib", "8192",
            "--minimum-start-host-available-ram-mib", "12288", "--resource-directory", str(directory)]


def job_plan():
    sys.path.insert(0, str(ROOT))
    from scripts.style_resume import discover_job
    from src.tasks.velocity.mdp.style_reference import load_gait_reference
    reference = load_gait_reference(ROOT / "data/style_dropout/walking_reference.npz")
    jobs = []
    for seed in range(42, 47):
        groups = list("ABCD")
        groups = groups[seed % 4:] + groups[:seed % 4]
        for group in groups:
            jobs.append(discover_job(ROOT, group, seed, reference, 2000, 1024))
    return jobs


def benchmark_worker(args):
    sys.path.insert(0, str(ROOT))
    os.environ["G1_RUN_PURPOSE"] = "PERFORMANCE_BENCHMARK_ONLY"
    import torch
    import mjlab.tasks
    import src.tasks
    from scripts.run_style_dropout import configure_reference
    from scripts.train import TrainConfig, launch_training
    from src.tasks.velocity.config.g1.style_dropout_cfg import STYLE_TASKS
    from src.tasks.velocity.mdp.style_reference import load_gait_reference
    from src.tasks.velocity.rl.resource_control import ResourceLimits
    limits = ResourceLimits(cpu_threads=args.threads, duty_cycle=1, minimum_rest_s=0,
                            maximum_gpu_memory_mib=28672, minimum_available_ram_mib=8192,
                            minimum_host_available_ram_mib=8192, minimum_start_host_available_ram_mib=12288)
    os.environ["G1_RESOURCE_LIMITS_JSON"] = json.dumps(limits.__dict__)
    os.environ["G1_RESOURCE_DIRECTORY"] = str(args.directory)
    os.environ["G1_ACTIVE_STYLE_GROUP"] = args.group
    os.environ["G1_ACTIVE_STYLE_SEED"] = "900042"
    torch.set_num_threads(args.threads)
    task = STYLE_TASKS[args.group][0]
    config = TrainConfig.from_task(task)
    reference = load_gait_reference(ROOT / "data/style_dropout/walking_reference.npz")
    configure_reference(config.env, reference)
    config.env.scene.num_envs = 1024
    config.env.seed = 900042
    config.agent.seed = 900042
    config.agent.max_iterations = args.iterations
    config.agent.save_interval = args.iterations + 1
    config.agent.logger = "tensorboard"
    config.agent.upload_model = False
    config.agent.experiment_name = "g1_dropout_style_CLOUD_BENCH"
    config.agent.run_name = args.directory.name
    launch_training(task, TrainConfig(env=config.env, agent=config.agent, gpu_ids=[0], enable_nan_guard=True))


def probe(parallel, iterations=80, threads=4):
    folder = CONTROL / f"benchmark_{parallel}p_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    folder.mkdir(parents=True)
    release = folder / "release"
    workers = []
    for index in range(parallel):
        group = "DABC"[index % 4]
        directory = folder / f"worker_{index}_{group}"
        directory.mkdir()
        env = environment(threads)
        env.update(G1_BENCH_READY_FILE=str(directory / "ready"), G1_BENCH_RELEASE_FILE=str(release))
        output = (directory / "worker.log").open("w")
        command = [PYTHON, __file__, "benchmark-worker", "--group", group, "--directory", str(directory),
                   "--iterations", str(iterations), "--threads", str(threads)]
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT)
        workers.append((process, directory, output))
    started = time.monotonic()
    try:
        while not all((d / "ready").exists() for _, d, _ in workers):
            if (CONTROL/'stop.request').exists():
                raise RuntimeError('Manual stop requested during benchmark initialization.')
            if any(p.poll() is not None for p, _, _ in workers):
                raise RuntimeError(f"Benchmark worker failed before readiness; inspect {folder}")
            if time.monotonic() - started > 900:
                raise TimeoutError("Benchmark initialization exceeded 15 minutes")
            time.sleep(2)
        warmup_seconds = time.monotonic() - started
        release.write_text(stamp())
        released = time.monotonic()
        while any(process.poll() is None for process, _, _ in workers):
            if (CONTROL / "stop.request").exists():
                raise RuntimeError("Manual stop requested during benchmark training.")
            if any(process.poll() not in (None, 0) for process, _, _ in workers):
                raise RuntimeError(f"Benchmark training failed; inspect {folder}")
            if time.monotonic() - released > 900:
                raise TimeoutError("Benchmark training exceeded 15 minutes")
            time.sleep(1)
        if any(process.returncode != 0 for process, _, _ in workers):
            raise RuntimeError(f"Benchmark training failed; inspect {folder}")
        wall = time.monotonic() - released
        results = []
        for _, directory, _ in workers:
            state = json.loads((directory / "resource_state.json").read_text())
            telemetry = list(csv.DictReader((Path(state["log_directory"]) / "resource_telemetry.csv").open()))
            steady = telemetry[min(15, len(telemetry)//3):]
            active = [float(row["active_seconds"]) for row in steady]
            timestamps = [datetime.fromisoformat(row["updated_at_utc"]).timestamp() for row in steady]
            intervals = [b-a for a, b in zip(timestamps, timestamps[1:])]
            results.append({"group": directory.name[-1], "updates": len(telemetry),
                            "median_compute_seconds": statistics.median(active),
                            "median_wall_seconds": statistics.median(intervals),
                            "p90_wall_seconds": sorted(intervals)[int(.9*(len(intervals)-1))],
                            "peak_gpu_memory_mib": max(float(row["memory_used_mib"]) for row in telemetry),
                            "max_temperature_c": max(float(row["temperature_c"]) for row in telemetry),
                            "run_directory": state["log_directory"]})
        result = {"parallel": parallel, "cpu_threads_per_job": threads, "num_envs_per_job": 1024,
                  "rollout_steps": 24, "iterations": iterations, "warmup_seconds": warmup_seconds,
                  "released_wall_seconds": wall,
                  "aggregate_steady_steps_per_second": sum(24576/r["median_wall_seconds"] for r in results),
                  "aggregate_end_to_end_steps_per_second": parallel*iterations*24576/wall,
                  "workers": results, "completed_at_utc": stamp()}
        publish(folder / "summary.json", result)
        return result
    finally:
        for process, directory, output in workers:
            if process.poll() is None:
                (directory / "stop.request").touch()
                try:
                    process.wait(timeout=90)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
            output.close()


def benchmark(args):
    existing = CONTROL / "benchmark.json"
    results = json.loads(existing.read_text())["results"] if existing.exists() else []
    if not results:
        results = [probe(1, args.iterations), probe(2, args.iterations)]
    if (len(results) == 2 and results[-1]["aggregate_steady_steps_per_second"] > results[0]["aggregate_steady_steps_per_second"]*1.10
            and max(w["peak_gpu_memory_mib"] for w in results[-1]["workers"]) < 20000):
        results.append(probe(3, args.iterations))
    if (len(results) == 3
            and results[-1]["aggregate_steady_steps_per_second"] > results[-2]["aggregate_steady_steps_per_second"] * 1.08
            and max(w["peak_gpu_memory_mib"] for w in results[-1]["workers"]) * 4 / 3 + 2048 < 28672):
        results.append(probe(4, args.iterations))
    # Use up to 24 of the instance's 25 vCPUs, retaining memory/thermal limits.
    # More processes are accepted only when their measured throughput helps.
    for parallel, threads in ((6, 4), (8, 3), (12, 2)):
        if any(r["parallel"] == parallel for r in results):
            continue
        previous = max(results, key=lambda r: r["parallel"])
        projected = max(w["peak_gpu_memory_mib"] for w in previous["workers"]) * parallel / previous["parallel"]
        if projected + 2048 >= 28672:
            break
        results.append(probe(parallel, args.iterations, threads))
        publish(existing, {"purpose":"PERFORMANCE_BENCHMARK_ONLY", "results":results,
                           "configuration_sweep_complete":False, "completed_at_utc":stamp()})
        if parallel >= 8 and results[-1]["aggregate_steady_steps_per_second"] < max(
                r["aggregate_steady_steps_per_second"] for r in results[:-1]) * 1.03:
            break
    best = max(results, key=lambda r: r["aggregate_steady_steps_per_second"])
    # Require a useful gain before adding a process and another CUDA context.
    if best["aggregate_steady_steps_per_second"] < results[0]["aggregate_steady_steps_per_second"]*1.08:
        best = results[0]
    report = {"purpose": "PERFORMANCE_BENCHMARK_ONLY", "results": results,
              "selected_parallel": best["parallel"], "selected": best,
              "configuration_sweep_complete": True,
              "completed_at_utc": stamp()}
    publish(CONTROL / "benchmark.json", report)
    print(json.dumps(report, indent=2), flush=True)


def time_plan(args):
    jobs = job_plan()
    benchmark_report = json.loads((CONTROL/'benchmark.json').read_text())
    selected = benchmark_report['selected']
    remaining = sum(max(0,2000-j['completed_iterations']) for j in jobs)
    rate = sum(1/w['median_wall_seconds'] for w in selected['workers'])
    conservative_rate = sum(1/w['p90_wall_seconds'] for w in selected['workers'])
    initialization = len([j for j in jobs if j['action']!='skip'])*selected['warmup_seconds']/selected['parallel']
    estimate = remaining/rate + initialization
    conservative = (remaining/conservative_rate+initialization)*1.25
    probe = CONTROL/'evaluation_speed_probe.json'
    probe_report = json.loads(probe.read_text()) if probe.exists() else {}
    case_seconds = probe_report.get('seconds_per_case',probe_report.get('total_wall_seconds',80))*1.3
    evaluation_seconds = case_seconds*180
    deadline = datetime.fromisoformat(args.deadline.replace('Z','+00:00')).timestamp()
    report = {'created_at_utc':stamp(),'remaining_updates':remaining,'remaining_environment_steps':remaining*24576,
              'completed_models':sum(j['action']=='skip' for j in jobs),'total_models':20,
              'selected_parallel':selected['parallel'],'estimated_training_hours':estimate/3600,
              'cpu_threads_per_job':selected['cpu_threads_per_job'],
              'conservative_training_hours':conservative/3600,'estimated_evaluation_hours':evaluation_seconds/3600,
              'estimated_full_1140_case_evaluation_hours':case_seconds*1140/3600,
              'core_evaluation_cases':180, 'extended_evaluation_cases':960,
              'training_deadline_utc':args.deadline,'evaluation_deadline_utc':args.evaluation_deadline,
              'final_export_deadline_utc':EXPORT_DEADLINE,
              'planning_priority':'Training throughput, matched experimental budgets, paired evaluation within the 48-hour instance window.',
              'fits_training_deadline_with_reserve':time.time()+conservative<deadline,
              'protocol':{'seeds':list(range(42,47)),'groups':list('ABCD'),'num_envs':1024,
                          'rollout_steps':24,'updates_per_model':2000},
              'scope':'Development comparison using one frozen walking reference; not a held-out paper validation.'}
    publish(CONTROL/'time_plan.json',report)
    print(json.dumps(report,indent=2),flush=True)
    return report


def backup():
    target = ROOT / "artifacts"
    target.mkdir(exist_ok=True)
    lock = (target / 'cloud_backup.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX)
    output = target / "cloud_latest_results.zip"
    temporary = target / "cloud_latest_results.tmp.zip"
    files = []
    models_to_write = {}
    log_root = ROOT / "logs/rsl_rl/g1_dropout_style"
    if log_root.exists():
        for run in log_root.iterdir():
            if not run.is_dir():
                continue
            models = sorted(run.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[-1]))
            for checkpoint in reversed(models):
                try:
                    data = checkpoint.read_bytes()
                    with zipfile.ZipFile(io.BytesIO(data)) as model_zip:
                        if model_zip.testzip() is not None: continue
                    models_to_write[checkpoint.relative_to(ROOT).as_posix()] = data
                    break
                except (OSError, zipfile.BadZipFile):
                    continue
            files.extend(p for p in run.rglob("*") if p.is_file() and p.suffix in (".yaml", ".json", ".csv", ".onnx"))
            files.extend(run.glob("events.out.tfevents.*"))
    for directory in (CONTROL, ROOT / "results/cloud_evaluation"):
        if directory.exists():
            files.extend(p for p in directory.rglob("*") if p.is_file() and p.suffix in (".json", ".csv"))
    for name in ('requirements-lock.txt','requirements-study.txt','deployment_manifest.json',
                 'deployment_cloud_changes.json','cloud_training_control.ipynb',
                 'data/style_dropout/walking_reference.npz','data/style_dropout/walking_reference.json'):
        path=ROOT/name
        if path.is_file(): files.append(path)
    files.extend(ROOT.glob('deployment_cloud*.json'))
    files.extend((ROOT/'src').rglob('*.py'))
    files.extend((ROOT/'scripts').glob('*.py'))
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for name,data in models_to_write.items(): archive.writestr(name,data)
        for path in dict.fromkeys(files):
            archive.write(path, str(path.relative_to(ROOT)))
    temporary.replace(output)
    publish(target / "cloud_latest_results.sha256.json", {
        "file": output.name, "sha256": hashlib.file_digest(output.open("rb"), "sha256").hexdigest(),
        "bytes": output.stat().st_size, "created_at_utc": stamp(), "files": len(files)+len(models_to_write)})
    return str(output)


def campaign(args):
    CONTROL.mkdir(parents=True, exist_ok=True)
    lock = (CONTROL / "campaign.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (CONTROL / "stop.request").exists():
        raise RuntimeError("A stop request exists. Explicitly archive it before starting again.")
    deadline = datetime.fromisoformat(args.deadline.replace("Z", "+00:00")).timestamp()
    benchmark_report = json.loads((CONTROL / "benchmark.json").read_text())
    parallel = args.parallel or benchmark_report["selected_parallel"]
    if not 1 <= parallel <= 12:
        raise ValueError("Parallel jobs must be 1..12")
    threads = benchmark_report['selected']['cpu_threads_per_job']
    plan = job_plan()
    pending = [j for j in plan if j["action"] != "skip"]
    running = []
    completed = [j for j in plan if j["action"] == "skip"]
    failures = []
    last_backup = 0.0
    started = stamp()
    stop = False
    stopping_since = None
    def cleanup():
        for process,directory,output,_ in list(running):
            if process.poll() is None:
                (directory/'stop.request').touch()
                try: process.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try: process.wait(timeout=20)
                    except subprocess.TimeoutExpired: process.kill(); process.wait()
            if not output.closed: output.close()
    atexit.register(cleanup)
    def request_stop(*_):
        (CONTROL / "stop.request").touch()
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    while pending or running:
        stop = stop or (CONTROL / "stop.request").exists() or time.time() >= deadline
        if stop:
            if stopping_since is None: stopping_since=time.monotonic()
            for _, directory, _, _ in running:
                (directory / "stop.request").touch()
            if time.monotonic()-stopping_since>180:
                for process,_,_,_ in running:
                    if process.poll() is None: process.terminate()
            if time.monotonic()-stopping_since>210:
                for process,_,_,_ in running:
                    if process.poll() is None: process.kill()
        while pending and len(running) < parallel and not stop:
            job = pending.pop(0)
            directory = CONTROL / f"{job['group']}_seed{job['seed']}"
            directory.mkdir(exist_ok=True)
            if (directory / "stop.request").exists():
                (directory / "stop.request").rename(directory / f"stop.archived.{time.time_ns()}")
            command = [PYTHON, "scripts/run_style_dropout.py", "--train", "--resume", "--groups", job["group"],
                       "--seeds", str(job["seed"]), "--num-envs", "1024", "--iterations", "2000",
                       "--report-file", str(directory / "preflight.json")] + resource_arguments(directory, threads)
            output = (directory / f"worker_{time.time_ns()}.log").open("w")
            process = subprocess.Popen(command, cwd=ROOT, env=environment(threads), stdout=output, stderr=subprocess.STDOUT)
            running.append((process, directory, output, job))
        for item in list(running):
            process, directory, output, job = item
            if process.poll() is not None:
                output.close()
                running.remove(item)
                result = {**job, "return_code": process.returncode, "finished_at_utc": stamp()}
                if process.returncode == 0:
                    completed.append(result)
                elif not (stop and process.returncode == 3):
                    failures.append(result)
                    stop = True
        live = []
        for process, directory, _, job in running:
            try:
                state = json.loads((directory / "resource_state.json").read_text())
            except (OSError, ValueError):
                state = {}
            live.append({"group": job["group"], "seed": job["seed"], "pid": process.pid, "state": state})
        status = "failed" if failures else "stopping" if stop else "training"
        publish(CONTROL / "campaign_state.json", {
            "status": status, "pid": os.getpid(), "started_at_utc": started, "updated_at_utc": stamp(),
            "deadline_utc": args.deadline, "parallel": parallel, "cpu_threads_per_job": threads, "planned_jobs": 20,
            "completed_jobs": len(completed), "pending_jobs": len(pending), "active": live,
            "completed": completed, "failures": failures,
            "protocol": {"seeds": list(range(42,47)), "groups": list("ABCD"), "num_envs":1024,
                         "rollout_steps":24, "updates_per_model":2000}})
        if time.monotonic() - last_backup > 900 or not running:
            backup()
            last_backup = time.monotonic()
        if stop and not running:
            break
        time.sleep(5)
    final = job_plan()
    state = json.loads((CONTROL / "campaign_state.json").read_text())
    state.update(status="completed" if all(j["action"] == "skip" for j in final) else "failed" if failures else "stopped",
                 final_jobs=final, finished_at_utc=stamp(), completed_jobs=sum(j["action"] == "skip" for j in final))
    publish(CONTROL / "campaign_state.json", state)
    backup()
    return 0 if state["status"] == "completed" else 3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["benchmark-worker", "benchmark", "plan", "time-plan", "run", "pipeline", "backup", "stop", "status"])
    parser.add_argument("--group", choices=list("ABCD"), default="D")
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--parallel", type=int, default=0)
    parser.add_argument("--deadline", default=TRAINING_DEADLINE)
    parser.add_argument("--evaluation-deadline", default=EVALUATION_DEADLINE)
    args = parser.parse_args()
    os.environ.update(environment(args.threads))
    os.chdir(ROOT)
    CONTROL.mkdir(parents=True, exist_ok=True)
    if args.action == "benchmark-worker": benchmark_worker(args)
    elif args.action == "benchmark": benchmark(args)
    elif args.action == "plan":
        jobs = job_plan()
        publish(CONTROL / "resume_plan.json", {"jobs":jobs, "created_at_utc":stamp()})
        print(json.dumps(jobs, indent=2))
    elif args.action == "run": return campaign(args)
    elif args.action == "time-plan": time_plan(args)
    elif args.action == "pipeline":
        pipeline_lock = (CONTROL/'pipeline.lock').open('a+')
        try:
            fcntl.flock(pipeline_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            print('The cloud pipeline is already running.', flush=True)
            return 0
        publish(CONTROL/'pipeline_state.json',{'pid':os.getpid(),'status':'benchmarking','updated_at_utc':stamp()})
        if not (CONTROL / "benchmark.json").exists() or not json.loads((CONTROL/'benchmark.json').read_text()).get('configuration_sweep_complete'):
            benchmark(args)
        if not (CONTROL / 'evaluation_speed_probe.json').exists() or json.loads((CONTROL/'evaluation_speed_probe.json').read_text()).get('metric_schema_version') != 2:
            subprocess.run([PYTHON,'scripts/cloud_evaluate_style.py','--probe'],cwd=ROOT,env=environment(),check=True)
        plan = time_plan(args)
        if not plan['fits_training_deadline_with_reserve']:
            raise RuntimeError('The full unchanged training budget does not fit the conservative deadline; revise the plan explicitly.')
        publish(CONTROL/'pipeline_state.json',{'pid':os.getpid(),'status':'training','updated_at_utc':stamp()})
        code = campaign(args)
        if code:
            state = json.loads((CONTROL / 'campaign_state.json').read_text())
            publish(CONTROL / 'pipeline_state.json', {
                'pid': os.getpid(), 'status': state['status'], 'updated_at_utc': stamp(),
                'reason': 'Training stopped before all matched-budget models completed.',
            })
            return code
        publish(CONTROL/'pipeline_state.json',{'pid':os.getpid(),'status':'evaluating','updated_at_utc':stamp()})
        eval_deadline=datetime.fromisoformat(args.evaluation_deadline.replace('Z','+00:00')).timestamp()
        try:
            code = subprocess.run([PYTHON, "scripts/cloud_evaluate_style.py", "--deadline", args.evaluation_deadline],cwd=ROOT,env=environment(),
                                  timeout=max(30,eval_deadline-time.time()+120)).returncode
        except subprocess.TimeoutExpired:
            code=3
        publish(CONTROL/'pipeline_state.json',{'pid':os.getpid(),'status':'completed' if code==0 else 'stopped','updated_at_utc':stamp()})
        backup()
        return code
    elif args.action == "backup": print(backup())
    elif args.action == "stop": (CONTROL / "stop.request").touch()
    elif args.action == "status": print((CONTROL / "campaign_state.json").read_text())
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        if len(sys.argv) > 1 and sys.argv[1] == "pipeline":
            publish(CONTROL / "pipeline_state.json", {
                "pid": os.getpid(), "status": "stopped" if (CONTROL / "stop.request").exists() else "failed",
                "updated_at_utc": stamp(), "reason": f"{type(error).__name__}: {error}",
            })
        raise
