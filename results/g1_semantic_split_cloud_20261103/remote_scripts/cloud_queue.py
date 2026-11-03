"""Run the frozen matrix while permitting the two identified vLLM services.

Only resource admission differs from the original standalone queue. Algorithm,
configuration, seeds, budgets, checkpoints and evaluation gates are unchanged.
"""
from datetime import datetime, timezone
import atexit
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

base = Path(__file__).resolve().parent
project = base/'g1_semantic_split'
sys.path.insert(0, str(project))
output = base/'experiment_4096'
output.mkdir(exist_ok=True)
lock = base/'cloud_scheduler.lock'
if lock.exists():
    previous = json.loads(lock.read_text())
    try:
        active = abs(psutil.Process(previous['pid']).create_time()-previous['created_at']) < .01
    except psutil.NoSuchProcess:
        active = False
    if active:
        raise RuntimeError('a live cloud scheduler already owns this experiment')
    lock.rename(base/f'cloud_scheduler_stale_{time.time_ns()}.json')
lock_identity = {'pid': os.getpid(), 'created_at': psutil.Process().create_time()}
with lock.open('x') as stream:
    json.dump(lock_identity, stream)


def release_lock():
    if lock.exists() and json.loads(lock.read_text()) == lock_identity:
        lock.unlink()


atexit.register(release_lock)
status_path = base/'cloud_queue_status.json'
state = {'status': 'WAITING_FOR_VALIDATION', 'pid': os.getpid(),
         'started_at': datetime.now(timezone.utc).isoformat(),
         'scheduler_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         'scope': 'four smoke, four pilot, twelve formal, seal, holdout, latency, analysis',
         'existing_services_stopped': False}


def save(**updates):
    state.update(updates, updated_at=datetime.now(timezone.utc).isoformat())
    temporary = status_path.with_suffix('.tmp')
    temporary.write_text(json.dumps(state, indent=2)+'\n')
    temporary.replace(status_path)


save()
while True:
    path = base/'validation_status.json'
    validation = json.loads(path.read_text()) if path.exists() else {}
    if validation.get('status') == 'FAILED':
        save(status='BLOCKED_VALIDATION_FAILED', validation=validation)
        raise SystemExit(1)
    if validation.get('status') == 'VALIDATED_AND_MEASURED':
        break
    time.sleep(5)

from scripts import g1_semantic_split_study as study
from scripts import run_g1_semantic_split as runner

allowed = {}
for pid in (17599, 18239):
    try:
        p = psutil.Process(pid)
        allowed[pid] = p.create_time()
    except psutil.NoSuchProcess:
        pass
probes = json.loads((base/'concurrency_probes.json').read_text())
full_probe = next((r for r in probes['records'] if r['num_envs'] == 4096 and r['status'] == 'PASSED'), None)
if probes['status'] != 'COMPLETED' or full_probe is None:
    save(status='BLOCKED_4096_ENV_PROBE_NOT_PASSED', probe_status=probes['status'])
    raise SystemExit(1)
required_train_mib = max(512, full_probe['peak_total_gpu_used_mib']-full_probe['before_gpu']['used_mib'])+512
save(allowed_existing_gpu_processes=allowed, training_required_free_mib=required_train_mib,
     resource_threshold_source='measured 4096-env peak plus 512MiB')


def wait_resources(required):
    while True:
        jobs = runner.gpu_processes()
        unexpected = [j for j in jobs if j['pid'] not in allowed
                      or abs(j.get('created_at', -1)-allowed[j['pid']]) > .01]
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                                capture_output=True, text=True, check=True)
        free = int(result.stdout.splitlines()[0])
        if not unexpected and free >= required:
            save(status='READY', free_gpu_mib=free, required_free_gpu_mib=required)
            return
        save(status='WAITING_FOR_GPU_CAPACITY', free_gpu_mib=free,
             required_free_gpu_mib=required, unexpected_gpu_jobs=unexpected)
        time.sleep(30)


try:
    protocol = runner.prepare(output)
    study.write(output/'cloud_launch.json', dict(state, allowed_existing_gpu_processes=allowed))
    for stage in ('smoke', 'pilot', 'formal'):
        jobs = ([dict(arm=a, seed=protocol['training']['pilot_seed']) for a in study.ARMS]
                if stage == 'smoke' else study.training_jobs(protocol, stage))
        for job in jobs:
            job = dict(job)
            job.pop('stage', None)
            directory = runner.run_directory(output, stage=stage, **job)
            manifest = directory/'manifest.json'
            if manifest.exists() and runner.read(manifest).get('status') == 'COMPLETED':
                runner.verify_training(runner.read(manifest), directory, protocol,
                                       runner.read(output/'source_manifest.json'), stage=stage, **job)
                continue
            save(stage=stage, arm=job['arm'], training_seed=job['seed'])
            wait_resources(required_train_mib if stage != 'smoke' else 1200)
            save(status='RUNNING')
            runner.child(output, stage, **job, device='cuda:0', resume=True)
    runner.seal_formal(output)
    for job in study.training_jobs(protocol, 'formal'):
        save(stage='evaluate', arm=job['arm'], training_seed=job['seed'])
        wait_resources(1200)
        save(status='RUNNING')
        runner.child(output, 'evaluate', arm=job['arm'], seed=job['seed'], device='cuda:0',
                     split='holdout', resume=True)
    for stage in ('latency', 'analyze'):
        save(status='RUNNING', stage=stage)
        runner.child(output, stage, device='cpu')
    save(status='COMPLETED', ended_at=datetime.now(timezone.utc).isoformat())
except BaseException as error:
    save(status='FAILED_OR_INTERRUPTED', error=repr(error))
    raise
