"""Four engineering pilots on the measured 1024-env coexistence configuration.

This explicit adapter changes only the engineering pilot environment count.
It cannot start formal training or produce a scientific holdout conclusion.
"""
from datetime import datetime, timezone
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

BASE = Path(__file__).resolve().parent
PROJECT = BASE/'g1_semantic_split'
sys.path.insert(0, str(PROJECT))
from scripts import g1_semantic_split_study as study
from scripts import run_g1_semantic_split as runner

OUTPUT = BASE/'engineering_pilot_1024'
original_protocol = study.default_protocol
adapter_sha = study.sha(__file__)


def pilot_protocol():
    protocol = copy.deepcopy(original_protocol())
    protocol['training']['num_envs'] = 1024
    protocol['execution_scope'] = 'ENGINEERING_PILOT_ONLY; formal and holdout execution prohibited by this adapter'
    protocol['engineering_amendment'] = {
        'date': '2026-11-03', 'adapter_sha256': adapter_sha,
        'original_protocol_sha256': study.protocol_digest(original_protocol()),
        'reason': '4096 environments exhausted GPU memory alongside two existing vLLM services; 1024 passed',
        'pilot_updates_per_arm': 1000, 'pilot_transitions_per_arm': 1024*24*1000,
        'interpretation': 'pipeline stability and speed only; not method ranking or original pilot completion',
        'formal_budget': 'pending user choice; no formal job authorized through this engineering adapter'}
    return protocol


study.default_protocol = pilot_protocol


def queue():
    OUTPUT.mkdir(exist_ok=True)
    lock = OUTPUT/'pilot_queue.lock'
    identity = {'pid': os.getpid(), 'created_at': psutil.Process().create_time()}
    if lock.exists():
        previous = runner.read(lock)
        try:
            alive = abs(psutil.Process(previous['pid']).create_time()-previous['created_at']) < .01
        except psutil.NoSuchProcess:
            alive = False
        if alive:
            raise RuntimeError('engineering pilot queue already active')
        lock.rename(OUTPUT/f'pilot_queue_stale_{time.time_ns()}.json')
    with lock.open('x') as stream:
        json.dump(identity, stream)
    state = dict(identity, status='STARTING', started_at=study.now(),
                 adapter_sha256=adapter_sha, formal_training_started=False,
                 existing_services_stopped=False)
    allowed = {}
    for pid in (17599, 18239):
        try:
            allowed[pid] = psutil.Process(pid).create_time()
        except psutil.NoSuchProcess:
            pass
    probe = runner.read(BASE/'concurrency_probe_1024/resource.json')
    if probe['status'] != 'PASSED':
        raise RuntimeError('1024 environment resource probe has not passed')
    required = probe['peak_total_gpu_used_mib']-probe['before_gpu']['used_mib']+512

    def save(**updates):
        state.update(updates, updated_at=study.now())
        study.write(OUTPUT/'pilot_queue_status.json', state)

    try:
        protocol = runner.prepare(OUTPUT)
        study.write(OUTPUT/'adapter.py.sha256.json', {'file': str(Path(__file__)), 'sha256': adapter_sha})
        study.write(OUTPUT/'allowed_gpu_jobs.json', allowed)
        for arm in study.ARMS:
            save(arm=arm, stage='pilot', required_free_gpu_mib=required)
            while True:
                jobs = runner.gpu_processes()
                unknown = [j for j in jobs if j['pid'] not in allowed
                           or abs(j.get('created_at', -1)-allowed[j['pid']]) > .01]
                free = int(subprocess.run(['nvidia-smi', '--query-gpu=memory.free',
                              '--format=csv,noheader,nounits'], capture_output=True,
                              text=True, check=True).stdout.splitlines()[0])
                if not unknown and free >= required:
                    break
                save(status='WAITING_FOR_GPU_CAPACITY', free_gpu_mib=free, unknown_gpu_jobs=unknown)
                time.sleep(30)
            save(status='RUNNING', free_gpu_mib=free)
            log_path = OUTPUT/f'pilot_{arm}_{time.time_ns()}.log'
            command = [sys.executable, '-u', str(Path(__file__)), '--mode', 'train', '--arm', arm]
            with log_path.open('w') as log:
                child = subprocess.Popen(command, cwd=PROJECT, stdin=subprocess.DEVNULL,
                                         stdout=log, stderr=subprocess.STDOUT)
                save(child_pid=child.pid, log=str(log_path))
                code = child.wait()
            if code:
                raise RuntimeError(f'{arm} pilot failed (exit {code}); see {log_path}')
            directory = runner.run_directory(OUTPUT, arm, protocol['training']['pilot_seed'], 'pilot')
            record = runner.read(directory/'manifest.json')
            runner.verify_training(record, directory, protocol, runner.read(OUTPUT/'source_manifest.json'),
                                   arm, protocol['training']['pilot_seed'], 'pilot')
            state.setdefault('completed_arms', []).append(arm)
            save()
        save(status='COMPLETED', ended_at=study.now(),
             interpretation='four engineering pilots; no formal performance conclusion')
    except BaseException as error:
        save(status='FAILED_OR_INTERRUPTED', error=repr(error))
        raise
    finally:
        if lock.exists() and runner.read(lock) == identity:
            lock.unlink()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['queue', 'train'], required=True)
    parser.add_argument('--arm', choices=study.ARMS)
    args = parser.parse_args()
    os.environ['MUJOCO_GL'] = 'disable'
    if args.mode == 'queue':
        queue()
    else:
        if args.arm is None:
            parser.error('--arm is required for train')
        result = runner.train(OUTPUT, args.arm, 20262100, 'pilot', device='cuda:0', resume=True)
        print(result['status'], args.arm, flush=True)
