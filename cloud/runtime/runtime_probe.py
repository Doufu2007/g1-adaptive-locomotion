"""Measure the unmodified policy/runtime beside existing GPU services."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

base = Path(__file__).resolve().parent
project = base/'g1_semantic_split'
python = str(base/'venv/bin/python')
env = dict(os.environ, PYTHONHASHSEED='20262100', PYTHONIOENCODING='utf-8',
           CUBLAS_WORKSPACE_CONFIG=':4096:8', MUJOCO_GL='disable')


def gpu():
    r = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.free,utilization.gpu',
                        '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
    values = [int(v.strip()) for v in r.stdout.splitlines()[0].split(',')]
    return dict(zip(('used_mib', 'free_mib', 'utilization_percent'), values))


def write(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2)+'\n')
    temporary.replace(path)


records = []
for n in [4, 1024, 4096]:
    directory = base/f'concurrency_probe_{n}'
    directory.mkdir(exist_ok=False)
    command = [python, '-u', '-m', 'scripts.run_g1_semantic_split', '--stage', 'smoke',
               '--arm', 'semantic', '--seed', '20262100', '--device', 'cuda:0',
               '--smoke-envs', str(n), '--output', str(directory/'output')]
    record = {'num_envs': n, 'started_at': datetime.now(timezone.utc).isoformat(),
              'command': command, 'before_gpu': gpu(), 'status': 'RUNNING',
              'scope': '2 PPO updates, engineering resource probe, not a scientific outcome'}
    write(directory/'resource.json', record)
    started = time.monotonic()
    with (directory/'stdout.log').open('w') as log:
        child = subprocess.Popen(command, cwd=project, env=env, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT)
        record['pid'] = child.pid
        samples = []
        while child.poll() is None:
            try:
                samples.append(dict(elapsed_seconds=time.monotonic()-started, **gpu()))
                record.update(peak_total_gpu_used_mib=max(s['used_mib'] for s in samples),
                              elapsed_seconds=time.monotonic()-started, samples=len(samples))
                write(directory/'resource.json', record)
            except Exception as error:
                record['monitor_error'] = repr(error)
            time.sleep(2)
    record.update(exit_code=child.returncode, status='PASSED' if child.returncode == 0 else 'FAILED',
                  ended_at=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic()-started,
                  after_gpu=gpu())
    write(directory/'resource.json', record)
    write(directory/'gpu_samples.json', samples)
    records.append(record)
    write(base/'concurrency_probes.json', {'status': 'RUNNING', 'records': records})
    if child.returncode != 0:
        write(base/'concurrency_probes.json', {'status': 'FAILED', 'records': records,
              'failed_num_envs': n, 'original_4096_env_configuration_fits': False,
              'existing_services_stopped': False})
        raise SystemExit(child.returncode or 1)
write(base/'concurrency_probes.json', {'status': 'COMPLETED', 'records': records,
      'original_4096_env_configuration_fits': any(r['num_envs'] == 4096 and r['status'] == 'PASSED' for r in records),
      'existing_services_stopped': False})
