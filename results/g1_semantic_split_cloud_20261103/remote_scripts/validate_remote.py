"""Finish installation checks, run the test suite, then measure GPU coexistence."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time

import psutil

base = Path(__file__).resolve().parent
project = base/'g1_semantic_split'
python = str(base/'venv/bin/python')
state = {'status': 'WAITING_FOR_INSTALLATION', 'started_at': datetime.now(timezone.utc).isoformat()}


def save():
    temporary = base/'validation_status.tmp'
    temporary.write_text(json.dumps(state, indent=2)+'\n')
    temporary.replace(base/'validation_status.json')


save()
installation_pid = json.loads((base/'deployment.json').read_text())['pid']
while True:
    try:
        alive = psutil.Process(installation_pid).status() != 'zombie'
    except psutil.NoSuchProcess:
        alive = False
    if not alive:
        break
    time.sleep(5)
env = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONHASHSEED='20262100',
           MUJOCO_GL='disable', CUBLAS_WORKSPACE_CONFIG=':4096:8')
for name, command in [
    ('dependencies', [python, '-c', 'import importlib.metadata as m,json; print(json.dumps({n:m.version(n) for n in ["torch","mjlab","rsl-rl-lib","mujoco","mujoco-warp","warp-lang","numpy","onnx","onnxruntime","scipy","pytest"]}))']),
    ('tests', [python, '-m', 'pytest', 'tests/test_g1_semantic_split.py', 'tests/test_g1_semantic_split_execution.py', '-q']),
    ('concurrency', [python, '-u', str(base/'runtime_probe.py')])]:
    state.update(status='RUNNING_'+name.upper(), stage=name, updated_at=datetime.now(timezone.utc).isoformat())
    save()
    with (base/(name+'.log')).open('w') as stream:
        result = subprocess.run(command, cwd=project, env=env, stdin=subprocess.DEVNULL,
                                stdout=stream, stderr=subprocess.STDOUT)
    state.setdefault('exit_codes', {})[name] = result.returncode
    if result.returncode:
        state.update(status='FAILED', failed_stage=name)
        save()
        raise SystemExit(result.returncode)
state.update(status='VALIDATED_AND_MEASURED', ended_at=datetime.now(timezone.utc).isoformat())
save()
