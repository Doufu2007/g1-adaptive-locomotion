"""Install the authorized G1 workload in its own environment."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

base = Path('/root/autodl-tmp/g1_semantic_split_20261103')
archive = base/'source_bundle.zip'
expected = '852747647b32b68171085d71c49446efb0fec70dadba64a5ed17a023b59e7f9a'
if hashlib.sha256(archive.read_bytes()).hexdigest() != expected:
    raise ValueError('source archive hash mismatch')
with zipfile.ZipFile(archive) as source:
    if source.testzip() is not None:
        raise ValueError('source archive CRC mismatch')
    for name in source.namelist():
        if not (base/name).resolve().is_relative_to(base):
            raise ValueError('archive path outside workspace')
    source.extractall(base)
project = base/'g1_semantic_split'
environment = base/'venv'
subprocess.run([sys.executable, '-m', 'venv', '--system-site-packages', str(environment)], check=True)
env = dict(os.environ, PIP_CACHE_DIR=str(base/'pip-cache'), PYTHONIOENCODING='utf-8')
log = (base/'installation.log').open('w', encoding='utf-8')
worker = subprocess.Popen([str(environment/'bin/python'), '-m', 'pip', 'install',
                           '-r', str(project/'requirements-semantic-split.txt')],
                          cwd=project, env=env, stdin=subprocess.DEVNULL,
                          stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
record = {'status': 'INSTALLING', 'pid': worker.pid, 'project': str(project),
          'python': str(environment/'bin/python'), 'archive_sha256': expected,
          'existing_vllm_services_stopped': False, 'global_python_modified': False}
(base/'deployment.json').write_text(json.dumps(record, indent=2)+'\n')
print(json.dumps(record))
