"""Execute the user-approved 1024 x 10001 G1 study after the engineering pilots.

All workers enter through this file so training, sealing and evaluation use the
same explicit amendment. Existing GPU services are never stopped or modified.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time
import traceback

import psutil

HERE = Path(__file__).resolve()
PROJECT = HERE.parent/'g1_semantic_split'
if not PROJECT.is_dir():
    PROJECT = HERE.parents[2]  # Local development copy in tmp/remote_20261103.
sys.path.insert(0, str(PROJECT))
from scripts import g1_semantic_split_study as study
from scripts import run_g1_semantic_split as runner

ORIGINAL_PROTOCOL = copy.deepcopy(study.default_protocol())


def amended_protocol():
    protocol = copy.deepcopy(ORIGINAL_PROTOCOL)
    protocol['training']['num_envs'] = 1024
    protocol['resource_amendment'] = {
        'recorded_date': '2026-11-03',
        'user_authorization': 'APPROVED',
        'user_choice': 'Keep both existing services; 1024 environments x 10001 updates for the first round',
        'reason': '4096 environments failed GPU memory admission alongside the two vLLM services; 1024 passed',
        'original_protocol_sha256': study.protocol_digest(ORIGINAL_PROTOCOL),
        'adapter_sha256': study.sha(HERE),
        'original_num_envs': 4096, 'amended_num_envs': 1024,
        'formal_transitions_per_run': 1024*24*10001,
        'total_formal_transitions': 12*1024*24*10001,
        'fraction_of_original_transitions': 0.25,
        'pilot_weights_used_for_formal': False,
        'interpretation': 'First-round evidence at the amended budget; not completion of the 4096-environment protocol',
        'endpoint_update': 'User supplied connect.weste.seetacloud.com:16963; source and outputs retained',
    }
    return protocol


def install_protocol():
    study.default_protocol = amended_protocol


def worker_command(base, mode, arm=None, seed=None):
    command = [sys.executable, '-u', str(HERE), '--base', str(Path(base).resolve()), '--mode', mode]
    if arm is not None:
        command += ['--arm', arm]
    if seed is not None:
        command += ['--seed', str(seed)]
    return command


def capacity_decision(jobs, allowed, free_mib, required_mib):
    unexpected = [j for j in jobs if j['pid'] not in allowed
                  or abs(j.get('created_at', -1)-allowed[j['pid']]) > .01]
    return 'READY' if not unexpected and free_mib >= required_mib else 'WAITING_FOR_GPU_CAPACITY'


def verify_pilots(base):
    output = Path(base)/'engineering_pilot_1024'
    state = runner.read(output/'pilot_queue_status.json')
    if state.get('status') != 'COMPLETED':
        return None
    protocol = runner.read(output/'protocol.json')
    source = runner.read(output/'source_manifest.json')
    if protocol['training']['num_envs'] != 1024 or protocol['training']['pilot_iterations'] != 1000:
        raise ValueError('engineering pilot budget differs from the approved dependency')
    if source['sha256'] != runner.source_snapshot()['sha256']:
        raise ValueError('engineering pilot and formal sources differ')
    if protocol['engineering_amendment']['adapter_sha256'] != study.sha(Path(base)/'pilot_1024.py'):
        raise ValueError('engineering pilot adapter changed')
    verified = []
    for job in study.training_jobs(protocol, 'pilot'):
        directory = runner.run_directory(output, **job)
        manifest = runner.read(directory/'manifest.json')
        runner.verify_training(manifest, directory, protocol, source, **job)
        verified.append({'arm': job['arm'], 'checkpoint_sha256': manifest['checkpoint_sha256'],
                         'onnx_parity': manifest['onnx_parity'],
                         'resumed': manifest.get('resumed', False)})
    return {'verified_at': study.now(), 'scope': 'engineering stability only', 'runs': verified}


def acquire_lock(output, resume):
    lock = output/'formal_queue.lock'
    if lock.exists():
        old = runner.read(lock)
        try:
            alive = abs(psutil.Process(old['pid']).create_time()-old['created_at']) < .01
        except psutil.NoSuchProcess:
            alive = False
        if alive or not resume:
            raise RuntimeError('formal queue lock exists; live queue or explicit --resume-queue required')
        lock.rename(output/f'formal_queue_stale_{time.time_ns()}.json')
    identity = {'pid': os.getpid(), 'created_at': psutil.Process().create_time()}
    with lock.open('x') as stream:
        json.dump(identity, stream)
    return lock, identity


def package_results(base, output):
    archive = Path(base)/'g1_semantic_split_formal_1024_results.tar.gz'
    final_name = f"model_{amended_protocol()['training']['formal_iterations']-1}.pt"
    inventory_path = output/'delivery_content_manifest.json'
    identity = {'protocol_sha256': study.protocol_digest(runner.read(output/'protocol.json')),
                'source_sha256': runner.read(output/'source_manifest.json')['sha256'],
                'adapter_sha256': study.sha(HERE)}
    if inventory_path.exists():
        inventory = runner.read(inventory_path)
        if inventory['identity'] != identity:
            raise ValueError('delivery identity changed')
    else:
        files = {}
        for path in sorted(output.rglob('*')):
            if not path.is_file() or path.suffix == '.lock' or path.name == 'queue_status.json':
                continue
            if path.name.startswith('model_') and path.suffix == '.pt' and path.name != final_name:
                continue
            files[path.relative_to(output).as_posix()] = study.sha(path)
        inventory = {'identity': identity, 'files': files}
        study.write(inventory_path, inventory)
    for name, expected in inventory['files'].items():
        if study.sha(output/name) != expected:
            raise ValueError('delivery file changed: '+name)
    expected_members = {output.name+'/'+name: digest for name, digest in inventory['files'].items()}
    expected_members[output.name+'/'+inventory_path.name] = study.sha(inventory_path)
    if not archive.exists():
        temporary = archive.with_suffix('.gz.tmp')
        if temporary.exists():
            temporary.rename(temporary.with_name(temporary.name+f'.interrupted_{time.time_ns()}'))
        with tarfile.open(temporary, 'w:gz') as bundle:
            for name in inventory['files']:
                bundle.add(output/name, arcname=output.name+'/'+name, recursive=False)
            bundle.add(inventory_path, arcname=output.name+'/'+inventory_path.name, recursive=False)
        temporary.replace(archive)
    # Covers interruption after the atomic archive rename but before its receipt.
    with tarfile.open(archive, 'r:gz') as bundle:
        members = bundle.getmembers()
        if len(members) != len(expected_members) or {m.name for m in members} != set(expected_members):
            raise ValueError('delivery archive membership changed')
        for member in members:
            if not member.isfile():
                raise ValueError('delivery archive member type changed')
            stream = bundle.extractfile(member)
            digest = hashlib.sha256()
            for block in iter(lambda: stream.read(1024*1024), b''):
                digest.update(block)
            if digest.hexdigest() != expected_members[member.name]:
                raise ValueError('delivery archive content changed: '+member.name)
    return {'archive': str(archive), 'sha256': study.sha(archive), 'bytes': archive.stat().st_size,
            'contents': 'all formal metadata, logs, metrics, evaluation trajectories, ONNX and final checkpoints',
            'intermediate_checkpoints': 'retained in remote run directories; excluded from portable delivery'}


def queue(base, resume=False):
    base = Path(base)
    output = base/'formal_1024'
    protocol = runner.prepare(output)
    lock, identity = acquire_lock(output, resume)
    state = dict(identity, status='STARTING', started_at=study.now(),
                 user_authorization='APPROVED', adapter_sha256=study.sha(HERE),
                 protocol_sha256=study.protocol_digest(protocol),
                 existing_services_stopped=False, existing_services_modified=False,
                 formal_training_started=False, completed_formal_runs=[])

    def save(**updates):
        state.update(updates, updated_at=study.now())
        study.write(output/'queue_status.json', state)

    def wait_resources(required, allowed):
        while True:
            jobs = runner.gpu_processes()
            free = int(subprocess.run(['nvidia-smi', '--query-gpu=memory.free',
                                      '--format=csv,noheader,nounits'], check=True,
                                     capture_output=True, text=True).stdout.splitlines()[0])
            decision = capacity_decision(jobs, allowed, free, required)
            save(status=decision, free_gpu_mib=free, required_free_gpu_mib=required, observed_gpu_jobs=jobs)
            if decision == 'READY':
                return
            time.sleep(30)

    def run_worker(mode, arm=None, seed=None):
        command = worker_command(base, mode, arm, seed)
        log_path = output/'logs'/f'{mode}_{arm}_{seed}_{time.time_ns()}.log'
        log_path.parent.mkdir(exist_ok=True)
        env = dict(os.environ, MUJOCO_GL='disable', PYTHONHASHSEED=str(seed or 20261026),
                   PYTHONIOENCODING='utf-8', CUBLAS_WORKSPACE_CONFIG=':4096:8')
        with log_path.open('w') as log:
            child = subprocess.Popen(command, cwd=PROJECT, env=env, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT)
            save(status='RUNNING', stage=mode, arm=arm, training_seed=seed, child_pid=child.pid,
                 child_created_at=psutil.Process(child.pid).create_time(), log=str(log_path))
            code = child.wait()
        save(child_pid=None, last_child_exit_code=code)
        if code:
            raise RuntimeError(f'{mode} {arm} {seed} failed ({code}); see {log_path}')

    try:
        save()
        support = output/'execution_support'
        support.mkdir(exist_ok=True)
        for name in ('formal_1024.py', 'pilot_1024.py', 'test_formal_1024.py',
                     'tests_after_endpoint_change.log', 'execution_tests_after_endpoint_change.log',
                     'formal_adapter_tests.log'):
            original = base/name
            if original.is_file():
                target = support/name
                if target.exists() and study.sha(target) != study.sha(original):
                    raise ValueError('execution support changed: '+name)
                if not target.exists():
                    target.write_bytes(original.read_bytes())
        while True:
            evidence = verify_pilots(base)
            if evidence is not None:
                study.write(output/'engineering_dependency_verified.json', evidence)
                break
            dependency = runner.read(base/'engineering_pilot_1024/pilot_queue_status.json')
            if dependency.get('status') == 'FAILED_OR_INTERRUPTED':
                raise RuntimeError('engineering pilot failed; formal jobs are blocked')
            try:
                live = abs(psutil.Process(dependency['pid']).create_time()-dependency['created_at']) < .01
            except psutil.NoSuchProcess:
                live = False
            if not live:
                raise RuntimeError('engineering pilot is interrupted; resume it before the formal queue')
            save(status='WAITING_FOR_ENGINEERING_PILOTS', dependency=dependency)
            time.sleep(30)

        source = runner.read(output/'source_manifest.json')
        probe = runner.read(base/'concurrency_probe_1024/resource.json')
        if probe['status'] != 'PASSED':
            raise RuntimeError('1024-environment coexistence probe did not pass')
        required = max(512, probe['peak_total_gpu_used_mib']-probe['before_gpu']['used_mib'])+512
        allowed = {j['pid']: j['created_at'] for j in runner.gpu_processes()
                   if 'vllm' in j['name'].lower() and 'created_at' in j}
        study.write(output/'allowed_existing_gpu_jobs.json', allowed)
        save(allowed_existing_gpu_jobs=allowed, required_train_mib=required,
             service_snapshot_note='Only existing vLLM processes are admitted; no services are started or stopped')
        for job in study.training_jobs(protocol, 'formal'):
            directory = runner.run_directory(output, **job)
            manifest = directory/'manifest.json'
            if manifest.exists() and runner.read(manifest).get('status') == 'COMPLETED':
                runner.verify_training(runner.read(manifest), directory, protocol, source, **job)
            else:
                save(stage='formal', arm=job['arm'], training_seed=job['seed'])
                wait_resources(required, allowed)
                save(formal_training_started=True)
                run_worker('train', job['arm'], job['seed'])
                runner.verify_training(runner.read(manifest), directory, protocol, source, **job)
            state['completed_formal_runs'].append({'arm': job['arm'], 'seed': job['seed']})
            save()

        seal = runner.seal_formal(output)
        save(stage='policies_sealed', sealed_policies=len(seal['policies']))
        for job in study.training_jobs(protocol, 'formal'):
            save(stage='evaluate', arm=job['arm'], training_seed=job['seed'])
            wait_resources(max(1400, required), allowed)
            run_worker('evaluate', job['arm'], job['seed'])
        completion_identity = {'status': 'COMPLETED', 'protocol_sha256': study.protocol_digest(protocol),
                               'adapter_sha256': study.sha(HERE), 'source_sha256': source['sha256']}
        if (output/'completion.json').exists():
            completed = runner.read(output/'completion.json')
            if any(completed.get(key) != value for key, value in completion_identity.items()):
                raise ValueError('completion identity changed')
        else:
            for mode in ('latency', 'analyze'):
                run_worker(mode)
            study.write(output/'completion.json', {
                'status': 'COMPLETED', 'at': study.now(), 'formal_runs_verified': 12,
                'formal_transitions_per_run': 1024*24*10001, 'holdout_evaluations': 12,
                'protocol_sha256': study.protocol_digest(protocol), 'adapter_sha256': study.sha(HERE),
                'source_sha256': source['sha256'], 'existing_services_stopped': False})
        save(status='PACKAGING', stage='delivery', arm=None, training_seed=None)
        delivery = package_results(base, output)
        study.write(base/'formal_1024_delivery.json', delivery)
        save(status='COMPLETED', ended_at=study.now(), delivery=delivery)
    except BaseException as error:
        save(status='FAILED_OR_INTERRUPTED', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        if lock.exists() and runner.read(lock) == identity:
            lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=HERE.parent)
    parser.add_argument('--mode', choices=['prepare', 'queue', 'train', 'evaluate', 'latency', 'analyze'], required=True)
    parser.add_argument('--arm', choices=study.ARMS)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--resume-queue', action='store_true')
    args = parser.parse_args()
    os.environ['MUJOCO_GL'] = 'disable'
    install_protocol()
    output = args.base/'formal_1024'
    if args.mode == 'queue':
        queue(args.base, args.resume_queue)
    elif args.mode == 'prepare':
        runner.prepare(output)
    elif args.mode in ('train', 'evaluate'):
        if args.arm is None or args.seed is None:
            parser.error('--arm and --seed are required')
        if {'arm': args.arm, 'seed': args.seed, 'stage': 'formal'} not in study.training_jobs(amended_protocol(), 'formal'):
            parser.error('outside the frozen formal matrix')
        if args.mode == 'train':
            record = runner.train(output, args.arm, args.seed, 'formal', device='cuda:0', resume=True)
        else:
            from scripts.evaluate_g1_semantic_split import evaluate
            record = evaluate(output, args.arm, args.seed, 'holdout', device='cuda:0', resume=True)
        print(record['status'], args.mode, args.arm, args.seed, flush=True)
    elif args.mode == 'latency':
        from scripts.evaluate_g1_semantic_split import benchmark_all
        benchmark_all(output)
    else:
        from scripts.analyze_g1_semantic_split import analyze
        analyze(output)


if __name__ == '__main__':
    main()
