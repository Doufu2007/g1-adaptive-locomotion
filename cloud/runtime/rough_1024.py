"""Independent rough-terrain cohort at the approved 1024 x 10001 budget.

Reuses the tested execution engine without modifying the frozen flat cohort.
Run --mode smoke once, then --mode queue. All four rough engineering runs must
finish before formal training. Existing services remain running.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
import formal_1024 as execution


def amended_protocol():
    protocol = copy.deepcopy(execution.ORIGINAL_PROTOCOL)
    protocol['training']['num_envs'] = 1024
    protocol.update(task='Unitree-G1-Rough', terrain_mode='blind_rough',
                    terrain_distribution='phase_random_rough')
    protocol['resource_amendment'] = {
        'user_authorization': 'APPROVED', 'date': '2026-11-03',
        'choice': 'Keep existing services; 1024 environments x 10001 updates',
        'original_num_envs': 4096, 'formal_transitions_per_run': 1024*24*10001,
        'total_formal_transitions': 12*1024*24*10001,
        'flat_execution_engine_sha256': execution.study.sha(HERE.with_name('formal_1024.py')),
    }
    protocol['terrain_amendment'] = {
        'date': '2026-11-03', 'user_authorization': 'APPROVED',
        'user_request': 'Switch the main experiment from flat to rough terrain',
        'adapter_sha256': execution.study.sha(HERE),
        'prior_source_sha256': '473b1ac73da939c4f069143471d3a06e26d1f9b1fccbba5670d97b6c485613f0',
        'reuse_flat_weights': False,
        'learned_phase_factor_in_this_cohort': False,
        'actor_observation': '98 features; body state 93 + command/analytic phase 5',
        'height_scan': 'Removed from both actor and critic; terrain ray sensor removed',
        'terrain': 'Same random_rough distribution as the existing E phase study: 1x5 patches, size 32x16m, height noise .02-.10m; terrain curriculum disabled',
        'numerics': {'ccd_iterations': 50, 'original_ccd_iterations': 500, 'nconmax': 128, 'njmax': 1500,
                     'reason': '500-iteration EPA scratch alone exceeds 8 GiB at 1024 worlds; existing services stay running',
                     'capacity_guard': 'Abort if contacts or constraints saturate; checked every control step'},
        'terrain_seeding': 'Explicit generator seed equals construction seed; evaluation geometry differs from training',
        'reward': 'Native fixed-clock gait and all other rewards common across all actors',
        'engineering_gate': 'All four actors: 1024 rough environments x 2 PPO updates, checkpoint and ONNX parity',
        'flat_pilots': 'Preserved as engineering evidence; do not satisfy rough smoke gate',
        'phase_study_relation': 'Same terrain distribution, separate cohort: phase study differs in rewards, initialization, commands and numerical iteration cap',
    }
    return protocol


def verify_rough_smokes(base):
    root = Path(base)/'rough_smoke_1024'
    gate = execution.runner.read(root/'smoke_gate.json')
    study, runner = execution.study, execution.runner
    protocol = amended_protocol()
    source = runner.source_snapshot()
    if (gate.get('status') != 'COMPLETED' or gate.get('source_sha256') != source['sha256']
            or gate.get('protocol_sha256') != study.protocol_digest(protocol)):
        raise ValueError('rough engineering gate identity differs')
    verified = []
    for arm in study.ARMS:
        seed = protocol['training']['pilot_seed']
        folder = runner.run_directory(root, arm, seed, 'smoke')
        manifest = runner.read(folder/'manifest.json')
        runner.verify_training(manifest, folder, protocol, source, arm, seed, 'smoke')
        if manifest['identity']['iterations'] != 2 or manifest['identity']['num_envs'] != 1024:
            raise ValueError('rough engineering gate budget differs')
        if runner.read(root/'smoke_evaluation'/arm/'manifest.json')['status'] != 'COMPLETED':
            raise ValueError('rough engineering evaluation did not complete')
        verified.append({'arm': arm, 'checkpoint_sha256': manifest['checkpoint_sha256'],
                         'onnx_parity': manifest['onnx_parity']})
    return {'verified_at': study.now(), 'scope': 'rough engineering stability only', 'runs': verified}


def install():
    execution.HERE = HERE
    execution.amended_protocol = amended_protocol
    execution.verify_pilots = verify_rough_smokes
    execution.worker_command = worker_command
    execution.install_protocol()


def worker_command(base, mode, arm=None, seed=None):
    command = [sys.executable, '-u', str(HERE), '--base', str(Path(base).resolve()), '--mode', mode]
    if arm is not None:
        command += ['--arm', arm]
    if seed is not None:
        command += ['--seed', str(seed)]
    return command


def gpu_memory():
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.free',
                             '--format=csv,noheader,nounits'], check=True,
                            capture_output=True, text=True)
    used, free = map(int, result.stdout.splitlines()[0].split(','))
    return {'used_mib': used, 'free_mib': free}


def smoke_all(base):
    output = Path(base)/'rough_smoke_1024'
    output.mkdir(parents=True, exist_ok=True)
    lock, identity = execution.acquire_lock(output, resume=True)
    try:
        _smoke_all(base)
    finally:
        if lock.exists() and execution.runner.read(lock) == identity:
            lock.unlink()


def _smoke_all(base):
    study, runner = execution.study, execution.runner
    base = Path(base)
    output = base/'rough_smoke_1024'
    protocol = runner.prepare(output)
    allowed = {j['pid']: j['created_at'] for j in runner.gpu_processes()
               if 'vllm' in j['name'].lower() and 'created_at' in j}
    before = gpu_memory()
    if execution.capacity_decision(runner.gpu_processes(), allowed, before['free_mib'], 1600) != 'READY':
        raise RuntimeError('rough smoke requires 1600 MiB free and only known vLLM services')
    peak = before['used_mib']
    for arm in study.ARMS:
        while True:
            memory = gpu_memory()
            if execution.capacity_decision(runner.gpu_processes(), allowed, memory['free_mib'], 1600) == 'READY':
                break
            study.write(output/'smoke_status.json', {'status': 'WAITING_FOR_GPU_CAPACITY',
                'arm': arm, 'updated_at': study.now(), 'free_gpu_mib': memory['free_mib']})
            time.sleep(30)
        log = output/f'{arm}_{time.time_ns()}.log'
        command = worker_command(base, 'smoke-arm', arm, protocol['training']['pilot_seed'])
        with log.open('w') as stream:
            child = subprocess.Popen(command, cwd=execution.PROJECT, stdin=subprocess.DEVNULL,
                                     stdout=stream, stderr=subprocess.STDOUT)
            while child.poll() is None:
                peak = max(peak, gpu_memory()['used_mib'])
                study.write(output/'smoke_status.json', {'status': 'RUNNING', 'arm': arm,
                    'child_pid': child.pid, 'updated_at': study.now(), 'peak_total_gpu_used_mib': peak})
                time.sleep(2)
        if child.returncode:
            study.write(output/'smoke_status.json', {'status': 'FAILED', 'arm': arm,
                'exit_code': child.returncode, 'log': str(log)})
            raise RuntimeError('rough smoke failed: '+str(log))
    gate = {'status': 'COMPLETED', 'at': study.now(),
            'source_sha256': runner.source_snapshot()['sha256'],
            'protocol_sha256': study.protocol_digest(protocol)}
    study.write(output/'smoke_gate.json', gate)
    verified = verify_rough_smokes(base)
    study.write(output/'verified_smokes.json', verified)
    study.write(base/'concurrency_probe_1024/resource.json', {
        'status': 'PASSED', 'before_gpu': before, 'peak_total_gpu_used_mib': peak,
        'after_gpu': gpu_memory(), 'scope': '4 actors x 2 updates; native rough; engineering only'})
    study.write(output/'smoke_status.json', gate)


def main():
    os.environ['MUJOCO_GL'] = 'disable'
    install()
    if '--mode' in sys.argv and sys.argv[sys.argv.index('--mode')+1] in ('smoke', 'smoke-arm'):
        parser = argparse.ArgumentParser()
        parser.add_argument('--base', type=Path, default=HERE.parent)
        parser.add_argument('--mode', choices=['smoke', 'smoke-arm'], required=True)
        parser.add_argument('--arm', choices=execution.study.ARMS)
        parser.add_argument('--seed', type=int)
        args = parser.parse_args()
        if args.mode == 'smoke':
            smoke_all(args.base)
        else:
            execution.runner.train(args.base/'rough_smoke_1024', args.arm, args.seed, 'smoke',
                                   device='cuda:0', smoke_envs=1024, resume=True)
            from scripts.evaluate_g1_semantic_split import smoke_evaluate
            target = args.base/'rough_smoke_1024/smoke_evaluation'/args.arm
            if not (target/'manifest.json').exists():
                if target.exists():
                    target.rename(target.with_name(target.name+f'_partial_{time.time_ns()}'))
                smoke_evaluate(args.base/'rough_smoke_1024', args.arm, args.seed, device='cuda:0')
    else:
        execution.main()


if __name__ == '__main__':
    main()
