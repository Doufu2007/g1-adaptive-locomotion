"""Prepare, train, seal and execute the four-arm G1 split study.

Run from the repository root with ``python -m scripts.run_g1_semantic_split``.
No cloud account, credentials, paid service or learned phase encoder is needed.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
from dataclasses import asdict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import g1_semantic_split_study as study

MODULE = 'scripts.run_g1_semantic_split'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def source_snapshot():
    paths = set()
    for pattern in ('setup.py', 'requirements-semantic-split.txt', 'src/**/*.py',
                    'src/assets/robots/unitree_g1/**/*', 'scripts/*g1_semantic_split*.py',
                    'tests/test_g1_semantic_split*.py', 'docs/superpowers/plans/2026-10-26-g1-semantic-split.md',
                    'docs/superpowers/specs/2026-10-26-g1-semantic-split-protocol.md',
                    'docs/superpowers/specs/2026-11-03-g1-semantic-split-rough-amendment.md'):
        paths.update(p for p in ROOT.glob(pattern) if p.is_file() and '__pycache__' not in p.parts)
    files = {p.relative_to(ROOT).as_posix(): study.sha(p) for p in sorted(paths)}
    return {'files': files, 'sha256': study.protocol_digest(files)}


def prepare(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    protocol = study.default_protocol()
    path = output/'protocol.json'
    if path.exists() and read(path) != protocol:
        raise ValueError('frozen protocol changed; retain this experiment and create an explicit amendment')
    source = source_snapshot()
    seal = output/'source_manifest.json'
    if seal.exists() and read(seal)['sha256'] != source['sha256']:
        raise ValueError('frozen sources changed; preserve outputs and use a new experiment directory/amendment')
    for target, value in ((path, protocol), (seal, source),
                           (output/'observation_layout.json', study.observation_layout(study.EXPECTED_TERMS)),
                           (output/'training_matrix.json', {s:study.training_jobs(protocol,s) for s in ('pilot','formal')}),
                           (output/'random_partition.json', dict(zip(('large_branch','small_branch'),
                                study.random_partition(98,5,protocol['partition_seed']))))):
        if not target.exists():
            study.write(target, value)
    return protocol


def environment_record():
    packages = {}
    for name in ('torch','mjlab','rsl-rl-lib','mujoco','mujoco-warp','warp-lang','numpy','onnx','onnxruntime','scipy'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = 'UNAVAILABLE'
    return {'at':study.now(),'python':sys.version,'executable':sys.executable,'platform':platform.platform(),
            'packages':packages,'pythonhashseed':os.environ.get('PYTHONHASHSEED'),
            'cublas_workspace_config':os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
            'seed_scope':'Python, NumPy, torch and CUDA; fixed cuDNN flags; no cross-hardware bitwise claim'}


def run_directory(output, arm, seed, stage):
    if arm not in study.ARMS or stage not in ('smoke','pilot','formal'):
        raise ValueError('invalid arm or stage')
    return Path(output)/stage/f'{arm}_seed{seed}'


def check_identity(record, expected):
    if record.get('identity') != expected:
        raise ValueError('run identity differs in arm/seed/stage/budget/protocol/source')


def state_sha(module):
    h = __import__('hashlib').sha256()
    for name, value in sorted(module.state_dict().items()):
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def verify_training(record, directory, protocol, source, arm, seed, stage):
    training = protocol['training']
    target = training[stage+'_iterations'] if stage != 'smoke' else record['identity']['iterations']
    expected = {'arm':arm,'seed':seed,'stage':stage,'iterations':target,
                'num_envs':training['num_envs'] if stage != 'smoke' else record['identity']['num_envs'],
                'protocol_sha256':study.protocol_digest(protocol),'source_sha256':source['sha256']}
    check_identity(record, expected)
    if (record['status'] != 'COMPLETED' or record['completed_updates'] != target
            or record['completed_transitions'] != target * expected['num_envs'] * training['steps_per_env']
            or not record['critic_initialization_controlled'] or not record['onnx_parity']['passed']):
        raise ValueError('run is incomplete or did not satisfy its training/export contract')
    for name in ('policy', 'checkpoint'):
        path = Path(directory)/record[name+'_file']
        if study.sha(path) != record[name+'_sha256']:
            raise ValueError('saved training artifact changed: '+str(path))


def choose_resume(directory, identity):
    import torch
    rejected = []
    candidates = sorted(Path(directory).glob('model_*.pt'),
                        key=lambda p:int(p.stem.rsplit('_',1)[1]), reverse=True)
    for path in candidates:
        try:
            saved = torch.load(path,map_location='cpu',weights_only=False)
            state = saved['infos']['split_study']
            if state['identity'] != identity or state['completed_updates'] != saved['iter'] + 1:
                raise ValueError('checkpoint identity or iteration mismatch')
            if not 0 < state['completed_updates'] <= identity['iterations']:
                raise ValueError('checkpoint budget invalid')
            return path, state, rejected
        except Exception as error:
            rejected.append({'path':str(path),'error':repr(error)})
    raise ValueError('no intact matching recovery checkpoint: '+str(rejected))


def recovery_for_run(directory, identity, *, requested):
    """A queue recovery request also covers jobs that have never started."""
    manifest = Path(directory)/'manifest.json'
    if not manifest.exists():
        return None
    previous = read(manifest)
    check_identity(previous, identity)
    if previous['status'] == 'COMPLETED':
        return None
    if not requested:
        raise FileExistsError('preserve failed/interrupted attempt; explicit --resume required')
    # Validate before overwriting the old manifest, including the no-checkpoint case.
    return choose_resume(directory, identity)


def train(output, arm, seed, stage, *, device='cuda:0', smoke_envs=4, resume=False):
    import torch
    import numpy as np
    protocol = prepare(output)
    if stage in ('pilot','formal') and {'arm':arm,'seed':seed,'stage':stage} not in study.training_jobs(protocol,stage):
        raise ValueError('outside frozen training matrix')
    source = read(Path(output)/'source_manifest.json')
    iterations = 2 if stage == 'smoke' else protocol['training'][stage+'_iterations']
    num_envs = smoke_envs if stage == 'smoke' else protocol['training']['num_envs']
    identity = dict(arm=arm,seed=seed,stage=stage,iterations=iterations,num_envs=num_envs,
                    protocol_sha256=study.protocol_digest(protocol),source_sha256=source['sha256'])
    directory = run_directory(output,arm,seed,stage)
    manifest = directory/'manifest.json'
    recovery = recovery_for_run(directory, identity, requested=resume)
    if manifest.exists():
        previous = read(manifest)
        check_identity(previous,identity)
        if previous['status'] == 'COMPLETED':
            verify_training(previous,directory,protocol,source,arm,seed,stage)
            return previous
        if not resume:
            raise FileExistsError('preserve failed/interrupted attempt; explicit --resume required')
        study.write(directory/'attempts'/f'manifest_{time.time_ns()}.json',previous)
    elif directory.exists() and (directory/'config.json').exists():
        raise FileExistsError('an existing unclassified attempt is preserved')
    directory.mkdir(parents=True,exist_ok=True)
    record = {'schema':'g1.semantic_split.run.v1','status':'RUNNING','started_at':study.now(),
              'identity':identity,'argv':sys.argv,'resumed':recovery is not None,'resume_requested':resume,
              'completed_updates':0,'completed_transitions':0,'fresh_actor_and_critic':recovery is None,
              'resume_limitation':'physics state is reset at recovery; not an uninterrupted trajectory' if recovery else None}
    study.write(manifest,record)
    env = None
    collected, carried = 0, 0
    try:
        study.bootstrap()
        study.seed_all(seed)
        from mjlab.envs import ManagerBasedRlEnv
        from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
        from mjlab.tasks.registry import load_rl_cfg
        cfg = study.make_env_config(protocol,seed,num_envs)
        common_config = study.canonical(asdict(cfg))
        study.write(directory/'env.json', common_config)
        raw = ManagerBasedRlEnv(cfg=cfg,device=device)
        agent = load_rl_cfg(protocol['task'])
        agent.seed,agent.logger,agent.upload_model = seed,'tensorboard',False
        agent.save_interval = protocol['training']['save_interval']
        if agent.num_steps_per_env != protocol['training']['steps_per_env']:
            raise ValueError('rollout length differs from protocol')
        layout = study.runtime_layout(raw)
        study.write(directory/'observation_layout.json',layout)
        env = RslRlVecEnvWrapper(raw,clip_actions=agent.clip_actions)
        config = asdict(agent)
        config['actor'] = study.actor_config(arm,layout,protocol['partition_seed'])
        config['max_iterations'] = iterations
        study.write(directory/'config.json',dict(identity=identity,agent=config))
        record['common_env_sha256'] = study.protocol_digest(common_config)
        shared_agent = copy.deepcopy(config)
        shared_agent.pop('actor');shared_agent.pop('max_iterations')
        record['common_agent_sha256'] = study.protocol_digest(shared_agent)
        # Actor sizes consume different RNG streams. Give the unchanged critic an
        # independent common initialization, then reseed the rollout stream.
        study.seed_all(seed+10000)
        runner = MjlabOnPolicyRunner(env,copy.deepcopy(config),str(directory),device=device)
        devices = [torch.device(device).index or 0] if device.startswith('cuda') else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed+20000)
            if devices:
                torch.cuda.manual_seed_all(seed+20000)
            for layer in runner.alg.critic.modules():
                if isinstance(layer,torch.nn.Linear):
                    layer.reset_parameters()
        record['critic_initialization_controlled'] = True
        record['initial_critic_sha256'] = state_sha(runner.alg.critic)
        record['initial_actor_sha256'] = state_sha(runner.alg.actor)
        record['actor_budget'] = study.model_budget(runner.alg.actor)
        record['critic_budget'] = study.model_budget(runner.alg.critic)
        record['device'] = device
        record['device_name'] = torch.cuda.get_device_name(device) if devices else platform.processor()
        study.write(directory/'environment.json',environment_record())
        # Capture startup domain randomization, which must match within each seed.
        record['startup_physics'] = {key:__import__('hashlib').sha256(
            getattr(raw.sim.model,key).detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            for key in ('body_mass','body_ipos','geom_friction')}
        study.seed_all(seed+30000)
        raw.reset(seed=seed+30000)
        if recovery:
            checkpoint,state,rejected = recovery
            runner.load(str(checkpoint),map_location=device,strict=True)
            carried = state['completed_updates']
            runner.current_learning_iteration = carried
            runner.alg.learning_rate = state['learning_rate']
            random = __import__('random')
            random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng'])
            torch.set_rng_state(state['torch_rng'].cpu())
            if devices:
                torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda_rng']])
            record.update(recovered_updates=carried,rejected_checkpoints=rejected,
                          recovery_checkpoint_sha256=study.sha(checkpoint))
        original_step = env.step
        def checked_step(action):
            nonlocal collected
            if not torch.isfinite(action).all():
                raise ValueError('nonfinite action')
            result = original_step(action)
            if protocol.get('terrain_mode') == 'blind_rough':
                contacts, constraints = study.check_rough_simulation(raw)
                record['peak_contacts'] = max(record.get('peak_contacts', 0), contacts)
                record['peak_constraints_per_world'] = max(record.get('peak_constraints_per_world', 0), constraints)
            collected += num_envs
            return result
        env.step = checked_step
        original_log = runner.logger.log
        def checked_log(**kwargs):
            losses = {k:float(v) for k,v in kwargs['loss_dict'].items()}
            if not all(np.isfinite(v) for v in losses.values()):
                raise ValueError('nonfinite PPO loss')
            row = dict(iteration=int(kwargs['it']),losses=losses,
                       collect_seconds=float(kwargs['collect_time']),learn_seconds=float(kwargs['learn_time']))
            with (directory/'raw_training_metrics.jsonl').open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(row,allow_nan=False)+'\n')
            return original_log(**kwargs)
        runner.logger.log = checked_log
        original_save = runner.save
        def checked_save(path,infos=None):
            completed = runner.current_learning_iteration+1
            metadata = dict(identity=identity,completed_updates=completed,learning_rate=runner.alg.learning_rate,
                            python_rng=__import__('random').getstate(),numpy_rng=np.random.get_state(),
                            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if devices else [])
            original_save(path,dict(infos or {},split_study=metadata))
            record.update(completed_updates=completed,
                          completed_transitions=completed*num_envs*agent.num_steps_per_env,
                          current_attempt_transitions=collected,latest_checkpoint=Path(path).name,
                          updated_at=study.now())
            study.write(manifest,record)
        runner.save = checked_save
        study.write(manifest,record)
        started = time.perf_counter()
        remaining = iterations-carried
        if remaining:
            runner.learn(num_learning_iterations=remaining,init_at_random_ep_len=True)
        else:
            runner.current_learning_iteration = iterations-1
            runner.save(str(directory/f'model_{iterations-1}.pt'))
        if collected != remaining*num_envs*agent.num_steps_per_env:
            raise ValueError('observed environment transitions differ from the fixed budget')
        checkpoint = directory/f'model_{iterations-1}.pt'
        if not checkpoint.is_file():
            raise ValueError('final-budget checkpoint missing')
        runner.export_policy_to_onnx(str(directory),'policy.onnx')
        record['onnx_parity'] = export_parity(runner.alg.actor,directory/'policy.onnx')
        if source_snapshot()['sha256'] != source['sha256']:
            raise ValueError('source content changed while training')
        record.update(status='COMPLETED',ended_at=study.now(),training_wall_seconds=time.perf_counter()-started,
                      checkpoint_file=checkpoint.name,checkpoint_sha256=study.sha(checkpoint),
                      policy_file='policy.onnx',policy_sha256=study.sha(directory/'policy.onnx'),
                      completed_updates=iterations,completed_transitions=iterations*num_envs*agent.num_steps_per_env)
        verify_training(record,directory,protocol,source,arm,seed,stage)
        return record
    except BaseException as error:
        record.update(status='FAILED',ended_at=study.now(),error=repr(error),traceback=traceback.format_exc())
        study.write(directory/'failure.json',{'error':repr(error),'traceback':traceback.format_exc()})
        raise
    finally:
        record['current_attempt_transitions'] = collected
        study.write(manifest,record)
        if env is not None:
            env.close()


def export_parity(actor,path):
    import numpy as np
    import torch
    import onnxruntime as ort
    model = actor.as_onnx(verbose=False).cpu().eval()
    options = ort.SessionOptions();options.intra_op_num_threads=options.inter_op_num_threads=1
    session = ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
    inputs = np.random.default_rng(20261026).normal(size=(32,98)).astype(np.float32)
    with torch.inference_mode():
        expected = model(torch.from_numpy(inputs)).numpy()
    actual = np.concatenate([session.run(None,{'obs':x[None]})[0] for x in inputs])
    np.testing.assert_allclose(actual,expected,rtol=2e-4,atol=2e-5)
    return {'passed':True,'backend':'torch CPU fp32 vs ONNX Runtime CPU fp32','samples':len(inputs),
            'max_abs_error':float(np.max(np.abs(actual-expected))),'rtol':2e-4,'atol':2e-5}


def seal_formal(output):
    output = Path(output)
    protocol = prepare(output);source=read(output/'source_manifest.json')
    rows=[];per_seed={}
    for job in study.training_jobs(protocol,'formal'):
        directory=run_directory(output,**job);record=read(directory/'manifest.json')
        verify_training(record,directory,protocol,source,**job)
        key=job['seed']
        shared={k:record[k] for k in ('common_env_sha256','common_agent_sha256','initial_critic_sha256','startup_physics')}
        if key in per_seed and per_seed[key] != shared:
            raise ValueError('training fairness identities differ within a seed')
        per_seed[key]=shared
        rows.append(dict(arm=job['arm'],seed=key,policy=str((directory/'policy.onnx').resolve()),
                         checkpoint=str((directory/record['checkpoint_file']).resolve()),
                         policy_sha256=record['policy_sha256'],checkpoint_sha256=record['checkpoint_sha256']))
    seal={'protocol_sha256':study.protocol_digest(protocol),'source_sha256':source['sha256'],
          'selection':protocol['checkpoint_selection'],'policies':rows}
    path=output/'formal_policies_sealed.json'
    if path.exists() and read(path)!=seal:
        raise ValueError('an existing final-policy seal would change')
    study.validate_policy_seal(seal,protocol)
    study.write(path,seal)
    return seal


def gpu_processes():
    import psutil
    result=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader,nounits'],
                          capture_output=True,text=True,check=True)
    jobs=[]
    for line in result.stdout.splitlines():
        if not line.strip():continue
        number,name=line.split(',',1);pid=int(number.strip())
        if pid==os.getpid():continue
        try:
            process=psutil.Process(pid)
            jobs.append({'pid':pid,'name':name.strip(),'created_at':process.create_time()})
        except psutil.NoSuchProcess:continue
        except psutil.AccessDenied:jobs.append({'pid':pid,'name':name.strip(),'identity':'ACCESS_DENIED'})
    return jobs


def admission_decision(jobs,*,free_mib,required_mib):
    if jobs:return 'WAITING_FOR_GPU'
    if free_mib<required_mib:return 'INSUFFICIENT_GPU_MEMORY'
    return 'READY'


def wait_gpu(output,required_mib,*,wait):
    while True:
        rows=subprocess.run(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],
                            capture_output=True,text=True,check=True).stdout.splitlines()
        status=admission_decision(gpu_processes(),free_mib=int(rows[0]),required_mib=required_mib)
        study.write(Path(output)/'queue_status.json',{'status':status,'at':study.now(),'pid':os.getpid(),
                    'required_free_mib':required_mib,'free_mib':int(rows[0])})
        if status=='READY':return
        if not wait or status=='INSUFFICIENT_GPU_MEMORY':
            raise RuntimeError(status+'; no other process was stopped')
        time.sleep(30)


def child(output,stage,*,arm=None,seed=None,device='cuda:0',resume=False,split=None):
    command=[sys.executable,'-u','-m',MODULE,'--output',str(Path(output).resolve()),'--stage',stage,'--device',device]
    for key,value in (('arm',arm),('seed',seed),('split',split)):
        if value is not None:command+=['--'+key,str(value)]
    if resume:command+=['--resume']
    env=dict(os.environ,PYTHONHASHSEED=str(seed or 20261026),PYTHONIOENCODING='utf-8',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    log=Path(output)/'logs'/f'{stage}_{arm}_{seed}_{time.time_ns()}.log';log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('w',encoding='utf-8') as stream:
        process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
        study.write(Path(output)/'queue_status.json',{'status':'RUNNING','stage':stage,'arm':arm,'seed':seed,
                    'pid':os.getpid(),'child_pid':process.pid,'log':str(log),'at':study.now()})
        code=process.wait()
    if code:raise RuntimeError(f'child exit {code}; see {log}')


def queue(output,*,device='cuda:0',resume=False,wait=False):
    import psutil
    output=Path(output);protocol=prepare(output)
    lock=output/'queue.lock'
    try:
        fd=os.open(lock,os.O_WRONLY|os.O_CREAT|os.O_EXCL)
    except FileExistsError:
        old=read(lock)
        try:
            process=psutil.Process(old['pid'])
            live=abs(process.create_time()-old['created_at'])<.01
        except psutil.NoSuchProcess:live=False
        if live or not resume:raise RuntimeError('queue lock exists; active queue or explicit recovery required')
        lock.rename(output/f'queue_stale_{time.time_ns()}.json')
        fd=os.open(lock,os.O_WRONLY|os.O_CREAT|os.O_EXCL)
    with os.fdopen(fd,'w',encoding='utf-8') as stream:
        json.dump({'pid':os.getpid(),'created_at':psutil.Process().create_time()},stream)
    try:
        for arm in study.ARMS:
            if device.startswith('cuda'):wait_gpu(output,2000,wait=wait)
            child(output,'smoke',arm=arm,seed=protocol['training']['pilot_seed'],device=device,resume=resume)
        for stage in ('pilot','formal'):
            for job in study.training_jobs(protocol,stage):
                if device.startswith('cuda'):wait_gpu(output,22000,wait=wait)
                child(output,**job,device=device,resume=resume)
        seal_formal(output)
        for job in study.training_jobs(protocol,'formal'):
            if device.startswith('cuda'):wait_gpu(output,3000,wait=wait)
            child(output,'evaluate',arm=job['arm'],seed=job['seed'],device=device,split='holdout',resume=resume)
        child(output,'latency',device='cpu')
        child(output,'analyze',device='cpu')
        study.write(output/'queue_status.json',{'status':'COMPLETED','at':study.now(),'pid':os.getpid()})
    except BaseException as error:
        study.write(output/'queue_status.json',{'status':'INCOMPLETE','at':study.now(),'pid':os.getpid(),
                    'error':repr(error),'traceback':traceback.format_exc()})
        raise
    finally:
        lock.unlink(missing_ok=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=study.DEFAULT_OUT)
    parser.add_argument('--stage',choices=['prepare','smoke','pilot','formal','seal','evaluate','evaluate-smoke','latency','latency-smoke','analyze','queue','status'],required=True)
    parser.add_argument('--arm',choices=study.ARMS)
    parser.add_argument('--seed',type=int)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--smoke-envs',type=int,default=4)
    parser.add_argument('--split',choices=['development','holdout'],default='holdout')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--wait-for-gpu',action='store_true')
    args=parser.parse_args()
    if args.stage=='prepare':
        prepare(args.output);print('Prepared',args.output)
    elif args.stage in ('smoke','pilot','formal'):
        if args.arm is None or args.seed is None:parser.error('--arm and --seed required')
        directory=run_directory(args.output,args.arm,args.seed,args.stage);directory.mkdir(parents=True,exist_ok=True)
        with (directory/f'console_{time.time_ns()}.log').open('w',encoding='utf-8') as stream:
            with contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
                result=train(args.output,args.arm,args.seed,args.stage,device=args.device,
                             smoke_envs=args.smoke_envs,resume=args.resume)
        print(result['status'],directory)
    elif args.stage=='seal':
        seal_formal(args.output);print('All 12 formal policies sealed')
    elif args.stage=='evaluate':
        if args.arm is None or args.seed is None:parser.error('--arm and --seed required')
        from scripts.evaluate_g1_semantic_split import evaluate
        evaluate(args.output,args.arm,args.seed,args.split,device=args.device,resume=args.resume)
    elif args.stage=='evaluate-smoke':
        if args.arm is None or args.seed is None:parser.error('--arm and --seed required')
        from scripts.evaluate_g1_semantic_split import smoke_evaluate
        smoke_evaluate(args.output,args.arm,args.seed,device=args.device)
    elif args.stage in ('latency','latency-smoke'):
        from scripts.evaluate_g1_semantic_split import benchmark_all
        benchmark_all(args.output,smoke=args.stage=='latency-smoke')
    elif args.stage=='analyze':
        from scripts.analyze_g1_semantic_split import analyze
        analyze(args.output)
    elif args.stage=='queue':
        queue(args.output,device=args.device,resume=args.resume,wait=args.wait_for_gpu)
    else:
        print(json.dumps(read(args.output/'queue_status.json'),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
