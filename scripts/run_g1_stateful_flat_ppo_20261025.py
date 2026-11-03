"""GPU-aware serial queue for frozen E, flat matched-budget PPO comparisons."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import g1_stateful_flat_ppo as study

MODULE = 'scripts.run_g1_stateful_flat_ppo_20261025'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def gpu_jobs():
    """Query real compute processes; unknown identities block rather than disappear."""
    import psutil
    result = subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name',
                             '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
    jobs = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        number, name = line.split(',',1)
        pid = int(number.strip())
        if pid == os.getpid():
            continue
        try:
            p = psutil.Process(pid)
            jobs.append(dict(pid=pid, name=name.strip(), argv=p.cmdline(), created_at=p.create_time()))
        except psutil.NoSuchProcess:
            continue
        except psutil.AccessDenied:
            jobs.append(dict(pid=pid, name=name.strip(), identity='ACCESS_DENIED'))
    return jobs


def wait_gpu():
    while True:
        jobs = gpu_jobs()
        if not jobs:
            return
        study.write(study.OUT/'queue_status.json',dict(status='WAITING_FOR_GPU',
                    updated_at=study.now(), queue_pid=os.getpid(), other_gpu_jobs=jobs))
        time.sleep(30)


def online_smoke():
    np, torch = study.bootstrap()
    from scripts.g1_stateful_online import load_online
    from scripts.g1_stateful_phase import causal_features
    from scipy.spatial.transform import Rotation
    rng = np.random.default_rng(20261900)
    q = np.cumsum(rng.normal(0,.004,(180,29)),axis=0).astype(np.float32)
    xyzw = Rotation.from_rotvec(np.cumsum(rng.normal(0,.003,(180,3)),axis=0)).as_quat()
    quat = xyzw[:,[3,0,1,2]].astype(np.float32)
    runtime = load_online(study.RUNTIME,num_envs=1,device='cuda:0')
    features = torch.tensor(causal_features(q,quat),device='cuda:0')
    with torch.inference_mode():
        expected = runtime.encoder(((features-runtime.mean)/runtime.std)[None])['phase'][0]
        actual = torch.stack([runtime.observe(torch.tensor(q[i:i+1],device='cuda:0'),
                            torch.tensor(quat[i:i+1],device='cuda:0'),i)[0].clone() for i in range(len(q))])
    torch.testing.assert_close(actual,expected,rtol=2e-3,atol=2e-3)
    study.write(study.OUT/'smoke/online_parity.json',dict(status='PASS',at=study.now(),
                max_abs_error=float((actual-expected).abs().max()),frames=len(q),
                encoder_sha256=study.sha(study.RUNTIME),device=torch.cuda.get_device_name(),
                precision='float32; TF32 disabled',rtol=.002,atol=.002,
                smoke_identity=study.smoke_identity()))


def worker(stage, arm, seed, *, resume=False, label=None):
    # Reuse the sealed training implementation without altering its source or old outputs.
    from scripts import run_g1_phase_adaptation_20261024 as original
    original.study = study
    study.bootstrap()
    if stage in ('smoke','train'):
        try:
            original.train('flat',arm,seed,smoke=stage=='smoke',resume=resume)
        finally:
            directory = study.OUT/'smoke'/f'flat_{arm}' if stage=='smoke' else study.train_dir('flat',arm,seed)
            path = directory/'manifest.json'
            if path.exists():
                record = read(path)
                record.update(experiment='new E flat-only PPO',encoder_selection='smallest E seed; not performance selected',
                    protocol_sha256=study.sha(ROOT/'docs/superpowers/specs/2026-10-25-g1-stateful-flat-ppo-design.md'),
                    online_adapter_sha256=study.sha(ROOT/'scripts/g1_stateful_online.py'))
                if stage=='smoke':
                    record['smoke_identity'] = study.smoke_identity()
                if record['status']=='FAILED':
                    record['resume_command'] = [sys.executable,'-u','-m',MODULE,'--stage','train',
                                                '--arm',arm,'--seed',str(seed),'--resume']
                study.write(path,record)
    else:
        verify_policies()
        from scripts import g1_best_phase_batched as batch
        batch.bootstrap = study.bootstrap
        original.evaluate('flat',seed,label)


def verify_smoke(record):
    if (record.get('status') not in ('PASS','COMPLETED')
            or record.get('encoder_sha256')!=study.RUNTIME_SHA
            or record.get('smoke_identity')!=study.smoke_identity()):
        raise ValueError('smoke identity changed or did not pass; preserve old attempt before technical rerun')


def verify_training(record, directory):
    if (record['status']!='COMPLETED' or record['encoder_sha256']!=study.RUNTIME_SHA
            or not record['encoder_frozen_verified'] or record['iterations']!=study.ITERATIONS
            or record['completed_learning_iterations']!=study.ITERATIONS
            or record['completed_environment_transitions']!=study.TRANSITIONS
            or study.sha(directory/'policy.onnx')!=record['policy_sha256']
            or study.sha(Path(record['final_checkpoint']))!=record['final_checkpoint_sha256']):
        raise ValueError('final training policy does not satisfy frozen identity/budget')


def freeze_policies():
    study.verify_assets()
    records = []
    for job in study.training_jobs():
        directory = study.train_dir(**job)
        manifest = read(directory/'manifest.json')
        verify_training(manifest,directory)
        records.append(dict(**job,policy=str(directory/'policy.onnx'),sha256=manifest['policy_sha256'],
                            checkpoint=manifest['final_checkpoint'],checkpoint_sha256=manifest['final_checkpoint_sha256']))
    path = study.OUT/'final_policies_frozen.json'
    if path.exists():
        if read(path)['policies']!=records:
            raise ValueError('final policy seal would change')
    else:
        study.write(path,dict(frozen_at=study.now(),policies=records,selection='all final 501-update policies'))
    return verify_policies()


def verify_policies():
    seal = read(study.OUT/'final_policies_frozen.json')
    expected = {(j['arm'],j['seed']) for j in study.training_jobs()}
    if len(seal['policies'])!=6 or {(r['arm'],r['seed']) for r in seal['policies']}!=expected:
        raise ValueError('all six policies must be frozen before test')
    for row in seal['policies']:
        if study.sha(row['policy'])!=row['sha256'] or study.sha(row['checkpoint'])!=row['checkpoint_sha256']:
            raise ValueError('frozen policy changed')
    return seal


def classify_effects(effects):
    return (effects[0]['paired_t_95pct_ci'][1] < 0 and effects[0]['holm_adjusted_p'] < .05
            and effects[1]['paired_t_95pct_ci'][1] <= 0)


def summarize():
    study.verify_assets()
    seal = verify_policies()
    import numpy as np
    from statsmodels.stats.multitest import multipletests
    from scripts.summarize_g1_best_phase_control import score,aggregate
    rows, aggregates, pair_checks, training_checks = [], [], [], []
    fields = ('source_actor_sha256','initial_actor_state_sha256','initial_critic_state_sha256',
              'iterations','num_envs','num_steps_per_env','completed_environment_transitions',
              'completed_learning_iterations','common_task_config_sha256','ppo_config_sha256','startup_model_hashes')
    for seed in study.TRAIN_SEEDS:
        training = {arm:read(study.train_dir('flat',arm,seed)/'manifest.json') for arm in ('clock','learned')}
        for arm, record in training.items():
            verify_training(record,study.train_dir('flat',arm,seed))
        for field in fields:
            if training['clock'][field]!=training['learned'][field]:
                raise ValueError(f'training pair mismatch: {seed}/{field}')
        training_checks.append(dict(seed=seed,fields_equal=list(fields),encoder_frozen=True,budget=study.TRANSITIONS))
        groups, manifests = {}, {}
        for job in study.evaluation_jobs('flat',seed):
            path = Path(job['output'])
            manifest,current,raw = score(path)
            expected = {('flat',c,s) for s in study.EVAL_SEEDS for c in study.SCENARIOS}
            if (len(current)!=30 or set(raw)!=expected or manifest['episode_count']!=30
                    or manifest['encoder_sha256']!=study.RUNTIME_SHA or manifest['started_at']<seal['frozen_at']):
                raise ValueError('evaluation cohort identity or freeze order invalid')
            source = training['learned' if job['label']=='learned_learned' else 'clock']
            if manifest['policy_sha256']!=source['policy_sha256']:
                raise ValueError('wrong policy used in evaluation')
            manifests[job['label']],groups[job['label']] = manifest,raw
            for row in current:
                row.update(train_seed=seed,label=job['label'])
            rows.extend(current)
            aggregates.append(dict(terrain='flat',train_seed=seed,label=job['label'],**aggregate(current)))
        for label in ('clock_learned','learned_learned'):
            if manifests[label]['startup_model_hashes']!=manifests['clock_clock']['startup_model_hashes']:
                raise ValueError('evaluation initial physics differ')
            for key,a in groups['clock_clock'].items():
                b = groups[label][key]
                errors = {k:float(np.max(np.abs(a[k][0].astype(float)-b[k][0].astype(float))))
                          for k in ('position','quaternion','joint_pos','velocity')}
                if max(errors.values()) > 1e-6:
                    raise ValueError('evaluation initial states not paired')
                n = min(len(a['command']),len(b['command']))
                np.testing.assert_array_equal(a['command'][:n],b['command'][:n])
                pair_checks.append(dict(seed=seed,label=label,key=key,errors=errors))
    if len(rows)!=270:
        raise ValueError('full 270 episode matrix is required')
    families = []
    for control in ('clock_clock','clock_learned'):
        effects = [study.paired_training_effect(rows,'flat','learned_learned',control,metric)
                   for metric in ('mae_forward_velocity_mps','fall_rate')]
        adjusted = multipletests([e['two_sided_p'] for e in effects],method='holm')[1]
        for effect,p in zip(effects,adjusted):
            effect['holm_adjusted_p'] = float(p)
        families.append(dict(control=control,treatment='learned_learned',effects=effects,
                             flat_improvement_established=bool(classify_effects(effects))))
    result = dict(status='COMPLETED',computed_at=study.now(),scope='flat finite-budget frozen E adaptation',
                  encoder_sha256=study.RUNTIME_SHA,training_checks=training_checks,pairing_checks=pair_checks,
                  aggregates=aggregates,episodes=rows,inference=families,
                  statement='three PPO runs; conditional on fixed finite test cohort; no rough/real-robot claim')
    study.write(study.OUT/'comparison.json',result)
    lines = ['# 新E冻结相位的平地PPO三组结果','',
             '3个配对PPO seed；每臂501更新、64环境、769536 transition；共270评测回合。',
             '候选按最小E种子编号选定，没有按离线或PPO成绩重选。',
             '', '| PPO seed | 训练→测试 | 跌倒/n | 速度MAE m/s | 存活 s |',
             '|---:|---|---:|---:|---:|']
    for r in aggregates:
        lines.append(f"| {r['train_seed']} | {r['label']} | {r['falls']}/{r['episodes']} | {r['mae_forward_velocity_mps']:.4f} | {r['survival_seconds']:.2f} |")
    for family in families:
        lines.extend(['',f"learned→learned 对 {family['control']}：平地固定预算改善{'已建立' if family['flat_improvement_established'] else '尚未建立'}。"])
        for e in family['effects']:
            lines.append(f"- {e['metric']}：差 {e['mean_difference']:.5f}；三训练seed的95% t区间 {e['paired_t_95pct_ci']}；Holm p={e['holm_adjusted_p']:.5g}。")
    lines.extend(['','跌倒后剩余时间按零速度计入20秒MAE。所有跌倒/无周期/未恢复记录保留。',
                  '这是短程warm-start平地仿真对照，未建立改善不等于等效或原理无效；不证明rough或真机表现。',''])
    (study.OUT/'comparison.md').write_text('\n'.join(lines),encoding='utf-8')
    study.write(study.OUT/'completion_verification.json',dict(status='PASS',at=study.now(),training_runs=6,
                evaluations=9,episodes=270,updates_per_run=study.ITERATIONS,transitions_per_run=study.TRANSITIONS,
                training_pairs_verified=3,evaluation_initial_state_pairs=len(pair_checks),encoder_frozen=True))


def child(stage, name, **kwargs):
    wait_gpu()
    logs = study.OUT/'logs'
    logs.mkdir(parents=True,exist_ok=True)
    command = [sys.executable,'-u','-m',MODULE,'--stage',stage]
    for key,value in kwargs.items():
        if value is True:
            command.append('--'+key)
        elif value is not False:
            command.extend(['--'+key,str(value)])
    log = logs/f'{name}_{time.time_ns()}.log'
    environment = dict(os.environ,PYTHONHASHSEED=str(kwargs.get('seed',20261900)),
                       PYTHONIOENCODING='utf-8',WANDB_MODE='disabled')
    with log.open('w',encoding='utf-8') as stream:
        process = subprocess.Popen(command,cwd=ROOT,env=environment,stdout=stream,stderr=subprocess.STDOUT)
        study.write(study.OUT/'queue_status.json',dict(status='RUNNING',stage=stage,task=name,
                    queue_pid=os.getpid(),child_pid=process.pid,argv=command,log_path=str(log),updated_at=study.now()))
        code = process.wait()
    if code:
        raise RuntimeError(f'child failed ({code}), preserved: {log}')


def queue():
    import psutil
    study.OUT.mkdir(parents=True,exist_ok=True)
    lock = study.OUT/'queue.lock'
    if lock.exists():
        previous = read(lock)
        try:
            process = psutil.Process(previous['pid'])
            if (abs(process.create_time()-previous['create_time'])<.1
                    and MODULE in ' '.join(process.cmdline())):
                raise RuntimeError('a verified flat PPO queue is already active')
        except psutil.NoSuchProcess:
            pass
        lock.rename(study.OUT/f'queue_stale_{time.time_ns()}.json')
    status = study.OUT/'queue_status.json'
    if status.exists():
        old = read(status)
        old = old.get('last_stage',old)
        if old.get('child_pid') and psutil.pid_exists(old['child_pid']):
            p = psutil.Process(old['child_pid'])
            if MODULE in ' '.join(p.cmdline()):
                raise RuntimeError('verified previous GPU child still active')
    identity = dict(pid=os.getpid(),create_time=psutil.Process().create_time(),argv=sys.argv,started_at=study.now())
    with lock.open('x',encoding='utf-8') as stream:
        json.dump(identity,stream)
    try:
        if not study.dependency_ready():
            raise RuntimeError('original offline delivery has not passed its final audit')
        study.verify_runtime()
        parity = study.OUT/'smoke/online_parity.json'
        if not parity.exists():
            child('online-smoke','online_parity')
        verify_smoke(read(parity))
        for arm in ('clock','learned'):
            path = study.OUT/'smoke'/f'flat_{arm}'/'manifest.json'
            if path.exists() and read(path)['status']=='COMPLETED':
                verify_smoke(read(path))
                continue
            child('smoke',f'smoke_{arm}',arm=arm,seed=study.TRAIN_SEEDS[0])
            verify_smoke(read(path))
        study.freeze_assets()
        for job in study.training_jobs():
            directory = study.train_dir(**job)
            path = directory/'manifest.json'
            if path.exists() and read(path)['status']=='COMPLETED':
                verify_training(read(path),directory)
                continue
            resume = path.exists()
            if resume:
                study.choose_resume_checkpoint(directory,job['seed'],'flat',job['arm'])
            child('train',f"{job['arm']}_{job['seed']}",arm=job['arm'],seed=job['seed'],resume=resume)
        freeze_policies()
        for seed in study.TRAIN_SEEDS:
            for job in study.evaluation_jobs('flat',seed):
                path = Path(job['output'])/'manifest.json'
                if path.exists() and read(path)['status']=='COMPLETED':
                    continue
                child('evaluate',f"eval_{seed}_{job['label']}",seed=seed,label=job['label'])
        summarize()
        study.write(status,dict(status='COMPLETED',ended_at=study.now(),comparison=str(study.OUT/'comparison.md')))
    except BaseException as error:
        previous = read(status) if status.exists() else None
        study.write(status,dict(status='FAILED',ended_at=study.now(),error=repr(error),
                    traceback=traceback.format_exc(),last_stage=previous))
        raise
    finally:
        if lock.exists() and read(lock).get('pid')==os.getpid():
            lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['queue','online-smoke','smoke','train','evaluate','summarize'],required=True)
    parser.add_argument('--arm',choices=['clock','learned'],default='clock')
    parser.add_argument('--seed',type=int,default=study.TRAIN_SEEDS[0])
    parser.add_argument('--label',choices=['clock_clock','clock_learned','learned_learned'])
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args()
    if args.stage in ('online-smoke','smoke','train','evaluate') and gpu_jobs():
        raise RuntimeError('another GPU compute process is active; use the waiting queue')
    if args.stage=='queue': queue()
    elif args.stage=='online-smoke': online_smoke()
    elif args.stage=='summarize': summarize()
    else: worker(args.stage,args.arm,args.seed,resume=args.resume,label=args.label)


if __name__=='__main__':
    main()
