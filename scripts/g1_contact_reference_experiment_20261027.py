"""One frozen causal-contact reference trained against the stage-A matched controls."""
from dataclasses import replace
import argparse,concurrent.futures,gc,json,math,os,statistics,subprocess,sys,time,traceback
from pathlib import Path
from scripts import g1_phase_lr_factorial_20261027 as a
ROOT=a.ROOT;AOUT=ROOT/'results/g1_phase_lr_factorial_20261027';OUT=ROOT/'results/g1_contact_phase_reference_20261027'
SPEC=ROOT/'docs/superpowers/specs/2026-10-27-g1-contact-phase-reference-design.md'
PLAN=ROOT/'docs/superpowers/plans/2026-10-27-g1-contact-phase-reference.md'
ORIGINAL_SETUP=a.setup
BASE_MAKE_CFG=None
PRE_CFG_DIGEST={}
FIELDS=('initial_actor_sha256','initial_critic_sha256','common_cfg_sha256','common_agent_sha256','startup_model_hashes')
PREFIX="import site;site.addsitedir('C:/Users/郑傅韬/AppData/Roaming/Python/Python313/site-packages');import runpy,sys;"

def check_initialization(left,right):
    for key in FIELDS:
        if key not in left or key not in right or left[key]!=right[key]:raise ValueError('Unpaired initialization: '+key)
    return True

def paired_differences(rows,treatment,control):
    grouped={}
    for r in rows:
        if r['arm'] not in (treatment,control):continue
        key=(r['seed'],r['arm'],r['eval_seed'])
        value=float(r['mae_forward_velocity_mps'])
        if key in grouped or not math.isfinite(value):raise ValueError('duplicate/nonfinite episode')
        grouped[key]=value
    seeds=sorted({s for s,_,_ in grouped});result=[]
    if not seeds:raise ValueError('empty paired results')
    for seed in seeds:
        left={e:v for (s,arm,e),v in grouped.items() if s==seed and arm==treatment}
        right={e:v for (s,arm,e),v in grouped.items() if s==seed and arm==control}
        if not left or set(left)!=set(right):raise ValueError('missing paired episode')
        result.append(statistics.mean(left[e]-right[e] for e in left))
    return result

def verify_a():
    protocol=a.read(AOUT/'prospective_protocol.json');files=list(protocol['identity']['sources'])
    for seal in ('extended_source_seal.json','parallel_execution_seal.json'):
        files+=a.read(AOUT/seal)['files']
    for row in files:assert a.sha(row['path'])==row['sha256'],('stage-A source changed',row['path'])
    assert a.sha(AOUT/'analysis.json')==a.read(AOUT/'queue_status.json')['analysis_sha256']
    assert a.read(AOUT/'analysis.json')['status']=='COMPLETED'
    return files

def lock():
    files=verify_a()
    calibration=a.read(OUT/'calibration_continuous.json')
    assert calibration['passed'],'reference calibration is infeasible; do not launch training'
    for row in a.read(OUT/'calibration_collection_lock.json')['files']:
        if a.sha(row['path'])!=row['sha256']:
            original=OUT/'technical_failure_01/source_snapshot'/Path(row['path']).relative_to(ROOT)
            assert Path(row['path']).name in ('g1_contact_phase_reference.py','test_g1_contact_phase_reference_20261027.py')
            assert a.sha(original)==row['sha256'],('original calibration source not preserved',row['path'])
    assert a.sha(calibration['raw_path'])==calibration['raw_sha256']
    paths=[Path(__file__),ROOT/'scripts/g1_contact_phase_reference.py',SPEC,PLAN,
        ROOT/'tests/test_g1_contact_phase_reference_20261027.py',ROOT/'tests/test_g1_contact_reference_experiment_20261027.py',
        OUT/'calibration_continuous.json',OUT/'calibration_collection_lock.json',OUT/'calibration_collection_manifest.json',
        OUT/'calibration_rule_lock.json',OUT/'calibration.json',OUT/'calibration_technical_amendment_01.md',OUT/'technical_amendment_02.json',
        OUT/'technical_failure_01/prospective_protocol.json',OUT/'technical_amendment_03.json',
        OUT/'technical_failure_02/prospective_protocol.json',AOUT/'analysis.json']
    files += [dict(path=str(p),sha256=a.sha(p)) for p in paths]
    unique={r['path']:r for r in files}
    identity=dict(files=list(unique.values()),train_seeds=list(a.SEEDS),checkpoints=list(a.CHECKPOINTS),
        arm='reference',initial_lr=.00001,adaptive_schedule=True,updates_per_run=501,num_envs=64,rollout_steps=24,
        formal_transitions=2308608,formal_episodes=135,technical_smoke_transitions=3072,failed_smoke_extra_transitions=64,calibration_transitions=32000,
        targets=calibration['targets'],controls='stage-A clock_low and learned_low; paired initialization/physics audited',
        primary='fixed501 full20s body-forward MAE with zero velocity after fall',
        family=['reference_minus_clock','reference_minus_E'],statistical_unit='3 training seeds; episodes nested',
        additional_evaluation_seeds=list(range(20263411,20263421)),confirmatory=False)
    path=OUT/'prospective_protocol.json'
    if path.exists():
        old=a.read(path);assert old['identity']==identity,'frozen stage-B source changed';return old
    value=dict(status='LOCKED_BEFORE_REFERENCE_TRAINING',at=a.now(),identity=identity)
    a.write(path,value)
    for row in paths:
        if row.is_relative_to(ROOT) and row.suffix in ('.py','.md'):
            target=OUT/'source_snapshot'/row.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(row.read_bytes())
    return value

def reference_phase(env,period=.6,command_name='twist'):
    import torch
    from scripts.g1_contact_phase_reference import ContactPhaseReference
    if not hasattr(env,'_contact_phase_reference'):
        targets=a.read(OUT/'calibration_continuous.json')['targets']
        env._contact_phase_reference=ContactPhaseReference(env.num_envs,env.device,targets)
    state=env._contact_phase_reference
    phase=state.update(int(env.common_step_counter),env.episode_length_buf,env.scene['feet_ground_contact'].data.found>0)
    stand=torch.linalg.norm(env.command_manager.get_command(command_name),dim=1)<.1
    return torch.where(stand[:,None],torch.zeros_like(phase),phase)

def reset_reference(env,env_ids):
    if hasattr(env,'_contact_phase_reference'):env._contact_phase_reference.reset(env_ids)

def make_cfg(terrain,arm,seed,*,play,num_envs,horizon):
    if arm!='reference':return BASE_MAKE_CFG(terrain,arm,seed,play=play,num_envs=num_envs,horizon=horizon)
    from mjlab.managers.event_manager import EventTermCfg
    cfg=BASE_MAKE_CFG(terrain,'clock',seed,play=play,num_envs=num_envs,horizon=horizon)
    for group in ('actor','critic'):
        cfg.observations[group].terms['phase']=replace(cfg.observations[group].terms['phase'],func=reference_phase,params={})
    cfg.events['reset_phasor_phase_history']=EventTermCfg(func=reset_reference,mode='reset',params={})
    if not play and num_envs==64:
        from scripts.g1_phase_adaptation_20261024 import common_cfg_digest
        digest=common_cfg_digest(cfg)
        for control in ('clock','learned'):
            assert digest==a.read(AOUT/'training'/f'{control}_low_{seed}'/'manifest.json')['common_cfg_sha256'],'pre-construction common configuration mismatch'
        PRE_CFG_DIGEST[seed]=digest
    return cfg

def setup():
    global BASE_MAKE_CFG
    study,torch=ORIGINAL_SETUP()
    if BASE_MAKE_CFG is None:BASE_MAKE_CFG=study.make_cfg
    study.make_cfg=make_cfg
    return study,torch

def install():
    a.OUT=OUT;a.setup=setup;a.lock=lock

def train(seed,smoke=False):
    install();study,_=setup()
    from mjlab.rl import MjlabOnPolicyRunner
    from mjlab.tasks.registry import load_rl_cfg
    from dataclasses import asdict
    from scripts import diagnose_g1_initial_update_20261026 as probe
    import hashlib
    native=MjlabOnPolicyRunner.save
    def prechecked_save(runner,path,infos=None):
        state=(infos or {}).get('factorial_state',{})
        if state.get('completed_updates')==0:
            raw=runner.env.unwrapped
            agent=load_rl_cfg('Unitree-G1-Flat');agent.seed=seed;agent.logger='tensorboard';agent.upload_model=False;agent.save_interval=10**9
            common=study.canonical(asdict(agent));common['algorithm'].pop('learning_rate')
            observed=dict(initial_actor_sha256=study.base.state_sha(runner.alg.actor),initial_critic_sha256=study.base.state_sha(runner.alg.critic),
                common_cfg_sha256=PRE_CFG_DIGEST[seed],common_agent_sha256=hashlib.sha256(json.dumps(common,sort_keys=True).encode()).hexdigest(),
                startup_model_hashes={k:probe.digest_tensor(getattr(raw.sim.model,k)) for k in ('body_mass','body_ipos','geom_friction')})
            for arm in ('clock','learned'):check_initialization(observed,a.read(AOUT/'training'/f'{arm}_low_{seed}'/'manifest.json'))
            a.write(OUT/('smoke' if smoke else 'training')/f'reference_low_{seed}'/'pre_sampling_pairing.json',dict(status='PASS',at=a.now(),transitions=0,optimizer_updates=0,fields=observed))
        return native(runner,path,infos)
    MjlabOnPolicyRunner.save=prechecked_save
    try:a.train('reference','low',seed,smoke=smoke)
    finally:MjlabOnPolicyRunner.save=native
    m=a.read(OUT/('smoke' if smoke else 'training')/f'reference_low_{seed}'/'manifest.json')
    check_initialization(m,a.read(AOUT/'training'/f'clock_low_{seed}'/'manifest.json'))
    check_initialization(m,a.read(AOUT/'training'/f'learned_low_{seed}'/'manifest.json'))

def evaluate(seed,point,smoke=False):
    install();_,_=setup()
    from src.tasks.velocity.mdp import observations
    original=observations.phase;observations.phase=reference_phase
    try:a.evaluate('reference','low',seed,point,smoke=smoke)
    finally:observations.phase=original

def verify_smoke():
    from scripts.summarize_g1_best_phase_control import score
    folder=OUT/'smoke/reference_low_20261901';m=a.read(folder/'manifest.json')
    assert m['status']=='COMPLETED' and m['updates']==2 and m['transitions']==3072
    assert m['protocol_sha256']==a.sha(OUT/'prospective_protocol.json') and set(m['checkpoints'])=={'0','1','2'}
    for entry in m['checkpoints'].values():assert a.sha(entry['path'])==entry['sha256'] and a.sha(entry['policy'])==entry['policy_sha256']
    for arm in ('clock','learned'):check_initialization(m,a.read(AOUT/'training'/f'{arm}_low_20261901'/'manifest.json'))
    evaluation=OUT/'smoke_evaluation/reference_low_20261901_u2'
    em,rows,raws=score(evaluation)
    assert em['status']=='COMPLETED' and em['horizon']==.2 and len(rows)==1
    assert em['protocol_sha256']==m['protocol_sha256'] and em['policy_sha256']==m['checkpoints']['2']['policy_sha256']
    assert len(next(iter(raws.values()))['time'])==10
    a.write(OUT/'smoke_verification.json',dict(status='PASS',at=a.now(),updates=2,transitions=3072,
        protocol_sha256=a.sha(OUT/'prospective_protocol.json'),training_manifest_sha256=a.sha(folder/'manifest.json'),evaluation_manifest_sha256=a.sha(evaluation/'manifest.json'),
        pre_sampling_pairing=a.read(folder/'pre_sampling_pairing.json')['status'],original_failed_extra_transitions=64))

def cell(seed,smoke=False):
    train(seed,smoke)
    for point in ((2,) if smoke else a.CHECKPOINTS):
        gc.collect()
        import torch
        torch.cuda.empty_cache();evaluate(seed,point,smoke)

def worker(index,seeds):
    completed=[];path=OUT/f'parallel_worker_{index}.json'
    for seed in seeds:
        args=['--stage','cell','--seed',str(seed)];module='scripts.g1_contact_reference_experiment_20261027'
        command=PREFIX+'sys.argv='+repr([module]+args)+';runpy.run_module('+repr(module)+',run_name="__main__")'
        log=OUT/'logs'/f'reference_low_{seed}.log';log.parent.mkdir(exist_ok=True)
        with log.open('a',encoding='utf-8') as stream:
            child=subprocess.Popen([sys.executable,'-u','-c',command],cwd=ROOT,
                env=dict(os.environ,PYTHONHASHSEED='20263300',PYTHONIOENCODING='utf-8'),stdout=stream,stderr=stream,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            a.write(path,dict(status='RUNNING',at=a.now(),seed=seed,child_pid=child.pid,completed=completed,log=str(log)))
            rc=child.wait()
        if rc:
            a.write(path,dict(status='FAILED',at=a.now(),seed=seed,completed=completed,exit_code=rc,log=str(log)))
            raise RuntimeError('reference training/evaluation failed: '+str(seed))
        completed.append(seed)
    a.write(path,dict(status='COMPLETED',at=a.now(),completed=completed));return completed

def queue():
    lock();smoke_gate=a.read(OUT/'smoke_verification.json');assert smoke_gate['status']=='PASS'
    assert smoke_gate['protocol_sha256']==a.sha(OUT/'prospective_protocol.json')
    assert smoke_gate['training_manifest_sha256']==a.sha(OUT/'smoke/reference_low_20261901/manifest.json')
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(worker,i,a.SEEDS[i::2]) for i in range(2)]
        while not all(f.done() for f in futures):
            for f in futures:
                if f.done() and f.exception():raise f.exception()
            a.write(OUT/'queue_status.json',dict(status='RUNNING',at=a.now(),pid=os.getpid(),maximum_concurrent_cells=2,
                workers=[a.read(p) for p in [OUT/f'parallel_worker_{i}.json' for i in range(2)] if p.exists()]))
            time.sleep(10)
        completed=[s for f in futures for s in f.result()]
    a.write(OUT/'queue_status.json',dict(status='TRAINING_EVALUATION_COMPLETED',at=a.now(),completed=completed))
    analyze()
    a.write(OUT/'queue_status.json',dict(status='STAGE_B_COMPLETED',at=a.now(),completed=completed,
        analysis_sha256=a.sha(OUT/'analysis.json'),next_branch=a.read(OUT/'analysis.json')['candidate']))

def analyze():
    lock()
    import numpy as np
    from statsmodels.stats.multitest import multipletests
    from scripts.analyze_g1_phase_lr_factorial_20261027 import extra,effect,METRICS
    from scripts.summarize_g1_best_phase_control import score
    training=[];episodes=[];pairing=[];aggregates=[];coverage=[]
    for seed in a.SEEDS:
        ref=a.read(OUT/'training'/f'reference_low_{seed}'/'manifest.json')
        for arm,base in [('reference',OUT),('clock',AOUT),('learned',AOUT)]:
            folder=base/'training'/f'{arm}_low_{seed}';m=a.read(folder/'manifest.json')
            assert m['status']=='COMPLETED' and m['updates']==501 and m['transitions']==769536 and m['encoder_frozen']
            assert m['protocol_sha256']==a.sha(base/'prospective_protocol.json')
            check_initialization(ref,m);training.append(m)
            traces=[json.loads(x) for x in (folder/'updates.jsonl').read_text(encoding='utf-8').splitlines()]
            assert [x['completed_update'] for x in traces]==list(range(1,502)) and all(len(x['minibatches'])==20 for x in traces)
        for point in a.CHECKPOINTS:
            scored={}
            for arm,base in [('reference',OUT),('clock',AOUT),('learned',AOUT)]:
                name=f'{arm}_low_{seed}';m=a.read(base/'training'/name/'manifest.json');entry=m['checkpoints'][str(point)]
                assert a.sha(entry['path'])==entry['sha256'] and a.sha(entry['policy'])==entry['policy_sha256']
                manifest,rows,raws=score(base/'evaluation'/f'{name}_u{point}')
                expected={(sc,e) for sc in (['steady','speed_change','push'] if point==501 else ['steady']) for e in (range(20263411,20263421) if point==501 else range(20263411,20263414))}
                assert len(rows)==len(expected) and {(r['scenario'],r['seed']) for r in rows}==expected
                assert manifest['policy_sha256']==entry['policy_sha256'] and manifest['protocol_sha256']==m['protocol_sha256']
                scored[arm]=(manifest,rows,raws);coverage.append(dict(arm=arm,seed=seed,point=point,episodes=len(rows)))
                for row in rows:
                    raw=raws[('flat',row['scenario'],row['seed'])]
                    episodes.append(extra(raw,dict(row,eval_seed=row['seed'],seed=seed,completed_updates=point,lr_mode='low')))
            for arm in ('reference','learned'):
                assert scored[arm][0]['startup_model_hashes']==scored['clock'][0]['startup_model_hashes']
                for key,raw in scored[arm][2].items():
                    control=scored['clock'][2][key]
                    errors={k:float(np.max(np.abs(raw[k][0].astype(float)-control[k][0].astype(float)))) for k in ('position','quaternion','joint_pos','velocity')}
                    assert max(errors.values())<=1e-6
                    prefix=min(len(raw['command']),len(control['command']));np.testing.assert_array_equal(raw['command'][:prefix],control['command'][:prefix])
                    pairing.append(dict(arm=arm,seed=seed,point=point,scenario=key[1],eval_seed=key[2],errors=errors))
    for point in a.CHECKPOINTS:
        for scenario in (['steady','speed_change','push'] if point==501 else ['steady']):
            for seed in (*a.SEEDS,'all'):
                for arm in ('reference','clock','learned'):
                    rows=[r for r in episodes if r['completed_updates']==point and r['scenario']==scenario and r['arm']==arm and (seed=='all' or r['seed']==seed)]
                    aggregates.append(dict(point=point,scenario=scenario,seed=seed,arm=arm,episodes=len(rows),falls=sum(r['fell'] for r in rows),
                        **{k:float(np.mean([r[k] for r in rows if r[k] is not None])) if any(r[k] is not None for r in rows) else None for k in METRICS}))
    final=[r for r in episodes if r['completed_updates']==501 and r['scenario']=='steady']
    contrasts={name:effect(paired_differences(final,'reference',control)) for name,control in [('reference_minus_clock','clock'),('reference_minus_E','learned')]}
    for value,q in zip(contrasts.values(),multipletests([r['p'] for r in contrasts.values()],method='holm')[1]):value['holm_p']=float(q)
    ref=[r for r in final if r['arm']=='reference'];clock=[r for r in final if r['arm']=='clock'];by={(r['seed'],r['eval_seed']):r for r in clock}
    avg=lambda rows,key:float(np.mean([r[key] for r in rows]))
    guards=dict(falls=sum(r['fell'] and not by[(r['seed'],r['eval_seed'])]['fell'] for r in ref),tilt=avg(ref,'tilt_rms_rad')-avg(clock,'tilt_rms_rad'),
        heading=avg(ref,'abs_heading_change_rad')-avg(clock,'abs_heading_change_rad'),displacement=avg(ref,'initial_direction_displacement_m')-avg(clock,'initial_direction_displacement_m'))
    candidate=dict(a.candidate_gate(contrasts['reference_minus_clock']['per_seed'],guards),guards=guards)
    assert len([r for r in episodes if r['arm']=='reference'])==135 and sum(m['transitions'] for m in training if m['arm']=='reference')==2308608
    result=dict(status='COMPLETED',at=a.now(),protocol_sha256=a.sha(OUT/'prospective_protocol.json'),training=training,coverage=coverage,pairing=pairing,
        episodes=episodes,aggregates=aggregates,contrasts=contrasts,candidate=candidate,
        scope='Exploratory three training seeds, reference adds contact information and nominal prior; no stable E superiority conclusion')
    a.write(OUT/'analysis.json',result)
    lines=['# 因果触地参考相位：等预算对照','', '新增3项501更新、135回合；A对照重新评分并审计初始化/物理配对。','',
        '|更新|场景|相位|MAE(m/s)|倾斜|航向|初始方向位移|跌倒/n|','|---:|---|---|---:|---:|---:|---:|---:|']
    for r in aggregates:
        if r['seed']=='all':lines.append(f"|{r['point']}|{r['scenario']}|{r['arm']}|{r['mae_forward_velocity_mps']:.5f}|{r['tilt_rms_rad']:.4f}|{r['abs_heading_change_rad']:.3f}|{r['initial_direction_displacement_m']:.3f}|{r['falls']}/{r['episodes']}|")
    lines+=['','预定seed级比较：'+json.dumps(contrasts,ensure_ascii=False),'','候选门：'+json.dumps(candidate,ensure_ascii=False),'',result['scope']]
    (OUT/'diagnosis.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',required=True,choices=['lock','smoke','train','cell','queue','analyze'])
    parser.add_argument('--seed',type=int);args=parser.parse_args()
    try:
        if args.stage=='lock':lock()
        elif args.stage=='smoke':cell(a.SEEDS[0],True);verify_smoke()
        elif args.stage=='train':train(args.seed)
        elif args.stage=='cell':cell(args.seed)
        elif args.stage=='queue':queue()
        else:analyze()
    except BaseException as error:
        if args.stage=='queue':a.write(OUT/'queue_status.json',dict(status='FAILED',at=a.now(),error=repr(error),traceback=traceback.format_exc()))
        raise
