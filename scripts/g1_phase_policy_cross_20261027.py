"""Frozen policy x causal phase cross; no new PPO updates or selected endpoints."""
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import argparse,json,math,os,random,statistics,subprocess,sys,traceback
from scripts import g1_phase_lr_factorial_20261027 as a
from scripts import g1_contact_reference_experiment_20261027 as b
ROOT=a.ROOT;OUT=ROOT/'results/g1_phase_policy_cross_20261027'
SPEC=ROOT/'docs/superpowers/specs/2026-10-27-g1-phase-policy-cross-design.md'
PLAN=ROOT/'docs/superpowers/plans/2026-10-27-g1-phase-policy-cross.md'
CONTRASTS={
 'E_input_clock_minus_E':(('learned','clock'),('learned','learned')),
 'E_weights_under_clock':(('learned','clock'),('clock','clock')),
 'reference_input_clock_minus_reference':(('reference','clock'),('reference','reference')),
 'reference_weights_under_clock':(('reference','clock'),('clock','clock')),
 'source_E_minus_clock':(('source','learned'),('source','clock')),
 'source_reference_minus_clock':(('source','reference'),('source','clock'))}
PREFIX="import site;site.addsitedir('C:/Users/郑傅韬/AppData/Roaming/Python/Python313/site-packages');import runpy,sys;"

def build_jobs():
    jobs=[dict(controller=p,phase=q) for p in ('source','clock','learned','reference') for q in ('clock','learned','reference')]
    random.Random(20263690).shuffle(jobs);return jobs

def world_slots():return [dict(train_seed=s,eval_seed=e) for s in a.SEEDS for e in range(20263611,20263614)]

def route_actions(obs,sessions,names):
    import numpy as np
    if obs.shape!=(9,98) or len(sessions)!=9 or len(names)!=9:raise ValueError('nine aligned worlds required')
    action=np.concatenate([session.run(None,{name:obs[i:i+1]})[0] for i,(session,name) in enumerate(zip(sessions,names))])
    if action.shape!=(9,29) or not np.isfinite(action).all():raise ValueError('invalid routed action')
    return action

def mechanism_contrasts(rows):
    expected={(j['controller'],j['phase'],s['train_seed'],s['eval_seed']) for j in build_jobs() for s in world_slots()};by={}
    for row in rows:
        key=(row['controller'],row['phase'],row['train_seed'],row['eval_seed'])
        if key in by or not math.isfinite(row['mae_forward_velocity_mps']):raise ValueError('duplicate or nonfinite cross episode')
        by[key]=row['mae_forward_velocity_mps']
    if set(by)!=expected:raise ValueError('missing or extra cross episode')
    output={}
    for name,(left,right) in CONTRASTS.items():
        values=[statistics.mean(by[(*left,s,e)]-by[(*right,s,e)] for e in range(20263611,20263614)) for s in a.SEEDS]
        output[name]=dict(per_seed=values,mean=statistics.mean(values),seeds=list(a.SEEDS),scope='descriptive, fixed nine-world cohort; source is shared')
    return output

def policy_entry(controller,seed):
    base=b.OUT if controller=='reference' else b.AOUT
    name=f'{controller}_low_{seed}' if controller!='source' else 'clock_low_20261901'
    folder=base/'training'/name;m=a.read(folder/'manifest.json');assert m['status']=='COMPLETED'
    point='0' if controller=='source' else '501';entry=m['checkpoints'][point]
    assert a.sha(entry['path'])==entry['sha256'] and a.sha(entry['policy'])==entry['policy_sha256']
    return dict(controller=controller,train_seed=seed,completed_updates=int(point),checkpoint=entry['path'],checkpoint_sha256=entry['sha256'],
        policy=entry['policy'],policy_sha256=entry['policy_sha256'],training_manifest=str(folder/'manifest.json'),training_manifest_sha256=a.sha(folder/'manifest.json'))

def lock():
    b.lock();assert a.read(b.OUT/'analysis.json')['status']=='COMPLETED'
    assert a.sha(b.OUT/'analysis.json')==a.read(b.OUT/'queue_status.json')['analysis_sha256']
    paths=[Path(__file__),SPEC,PLAN,ROOT/'tests/test_g1_phase_policy_cross_20261027.py',b.OUT/'analysis.json',b.OUT/'prospective_protocol.json',b.AOUT/'analysis.json']
    sources=[dict(path=str(p),sha256=a.sha(p)) for p in paths]
    policies=[policy_entry(p,s) for p in ('source','clock','learned','reference') for s in a.SEEDS]
    identity=dict(sources=sources,policies=policies,jobs=build_jobs(),slots=world_slots(),construct_seed=20263600,horizon=20.,dt=.02,
        formal_episodes=108,maximum_environment_transitions=108000,PPO_updates=0,contrasts=CONTRASTS,
        primary='20s body-forward MAE; zero velocity after fall',statistics='seed-level descriptive, no superiority claim',
        selection='all frozen501 and source0; no score-selected checkpoint')
    # JSON canonicalization makes tuple/list identity stable across process boundaries.
    identity=json.loads(json.dumps(identity))
    path=OUT/'prospective_protocol.json'
    if path.exists():
        old=a.read(path);assert old['identity']==identity,'cross frozen source changed';return old
    value=dict(status='LOCKED_BEFORE_NEW_CROSS_ROLLOUTS',at=a.now(),identity=identity);a.write(path,value)
    for p in paths:
        if p.suffix in ('.py','.md'):
            target=OUT/'source_snapshot'/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(p.read_bytes())
    return value

def rollout(controller,phase):
    lock();study,torch=b.setup()
    import numpy as np
    import onnxruntime as ort
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.utils.os import dump_yaml
    from src.tasks.velocity.mdp import observations
    from src.tasks.velocity.phasor.data import G1_JOINT_ORDER
    from scripts.g1_closed_loop_metrics import command_schedule,summarize_episode
    name=f'{controller}_input_{phase}';folder=OUT/'rollouts'/name
    if (folder/'manifest.json').exists():
        old=a.read(folder/'manifest.json');assert old['status']=='COMPLETED' and old['protocol_sha256']==a.sha(OUT/'prospective_protocol.json');return
    folder.mkdir(parents=True,exist_ok=False);slots=world_slots();entries=[policy_entry(controller,s['train_seed']) for s in slots]
    options=ort.SessionOptions();options.intra_op_num_threads=options.inter_op_num_threads=1
    cache={};sessions=[];input_names=[]
    for entry in entries:
        key=entry['policy'];session=cache.setdefault(key,None)
        if session is None:session=cache[key]=ort.InferenceSession(key,sess_options=options,providers=['CPUExecutionProvider'])
        sessions.append(session);input_names.append(session.get_inputs()[0].name)
    random.seed(20263600);np.random.seed(20263600);torch.manual_seed(20263600);torch.cuda.manual_seed_all(20263600)
    cfg=b.make_cfg('flat',phase,20263600,play=True,num_envs=9,horizon=20.)
    dump_yaml(folder/'env.yaml',asdict(cfg))
    phase_func=b.reference_phase if phase=='reference' else (lambda env:observations.learned_phase(env,str(a.ENCODER))) if phase=='learned' else (lambda env:observations.phase(env,.6,'twist'))
    record=dict(status='RUNNING',started_at=a.now(),terrain='flat',arm=controller,controller=controller,phase=phase,slots=slots,
        policies=entries,horizon=20.,scenarios=['steady'],parallel_worlds=9,protocol_sha256=a.sha(OUT/'prospective_protocol.json'),
        encoder_sha256=a.ENCODER_SHA,transitions=0,episode_count=0,common_cfg_sha256=study.common_cfg_digest(cfg),PPO_updates=0)
    a.write(folder/'manifest.json',record);env=None;episodes=[]
    try:
        env=ManagerBasedRlEnv(cfg=cfg,device='cuda:0');assert abs(float(env.step_dt)-.02)<1e-9
        record['startup_model_hashes']={k:__import__('hashlib').sha256(np.ascontiguousarray(getattr(env.sim.model,k).cpu().numpy()).tobytes()).hexdigest() for k in ('body_mass','body_ipos','geom_friction')}
        for i,slot in enumerate(slots):
            seed=slot['eval_seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            env.reset(seed=seed,env_ids=torch.tensor([i],device=env.device))
        origins=env.scene.env_origins.cpu().numpy().copy();robot=env.scene['robot'];contacts=env.scene['feet_ground_contact']
        keys=('time','command','position','quaternion','velocity','angle','contact','joint_pos','actions','gravity')
        raw=[{k:[] for k in keys} for _ in slots];done=np.zeros(9,bool);fell=done.copy();commands=command_schedule(1000,.02,'steady')
        for step,command in enumerate(commands):
            env.command_manager.get_command('twist')[:]=torch.tensor([command,0.,0.],device=env.device)
            env.observation_manager._obs_buffer=None
            obs=env.observation_manager.compute()['actor'].cpu().numpy().astype(np.float32)
            vector=phase_func(env).cpu().numpy().copy()
            np.testing.assert_allclose(obs[:,9:11],vector,atol=1e-7);np.testing.assert_allclose(np.linalg.norm(vector,axis=1),1.,atol=2e-6)
            np.testing.assert_allclose(obs[:,6:9],np.tile([command,0.,0.],(9,1)),atol=1e-7)
            if not np.isfinite(obs).all():raise ValueError('nonfinite cross observations')
            action=route_actions(obs,sessions,input_names)
            values={k:v.detach().cpu().numpy().copy() for k,v in (
                ('position',robot.data.root_link_pos_w),('quaternion',robot.data.root_link_quat_w),('velocity',robot.data.root_link_lin_vel_b),
                ('gravity',robot.data.projected_gravity_b),('joint_pos',observations._joint_positions_in_order(robot,robot.data.joint_pos,G1_JOINT_ORDER)),
                ('contact',contacts.data.current_contact_time>0))}
            values.update(angle=np.arctan2(vector[:,0],vector[:,1]),actions=action.copy())
            for i in np.flatnonzero(~done):
                raw[i]['time'].append(step*.02);raw[i]['command'].append(command)
                for k,v in values.items():raw[i][k].append(v[i].copy())
            record['transitions']+=int((~done).sum());action[done]=0
            result=env.step(torch.as_tensor(action,device=env.device));terminated=result[2].cpu().numpy().astype(bool);timed=result[3].cpu().numpy().astype(bool)
            if bool((timed&~done).any()):raise ValueError('unexpected timeout')
            fell|=terminated&~done;done|=terminated
            if step%250==249:
                a.write(folder/'progress.json',dict(step=step+1,recorded_transitions=record['transitions'],at=a.now(),active=int((~done).sum())))
            if done.all():break
        for i,slot in enumerate(slots):
            values={k:np.asarray(v) for k,v in raw[i].items()};episode_id=20263700+i;path=folder/f'steady_{episode_id}.npz';np.savez_compressed(path,**values)
            row=summarize_episode(dict(values,scenario='steady'),commands,.02,bool(fell[i]));row.update(scenario='steady',seed=episode_id,
                **slot,slot=i,raw_path=path.name,raw_sha256=a.sha(path),initial_env_origin=origins[i].tolist())
            episodes.append(row);a.write(folder/'episodes.json',episodes)
        record.update(status='COMPLETED',ended_at=a.now(),episode_count=len(episodes))
    except BaseException as error:
        record.update(status='FAILED',ended_at=a.now(),error=repr(error),traceback=traceback.format_exc());raise
    finally:
        a.write(folder/'manifest.json',record)
        if env is not None:env.close()

def analyze():
    lock()
    import numpy as np
    from scripts.summarize_g1_best_phase_control import score
    from scripts.analyze_g1_phase_lr_factorial_20261027 import extra,METRICS
    episodes=[];pairing=[];starts={};manifests=[];aggregates=[]
    for job in build_jobs():
        controller,phase=job['controller'],job['phase'];folder=OUT/'rollouts'/f'{controller}_input_{phase}'
        manifest,rows,raws=score(folder);assert manifest['protocol_sha256']==a.sha(OUT/'prospective_protocol.json') and len(rows)==9
        manifests.append(manifest);originals={r['seed']:r for r in a.read(folder/'episodes.json')}
        assert set(originals)==set(range(20263700,20263709))
        for row in rows:
            source=originals[row['seed']];slot=source['slot'];assert dict(train_seed=source['train_seed'],eval_seed=source['eval_seed'])==world_slots()[slot]
            raw=raws[('flat','steady',row['seed'])];starts[(controller,phase,slot)]={k:raw[k][0].copy() for k in ('position','quaternion','joint_pos','velocity')}
            r=extra(raw,dict(row,controller=controller,phase=phase,train_seed=source['train_seed'],eval_seed=source['eval_seed'],slot=slot));episodes.append(r)
        for seed in (*a.SEEDS,'all'):
            chosen=[r for r in episodes if r['controller']==controller and r['phase']==phase and (seed=='all' or r['train_seed']==seed)]
            aggregates.append(dict(controller=controller,phase=phase,train_seed=seed,n=len(chosen),falls=sum(r['fell'] for r in chosen),
                **{k:float(np.mean([r[k] for r in chosen if r[k] is not None])) if any(r[k] is not None for r in chosen) else None for k in METRICS}))
    for manifest in manifests:
        assert manifest['startup_model_hashes']==manifests[0]['startup_model_hashes'] and manifest['common_cfg_sha256']==manifests[0]['common_cfg_sha256']
    for key,values in starts.items():
        base=starts[('clock','clock',key[2])];errors={k:float(np.max(np.abs(values[k].astype(float)-base[k].astype(float)))) for k in base}
        assert max(errors.values())<=1e-6;pairing.append(dict(key=list(key),errors=errors))
    assert len(episodes)==108 and sum(m['transitions'] for m in manifests)<=108000
    result=dict(status='COMPLETED',at=a.now(),protocol_sha256=a.sha(OUT/'prospective_protocol.json'),episodes=episodes,aggregates=aggregates,
        contrasts=mechanism_contrasts(episodes),pairing=pairing,recorded_transitions=sum(m['transitions'] for m in manifests),PPO_updates=0,
        scope='new nine-world cohort; descriptive mechanism interventions; source weights shared across seed slots; no stable superiority conclusion')
    a.write(OUT/'analysis.json',result)
    lines=['# 冻结策略×相位交叉诊断','',result['scope'],'','|策略来源|输入相位|MAE|航向变化|初始方向位移|倾斜|跌倒/n|','|---|---|---:|---:|---:|---:|---:|']
    for r in aggregates:
        if r['train_seed']=='all':lines.append(f"|{r['controller']}|{r['phase']}|{r['mae_forward_velocity_mps']:.5f}|{r['abs_heading_change_rad']:.3f}|{r['initial_direction_displacement_m']:.3f}|{r['tilt_rms_rad']:.4f}|{r['falls']}/{r['n']}|")
    lines+=['','预定六项差值：'+json.dumps(result['contrasts'],ensure_ascii=False)]
    (OUT/'diagnosis.md').write_text(chr(10).join(lines)+chr(10),encoding='utf-8')

def queue():
    lock();completed=[]
    for job in build_jobs():
        name=f"{job['controller']}_input_{job['phase']}";folder=OUT/'rollouts'/name
        if (folder/'manifest.json').exists():
            assert a.read(folder/'manifest.json')['status']=='COMPLETED','preserve failed attempt';completed.append(name);continue
        args=['--stage','rollout','--controller',job['controller'],'--phase',job['phase']];module='scripts.g1_phase_policy_cross_20261027'
        command=PREFIX+'sys.argv='+repr([module]+args)+';runpy.run_module('+repr(module)+',run_name="__main__")'
        log=OUT/'logs'/f'{name}.log';log.parent.mkdir(parents=True,exist_ok=True)
        with log.open('a',encoding='utf-8') as stream:
            child=subprocess.Popen([sys.executable,'-u','-c',command],cwd=ROOT,stdout=stream,stderr=stream,
                env=dict(os.environ,PYTHONHASHSEED='20263300',PYTHONIOENCODING='utf-8',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'),creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            a.write(OUT/'queue_status.json',dict(status='RUNNING',at=a.now(),pid=os.getpid(),current=name,child_pid=child.pid,completed=completed,log=str(log)))
            rc=child.wait()
        if rc:raise RuntimeError(name+' failed; preserve attempt: '+str(rc))
        completed.append(name)
    a.write(OUT/'queue_status.json',dict(status='ANALYZING',at=a.now(),completed=completed));analyze()
    a.write(OUT/'queue_status.json',dict(status='STAGE_C_COMPLETED',at=a.now(),completed=completed,analysis_sha256=a.sha(OUT/'analysis.json')))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',required=True,choices=['lock','rollout','queue','analyze']);parser.add_argument('--controller');parser.add_argument('--phase');args=parser.parse_args()
    try:
        if args.stage=='lock':lock()
        elif args.stage=='rollout':rollout(args.controller,args.phase)
        elif args.stage=='queue':queue()
        else:analyze()
    except BaseException as error:
        if args.stage=='queue':a.write(OUT/'queue_status.json',dict(status='FAILED',at=a.now(),error=repr(error),traceback=traceback.format_exc()))
        raise
