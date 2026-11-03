"""Finite frozen-policy rough transfer. No fitting, PPO, or score selection."""
from copy import deepcopy
from dataclasses import asdict,replace
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import os
import random
import subprocess
import sys
import traceback
import numpy as np

from scripts import g1_phase_lr_factorial_20261027 as a
from scripts import g1_rough_transfer_contract_20261028 as c

ROOT=a.ROOT
OUT=ROOT/'results/g1_rough_transfer_20261028'
AOUT=ROOT/'results/g1_phase_lr_factorial_20261027'
DOUT=ROOT/'results/g1_source_anchor_20261027'
SPEC=ROOT/'docs/superpowers/specs/2026-10-28-g1-rough-transfer-design.md'
PLAN=ROOT/'docs/superpowers/plans/2026-10-28-g1-rough-transfer.md'
PREFIX="import site;site.addsitedir('C:/Users/郑傅韬/AppData/Roaming/Python/Python313/site-packages');import runpy,sys;"


def seed_all(torch,seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)


def deterministic_terrain(env,env_ids=None):
    """Replace, rather than append to, native randomize_terrain reset event."""
    import torch
    terrain=env.scene.terrain
    if terrain.terrain_origins is None:return
    ids=torch.arange(env.num_envs,device=env.device) if env_ids is None else env_ids
    cols=ids%5
    terrain.terrain_levels[ids]=0;terrain.terrain_types[ids]=cols
    terrain.env_origins[ids]=terrain.terrain_origins[0,cols]


def make_cfg(study,method,terrain_seed,num_envs=15):
    from mjlab.managers.event_manager import EventTermCfg
    arm='learned' if method.endswith('E') else 'clock'
    cfg=study.make_cfg('flat',arm,20264200,play=True,num_envs=num_envs,horizon=20.)
    if terrain_seed:
        native=study.previous.make_cfg('rough','clock',20264200,play=True,num_envs=num_envs,horizon=20.)
        cfg.scene.terrain=deepcopy(native.scene.terrain)
        g=cfg.scene.terrain.terrain_generator
        g.seed=terrain_seed;g.curriculum=False;g.num_rows=1;g.num_cols=5;g.size=(32.,16.)
        rough=deepcopy(g.sub_terrains['random_rough'])
        rough.proportion=1.;rough.noise_range=(.02,.10);rough.noise_step=.02
        rough.horizontal_scale=.1;rough.vertical_scale=.005;rough.border_width=.25
        g.sub_terrains={'random_rough':rough}
    cfg.sim.mujoco.ccd_iterations=500;cfg.sim.contact_sensor_maxmatch=500
    cfg.sim.njmax=1500;cfg.sim.nconmax=128
    assert 'randomize_terrain' in cfg.events
    cfg.events['randomize_terrain']=EventTermCfg(func=deterministic_terrain,mode='reset')
    order=list(cfg.events)
    assert order.index('randomize_terrain')<order.index('reset_base')
    assert list(cfg.observations['actor'].terms)==['base_ang_vel','projected_gravity','command','phase','joint_pos','joint_vel','actions']
    return cfg


def prepare_output(folder):
    Path(folder).mkdir(parents=True,exist_ok=False)


def assigned_patches(geom_pos,hfield_ids,origins,rough):
    assigned=[];bounds=[]
    for origin in origins:
        if rough:
            ids=[int(g) for g in hfield_ids if np.allclose(geom_pos[g,:2],origin[:2],atol=1e-5,rtol=0)]
            if len(ids)!=1:raise ValueError('assigned patch does not match exactly one physical hfield')
        else:ids=[]
        assigned.append(set(ids));x,y=origin[:2];bounds.append([x-16,x+16,y-8,y+8])
    return assigned,np.asarray(bounds)


def policy_entry(method,seed):
    if method not in c.METHODS or seed not in c.TRAIN_SEEDS:raise ValueError('outside frozen policy matrix')
    phase='learned' if method.endswith('E') else 'clock'
    folder=(AOUT if method.startswith('A') else DOUT)/'training'/f'{phase}_low_{seed}'
    m=a.read(folder/'manifest.json')
    if m['status']!='COMPLETED':raise ValueError('incomplete policy source')
    entry=m['checkpoints']['501']
    if a.sha(entry['path'])!=entry['sha256'] or a.sha(entry['policy'])!=entry['policy_sha256']:raise ValueError('policy source hash mismatch')
    return dict(method=method,train_seed=seed,phase=phase,completed_updates=501,
        checkpoint=entry['path'],checkpoint_sha256=entry['sha256'],policy=entry['policy'],policy_sha256=entry['policy_sha256'],
        training_manifest=str(folder/'manifest.json'),training_manifest_sha256=a.sha(folder/'manifest.json'))


def sources():
    paths={Path(__file__),Path(c.__file__),SPEC,PLAN,OUT/'PROJECT_PLAN.md',ROOT/'tests/test_g1_rough_transfer_20261028.py'}
    paths.update(OUT/n for n in ('experiment_matrix.md','preregistration.md','reproducibility-checklist.md',
        'plan_spec.json','target-chain.json','target_chain_report.json','failure-tree.json','failure_tree_report.json',
        'plan_findings.json','plan_package.json','research_package_report.json','builder_check.md','implementation_review.md','cpu_preflight.json'))
    # Verify, rather than mutate/relock, all transitive frozen A/D source manifests.
    for directory,field in ((AOUT,'sources'),(DOUT,'files')):
        protocol=directory/'prospective_protocol.json'
        paths.add(protocol);paths.add(directory/'analysis.json')
        for entry in a.read(protocol)['identity'][field]:
            p=Path(entry['path'])
            if a.sha(p)!=entry['sha256']:raise ValueError('parent frozen source changed: '+str(p))
            paths.add(p)
    lib=Path(importlib.metadata.distribution('mjlab').locate_file('mjlab'))
    paths.update(lib/p for p in ('terrains/terrain_entity.py','terrains/terrain_generator.py','terrains/heightfield_terrains.py','envs/manager_based_rl_env.py','sim/sim.py'))
    warp_lib=Path(importlib.metadata.distribution('mujoco-warp').locate_file('mujoco_warp'))
    paths.add(warp_lib/'_src/types.py')
    return [dict(path=str(p),sha256=a.sha(p)) for p in sorted(paths,key=str)]


def lock():
    pre=a.read(OUT/'cpu_preflight.json')
    if pre['status']!='PASS' or pre['experiment_sha256']!=a.sha(__file__) or pre['contract_sha256']!=a.sha(c.__file__):raise ValueError('matching zero-sampling preflight required')
    if a.read(OUT/'research_package_report.json')['verdict'] not in ('PASS','WARN'):raise ValueError('research plan package has unresolved errors')
    if a.sha(DOUT/'prospective_protocol.json')!='ed980fdda67e18b9ffb1c9de3b5b69a385137b383838966ccc66478d97e00ab1':raise ValueError('D parent protocol changed')
    if a.sha(DOUT/'analysis.json')!='bb647176dd7b5227264324f66a98841b00f6f8d5648789486a608bcb1729bbf7':raise ValueError('D parent analysis changed')
    identity=dict(files=sources(),policies=[policy_entry(m,s) for m in c.METHODS for s in c.TRAIN_SEEDS],
        jobs=c.build_jobs(),slots=c.world_slots(),scenarios=c.SCENARIOS,dt=.02,horizon=20.,command=.5,
        push_world_y_delta_mps={5.:.3,10.:-.3},construct_seed=20264200,formal_episodes=480,
        maximum_effective_eval_transitions=480000,smoke_maximum_eval_transitions=40,PPO_updates=0,Adam_steps=0,
        candidate_guards=c.GUARDS,unit='three paired training seeds; conditional on 15 shared rough patches',
        family='4 contrasts x completion/fall; exact seed sign flip; 8 Holm',
        selection='all AC/AE/DC/DE checkpoint501; no outcome-selected seed/checkpoint/terrain',
        packages={n:importlib.metadata.version(n) for n in ('torch','mjlab','mujoco','mujoco-warp','warp-lang','onnxruntime','numpy','scipy')})
    identity=json.loads(json.dumps(identity))
    path=OUT/'prospective_protocol.json'
    if path.exists():
        old=a.read(path)
        if old['identity']!=identity:raise ValueError('frozen R protocol/source changed')
        return old
    value=dict(status='LOCKED_BEFORE_R_ENVIRONMENT_SAMPLING',at=a.now(),identity=identity)
    a.write(path,value)
    for entry in identity['files']:
        p=Path(entry['path'])
        if p.suffix in ('.py','.md'):
            relative=p.relative_to(ROOT) if p.is_relative_to(ROOT) else Path('dependencies')/entry['sha256'][:12]/p.name
            dst=OUT/'source_snapshot'/relative;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(p.read_bytes())
    return value


def physical_snapshot(env):
    model=env.sim.mj_model
    arrays={k:np.asarray(getattr(model,k)).copy() for k in ('hfield_data','hfield_size','hfield_nrow','hfield_ncol','hfield_adr','geom_pos','geom_quat','geom_type','geom_dataid','geom_bodyid','body_pos','body_quat')}
    for k in ('body_mass','body_ipos','geom_friction'):
        arrays['runtime_'+k]=getattr(env.sim.model,k).detach().cpu().numpy().copy()
    return arrays


def pairing_guard(path,identity):
    if path.exists():
        if a.read(path)!=identity:raise ValueError('pre-sampling physical/configuration/initial-state pairing failed: '+str(path))
    else:a.write(path,identity)


def read_contacts(env,foot_ids,terrain_ids,assigned,bounds):
    data=env.sim.wp_data;contact=data.contact
    count=int(data.nacon.numpy()[0]);capacity=contact.geom.shape[0]
    if count<0 or count>=capacity or int(data.nefc.numpy().max())>=1500:raise ValueError('contact/constraint capacity saturation; refusing truncation')
    arrays={k:getattr(contact,k).numpy()[:count].copy() for k in ('geom','worldid','pos','dist','includemargin','efc_address')}
    result=c.contact_exposure(arrays['geom'],arrays['worldid'],arrays['pos'],arrays['dist'],arrays['includemargin'],arrays['efc_address'],foot_ids,terrain_ids,assigned,bounds,env.num_envs)
    return result,{k:v[result['indices']] for k,v in arrays.items()},count


def rollout(method,terrain_seed,smoke=False):
    lock()
    study,torch=a.setup()
    import onnxruntime as ort
    import mujoco
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.utils.os import dump_yaml
    from scripts.g1_stateful_online import learned_phase
    from src.tasks.velocity.mdp import observations
    from src.tasks.velocity.phasor.data import G1_JOINT_ORDER
    name=('smoke_' if smoke else '')+f'{method}_terrain{terrain_seed}'
    folder=OUT/('smoke' if smoke else 'rollouts')/name
    prepare_output(folder)
    slots=c.world_slots()[:1] if smoke else c.world_slots();batch=len(slots);limit=10 if smoke else 1000
    scenarios=('steady',) if smoke else c.SCENARIOS
    entries=[policy_entry(method,s['train_seed']) for s in slots]
    options=ort.SessionOptions();options.intra_op_num_threads=options.inter_op_num_threads=1
    cache={};sessions=[];names=[]
    for entry in entries:
        p=entry['policy']
        if p not in cache:cache[p]=ort.InferenceSession(p,sess_options=options,providers=['CPUExecutionProvider'])
        session=cache[p]
        if session.get_inputs()[0].shape!=[1,98]:raise ValueError('policy actor shape mismatch')
        sessions.append(session);names.append(session.get_inputs()[0].name)
    cfg=make_cfg(study,method,terrain_seed,batch)
    dump_yaml(folder/'env.yaml',asdict(cfg))
    record=dict(status='RUNNING',started_at=a.now(),method=method,terrain_seed=terrain_seed,smoke=smoke,
        policies=entries,slots=slots,scenarios=scenarios,common_cfg_sha256=study.common_cfg_digest(cfg),
        protocol_sha256=a.sha(OUT/'prospective_protocol.json'),encoder_sha256=a.ENCODER_SHA,
        transitions=0,simulator_world_steps=0,episode_count=0,PPO_updates=0,Adam_steps=0)
    a.write(folder/'manifest.json',record)
    env=None;episodes=[]
    try:
        seed_all(torch,20264200)
        env=ManagerBasedRlEnv(cfg=cfg,device='cuda:0')
        if abs(float(env.step_dt)-.02)>1e-9:raise ValueError('50Hz required')
        model=env.sim.mj_model
        foot_ids={i for i in range(model.ngeom) if any((model.geom(i).name or '').split('/')[-1]==f'{side}_foot{k}_collision' for side in ('left','right') for k in range(1,8))}
        if len(foot_ids)!=14:raise ValueError('missing native 14 foot collision geometries')
        # Entity indexing uses compiled model IDs, including borders and hfields.
        terrain_ids=set(int(x) for x in env.scene.terrain.indexing.geom_ids.tolist())
        hfield_ids=np.array([g for g in terrain_ids if model.geom_type[g]==mujoco.mjtGeom.mjGEOM_HFIELD],dtype=int)
        if terrain_seed and len(hfield_ids)!=5:raise ValueError('exactly five generated hfields required')
        robot=env.scene['robot'];sensor=env.scene['feet_ground_contact']
        site_ids,_=robot.find_sites(('left_foot','right_foot'),preserve_order=True)
        if len(site_ids)!=2:raise ValueError('missing paired foot sites')
        record.update(foot_geom_ids=sorted(foot_ids),terrain_geom_ids=sorted(terrain_ids),hfield_geom_ids=hfield_ids.tolist())
        for scenario in scenarios:
            for i,s in enumerate(slots):
                seed_all(torch,s['eval_seed']);env.reset(seed=s['eval_seed'],env_ids=torch.tensor([i],device=env.device))
            origins=env.scene.env_origins.cpu().numpy().copy()
            if terrain_seed:
                expected=env.scene.terrain.terrain_origins[0,torch.arange(batch,device=env.device)%5].cpu().numpy()
                np.testing.assert_array_equal(origins,expected)
            assigned,bounds=assigned_patches(model.geom_pos,hfield_ids,origins,bool(terrain_seed))
            physical=physical_snapshot(env);physical_hash=c.array_digest(physical)
            physical_path=folder/f'{scenario}_physics.npz';np.savez_compressed(physical_path,**physical)
            initial={k:v.detach().cpu().numpy().copy() for k,v in (
                ('position',robot.data.root_link_pos_w),('quaternion',robot.data.root_link_quat_w),
                ('joint_pos',robot.data.joint_pos),('joint_vel',robot.data.joint_vel),('root_velocity',robot.data.root_link_vel_w))}
            pairing=dict(common_cfg_sha256=record['common_cfg_sha256'],physical_sha256=physical_hash,
                initial={k:v.tolist() for k,v in initial.items()},origins=origins.tolist(),slots=slots,
                command=[.5,0.,0.],assigned=[sorted(x) for x in assigned],bounds=bounds.tolist())
            # Pair before this method's first step; later methods fail closed on any mismatch.
            pairpath=OUT/('smoke_pairing' if smoke else 'pairing')/f'terrain{terrain_seed}_{scenario}.json'
            pairing_guard(pairpath,pairing)
            a.write(folder/f'{scenario}_initial.json',pairing)
            record[scenario+'_physics']=dict(path=physical_path.name,sha256=a.sha(physical_path),array_sha256=physical_hash)
            keys=('time','command','position','quaternion','velocity','gravity','joint_pos','joint_vel','angle','contact','actions','observation','foot_sites','foot_contact_total','foot_contact_interior','nacon','terminated','timed_out')
            raw=[{k:[] for k in keys} for _ in slots]
            flattened=[{k:[] for k in ('geom','pos','dist','includemargin','efc_address','interior')} for _ in slots]
            offsets=[[0] for _ in slots]
            done=np.zeros(batch,bool);fell=np.zeros(batch,bool)
            for step in range(limit):
                env.command_manager.get_command('twist')[:]=torch.tensor([.5,0.,0.],device=env.device)
                if scenario=='push' and step in (250,500):
                    ids=torch.as_tensor(np.flatnonzero(~done),device=env.device)
                    velocity=robot.data.root_link_vel_w[ids].clone();velocity[:,1]+=.3 if step==250 else -.3
                    robot.write_root_link_velocity_to_sim(velocity,env_ids=ids);env.sim.forward();env.sim.sense()
                env.observation_manager._obs_buffer=None
                obs=env.observation_manager.compute()['actor'].cpu().numpy().astype(np.float32)
                phase=(learned_phase(env,str(a.ENCODER)) if method.endswith('E') else observations.phase(env,.6,'twist')).cpu().numpy().copy()
                np.testing.assert_allclose(obs[:,9:11],phase,atol=1e-7)
                np.testing.assert_allclose(np.linalg.norm(phase,axis=1),1.,atol=2e-6)
                np.testing.assert_allclose(obs[:,6:9],np.tile([.5,0.,0.],(batch,1)),atol=1e-7)
                action=c.route_actions(obs,sessions,names)
                exposure,points,nacon=read_contacts(env,foot_ids,terrain_ids,assigned,bounds)
                values={k:v.detach().cpu().numpy().copy() for k,v in (
                    ('position',robot.data.root_link_pos_w),('quaternion',robot.data.root_link_quat_w),
                    ('velocity',robot.data.root_link_lin_vel_b),('gravity',robot.data.projected_gravity_b),
                    ('joint_pos',observations._joint_positions_in_order(robot,robot.data.joint_pos,G1_JOINT_ORDER)),
                    ('joint_vel',observations._joint_positions_in_order(robot,robot.data.joint_vel,G1_JOINT_ORDER)),
                    ('contact',sensor.data.current_contact_time>0),('foot_sites',robot.data.site_pos_w[:,site_ids]))}
                values.update(angle=np.arctan2(phase[:,0],phase[:,1]),actions=action.copy(),observation=obs.copy(),
                    foot_contact_total=exposure['total'],foot_contact_interior=exposure['interior'])
                if not all(np.isfinite(v).all() for v in values.values()):raise ValueError('nonfinite raw state')
                for i in np.flatnonzero(~done):
                    raw[i]['time'].append(step*.02);raw[i]['command'].append(.5);raw[i]['nacon'].append(nacon)
                    for k,v in values.items():raw[i][k].append(v[i].copy())
                    chosen=np.flatnonzero(exposure['world']==i)
                    for k in ('geom','pos','dist','includemargin','efc_address'):flattened[i][k].append(points[k][chosen].copy())
                    flattened[i]['interior'].append(exposure['interior_mask'][chosen].copy());offsets[i].append(offsets[i][-1]+len(chosen))
                active=int((~done).sum());action[done]=0
                result=env.step(torch.as_tensor(action,device=env.device))
                record['transitions']+=active;record['simulator_world_steps']+=batch
                terminated=result[2].cpu().numpy().astype(bool);timed=result[3].cpu().numpy().astype(bool)
                for i in np.flatnonzero(~done):
                    raw[i]['terminated'].append(bool(terminated[i]));raw[i]['timed_out'].append(bool(timed[i]))
                if np.any(timed&~done):raise ValueError('unexpected timeout before frozen horizon')
                fell|=terminated&~done;done|=terminated
                if step%50==49 or step==limit-1 or done.all():
                    a.write(folder/'progress.json',dict(at=a.now(),scenario=scenario,step=step+1,active_worlds=int((~done).sum()),
                        effective_transitions=record['transitions'],simulator_world_steps=record['simulator_world_steps'],PPO_updates=0))
                    print(json.dumps(dict(method=method,terrain_seed=terrain_seed,scenario=scenario,step=step+1,active=int((~done).sum()))),flush=True)
                if done.all():break
            for i,slot in enumerate(slots):
                value={k:np.asarray(v) for k,v in raw[i].items()}
                value['contact_offsets']=np.asarray(offsets[i],dtype=np.int64)
                for k,v in flattened[i].items():value['points_'+k]=np.concatenate(v,axis=0)
                path=folder/f'{scenario}_train{slot["train_seed"]}_eval{slot["eval_seed"]}.npz'
                np.savez_compressed(path,**value)
                if smoke:
                    score=dict(observed_steps=len(value['time']),fell=bool(fell[i]),foot_contact_total=int(np.sum(value['foot_contact_total'])),
                        foot_contact_interior=int(np.sum(value['foot_contact_interior'])))
                else:score=c.episode_metrics(value,20.,scenario,bool(fell[i]),bool(terrain_seed))
                score.update(method=method,terrain_seed=terrain_seed,scenario=scenario,**slot,raw_path=path.name,raw_sha256=a.sha(path))
                episodes.append(score)
            a.write(folder/'episodes.json',episodes)
        record.update(status='COMPLETED',ended_at=a.now(),episode_count=len(episodes))
    except BaseException as error:
        record.update(status='FAILED',ended_at=a.now(),error=repr(error),traceback=traceback.format_exc())
        raise
    finally:
        a.write(folder/'manifest.json',record)
        if env is not None:env.close()


def child_arguments(command,*args):
    return [sys.executable,'-X','utf8','-c',PREFIX+"sys.argv=['g1_rough_transfer_experiment_20261028',"+','.join(repr(x) for x in (command,*args))+"];runpy.run_module('scripts.g1_rough_transfer_experiment_20261028',run_name='__main__')"]


def run_child(command,args,log):
    env=os.environ.copy();env.update(PYTHONHASHSEED='20264200',PYTHONUTF8='1',PYTHONIOENCODING='utf-8',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    with log.open('x',encoding='utf-8') as stream:
        result=subprocess.run(child_arguments(command,*args),cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    if result.returncode:raise RuntimeError('fresh rollout child failed; preserve directory and stop: '+str(log))


def smoke_queue():
    lock()
    for method,terrain in (('AC',0),('AE',0),('AC',20264201),('AE',20264201)):
        run_child('rollout',['--method',method,'--terrain-seed',str(terrain),'--smoke'],OUT/f'smoke_{method}_{terrain}.log')
    manifests=[a.read(p) for p in sorted((OUT/'smoke').glob('*/manifest.json'))]
    if len(manifests)!=4 or any(m['status']!='COMPLETED' or m['transitions']!=10 for m in manifests):raise ValueError('incomplete smoke accounting')
    for p in (OUT/'smoke').glob('*/episodes.json'):
        row=a.read(p)[0]
        if a.sha(p.parent/row['raw_path'])!=row['raw_sha256']:raise ValueError('smoke raw hash mismatch')
        with np.load(p.parent/row['raw_path'],allow_pickle=False) as saved:raw=dict(saved)
        if c.termination_from_raw(raw):raise ValueError('smoke terminated before full engineering horizon')
        m=a.read(p.parent/'manifest.json');initial=a.read(p.parent/'steady_initial.json')
        points=len(raw['points_pos'])
        exposure=c.contact_exposure(raw['points_geom'],np.zeros(points,int),raw['points_pos'],raw['points_dist'],raw['points_includemargin'],raw['points_efc_address'],
            set(m['foot_geom_ids']),set(m['terrain_geom_ids']),[set(x) for x in initial['assigned']],np.asarray(initial['bounds']),1)
        if len(exposure['indices'])!=points or points!=int(np.sum(raw['foot_contact_total'])):raise ValueError('smoke contact raw invalid')
        if not np.array_equal(exposure['interior_mask'],raw['points_interior']):raise ValueError('smoke assigned-patch reconstruction mismatch')
        if m['terrain_seed'] and (points==0 or not exposure['interior_mask'].any()):raise ValueError('rough smoke lacks actual assigned-hfield foot contact')
    a.write(OUT/'smoke_audit.json',dict(status='PASS',at=a.now(),protocol_sha256=a.sha(OUT/'prospective_protocol.json'),
        actual_eval_transitions=40,PPO_updates=0,Adam_steps=0,paired_before_sampling=True,
        scope='four fresh children, one world ten steps; engineering only, no efficacy selection'))


def preflight():
    study,torch=a.setup()
    configs=[]
    for t in c.TERRAINS:
        cfgs=[make_cfg(study,m,t) for m in c.METHODS]
        digests=[study.common_cfg_digest(cfg) for cfg in cfgs]
        if len(set(digests))!=1:raise ValueError('cross-method common configuration mismatch')
        configs.append(dict(terrain_seed=t,common_cfg_sha256=digests[0],actor_terms=list(cfgs[0].observations['actor'].terms),
            terrain=study.previous.canonical(asdict(cfgs[0].scene.terrain))))
    policies=[policy_entry(m,s) for m in c.METHODS for s in c.TRAIN_SEEDS]
    a.write(OUT/'cpu_preflight.json',dict(status='PASS',at=a.now(),configs=configs,policies=policies,
        effective_eval_transitions=0,PPO_updates=0,environment_created=False,
        experiment_sha256=a.sha(__file__),contract_sha256=a.sha(c.__file__)))


def queue():
    lock()
    smoke=a.read(OUT/'smoke_audit.json')
    if smoke['status']!='PASS' or smoke['protocol_sha256']!=a.sha(OUT/'prospective_protocol.json'):raise ValueError('matching smoke required')
    # Exclusive sentinel blocks duplicate launch and incomplete recovery, even after a crash.
    with (OUT/'queue_started.json').open('x',encoding='utf-8') as stream:json.dump(dict(at=a.now(),pid=os.getpid()),stream)
    status=dict(status='RUNNING',at=a.now(),pid=os.getpid(),completed_jobs=0,total_jobs=16,PPO_updates=0)
    try:
        for j in c.build_jobs():
            status.update(active=j,at=a.now());a.write(OUT/'queue_status.json',status)
            run_child('rollout',['--method',j['method'],'--terrain-seed',str(j['terrain_seed'])],OUT/f'job_{j["method"]}_{j["terrain_seed"]}.log')
            status['completed_jobs']+=1
        analyze()
        status.update(status='COMPLETED',ended_at=a.now(),analysis_sha256=a.sha(OUT/'analysis.json'))
    except BaseException as error:
        status.update(status='FAILED',at=a.now(),error=repr(error),traceback=traceback.format_exc());raise
    finally:a.write(OUT/'queue_status.json',status)


def analyze():
    lock();rows=[];checks=[]
    for j in c.build_jobs():
        folder=OUT/'rollouts'/f'{j["method"]}_terrain{j["terrain_seed"]}'
        m=a.read(folder/'manifest.json');episodes=a.read(folder/'episodes.json')
        if m['status']!='COMPLETED' or m['episode_count']!=30 or len(episodes)!=30:raise ValueError('incomplete R job')
        if m['protocol_sha256']!=a.sha(OUT/'prospective_protocol.json'):raise ValueError('protocol mismatch')
        for scenario in c.SCENARIOS:
            initial=a.read(folder/f'{scenario}_initial.json')
            if initial!=a.read(OUT/'pairing'/f'terrain{j["terrain_seed"]}_{scenario}.json'):raise ValueError('initial state pairing changed')
            physics=m[scenario+'_physics'];p=folder/physics['path']
            if a.sha(p)!=physics['sha256']:raise ValueError('physical raw changed')
            with np.load(p,allow_pickle=False) as saved:
                if c.array_digest(dict(saved))!=physics['array_sha256']:raise ValueError('physical arrays changed')
        for row in episodes:
            p=folder/row['raw_path']
            if a.sha(p)!=row['raw_sha256']:raise ValueError('episode raw changed')
            with np.load(p,allow_pickle=False) as saved:raw=dict(saved)
            native_fell=c.termination_from_raw(raw)
            if native_fell!=row['fell']:raise ValueError('native raw termination disagrees with fall endpoint')
            score=c.episode_metrics(raw,20.,row['scenario'],native_fell,bool(j['terrain_seed']))
            for k in ('completed','fell',*c.GUARDS,'rough_contact_fraction','foot_contact_total','foot_contact_interior'):
                if score[k]!=row[k]:raise ValueError('raw metric binding mismatch: '+k)
            # Recompute contact validity and interior exposure from the saved physical points.
            slot=next(i for i,s in enumerate(c.world_slots()) if s['train_seed']==row['train_seed'] and s['eval_seed']==row['eval_seed'])
            points=len(raw['points_pos'])
            exposed=c.contact_exposure(raw['points_geom'],np.full(points,slot),raw['points_pos'],raw['points_dist'],raw['points_includemargin'],raw['points_efc_address'],
                set(m['foot_geom_ids']),set(m['terrain_geom_ids']),[set(x) for x in initial['assigned']],np.asarray(initial['bounds']),15)
            if len(exposed['indices'])!=points or not np.array_equal(exposed['interior_mask'],raw['points_interior']):raise ValueError('contact raw assignment changed')
            if raw['contact_offsets'][-1]!=points or not np.array_equal(np.diff(raw['contact_offsets']),raw['foot_contact_total']):raise ValueError('contact offsets/counts changed')
            cumulative=np.r_[0,np.cumsum(raw['points_interior'].astype(int))]
            if not np.array_equal(np.diff(cumulative[raw['contact_offsets']]),raw['foot_contact_interior']):raise ValueError('interior contact counts changed')
            rows.append(row);checks.append(dict(path=str(p),sha256=row['raw_sha256']))
        if sum(r['observed_steps'] for r in episodes)!=m['transitions']:raise ValueError('effective sampling budget mismatch')
    result=c.paired_analysis(rows)
    value=dict(status='COMPLETED',at=a.now(),protocol_sha256=a.sha(OUT/'prospective_protocol.json'),
        raw_count=len(rows),raw_checks=checks,episodes=rows,statistics=result,PPO_updates=0,Adam_steps=0,
        effective_eval_transitions=sum(r['observed_steps'] for r in rows),
        limitation='exploratory transfer of flat-trained frozen policies; no fresh-seed or causal robustness confirmation')
    a.write(OUT/'analysis.json',value)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('preflight','lock','smoke','rollout','queue','analyze'))
    parser.add_argument('--method',choices=c.METHODS);parser.add_argument('--terrain-seed',type=int,choices=c.TERRAINS,default=0);parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    if args.command=='rollout':rollout(args.method,args.terrain_seed,args.smoke)
    else:globals()[{'smoke':'smoke_queue'}.get(args.command,args.command)]()


if __name__=='__main__':main()
