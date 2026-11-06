"""Prospective finite S: training domain x phase, full D501 warm start.
No environment creation or sampling on import. Frozen A/D/R are read-only.
"""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import random
import shutil
import subprocess
import sys
import traceback
import numpy as np
from scripts import g1_phase_lr_factorial_20261027 as a
from scripts import g1_rough_domain_contract_20261028 as c
from scripts import g1_rough_transfer_experiment_20261028 as r

ROOT=a.ROOT
OUT=ROOT/'results/g1_rough_domain_adaptation_20261028'
DOUT=ROOT/'results/g1_source_anchor_20261027'
ROUT=ROOT/'results/g1_rough_transfer_20261028'
SPEC=ROOT/'docs/superpowers/specs/2026-10-28-g1-rough-domain-adaptation-design.md'
PLAN=ROOT/'docs/superpowers/plans/2026-10-28-g1-rough-domain-adaptation.md'
PREFIX=r.PREFIX
ANCHOR_CLASS='scripts.g1_source_anchor_ppo_20261027:SourceAnchoredPPO'
STATE_KEYS=('actor_state_dict','critic_state_dict','optimizer_state_dict','source_teacher_state_dict')


def state_digest(value):
    """Type/shape/key aware recursive state hash, including normalizer buffers."""
    h=hashlib.sha256()
    def add(x):
        if hasattr(x,'detach'):
            x=x.detach().cpu().contiguous().numpy()
        if isinstance(x,np.ndarray):
            h.update(str(x.dtype).encode());h.update(str(x.shape).encode());h.update(x.tobytes())
        elif isinstance(x,dict):
            h.update(b'dict')
            for k in sorted(x,key=lambda k:(type(k).__name__,str(k))):add(k);add(x[k])
        elif isinstance(x,(tuple,list)):
            h.update(type(x).__name__.encode());h.update(str(len(x)).encode())
            for v in x:add(v)
        else:h.update((type(x).__name__+':'+repr(x)).encode())
    add(value)
    return h.hexdigest()


def verify_files(entries):
    for f in entries:
        if a.sha(f['path'])!=f['sha256']:raise ValueError('frozen source changed: '+f['path'])


def source_entry(method,seed):
    phase=c.source_phase(method);c.training_map(method,seed)
    folder=DOUT/'training'/f'{phase}_low_{seed}'
    m=a.read(folder/'manifest.json')
    if m['status']!='COMPLETED' or m['updates']!=501 or m['transitions']!=769536:raise ValueError('incomplete D501 source')
    p=m['checkpoints']['501']
    if a.sha(p['path'])!=p['sha256'] or a.sha(p['policy'])!=p['policy_sha256']:raise ValueError('D501 source changed')
    return dict(method=method,source_seed=seed,phase=phase,checkpoint=p['path'],checkpoint_sha256=p['sha256'],
        policy=p['policy'],policy_sha256=p['policy_sha256'],manifest=str(folder/'manifest.json'),manifest_sha256=a.sha(folder/'manifest.json'))


def validate_source_state(saved):
    required=set(STATE_KEYS)|{'infos','iter','source_anchor_coefficient'}
    if not required.issubset(saved):raise ValueError('incomplete D501 actor/critic/Adam/teacher state')
    st=saved['infos'].get('factorial_state',{})
    if saved['iter']!=500 or st.get('completed_updates')!=501 or saved['source_anchor_coefficient']!=.1:raise ValueError('not locked D501 beta.1 state')
    if not {'python_rng','numpy_rng','torch_rng','cuda_rng','learning_rate'}.issubset(st) or not st['cuda_rng']:raise ValueError('incomplete source RNG/LR')
    lr=st['learning_rate'];opt=saved['optimizer_state_dict']
    if not math.isfinite(lr) or lr<=0 or not opt['param_groups'] or any(g['lr']!=lr for g in opt['param_groups']):raise ValueError('source optimizer/algorithm LR mismatch')
    if len(opt['state'])!=17 or any(float(v['step'])!=10020 for v in opt['state'].values()):raise ValueError('source native Adam budget mismatch')
    return st


def rng_state(torch,include_cuda=True):
    value=dict(python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state())
    if include_cuda:value['cuda_rng']=torch.cuda.get_rng_state_all()
    return value


def restore_source(runner,checkpoint,torch,*,restore_cuda=True):
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    st=validate_source_state(saved)
    infos=runner.load(str(checkpoint),load_cfg=None,strict=True,map_location=runner.alg.device)
    if state_digest(infos)!=state_digest(saved['infos']):raise ValueError('native load infos changed')
    runner.alg.learning_rate=float(st['learning_rate'])
    if any(g['lr']!=runner.alg.learning_rate for g in runner.alg.optimizer.param_groups):raise ValueError('restored Adam LR mismatch')
    actual=runner.alg.save()
    hashes={k:state_digest(saved[k]) for k in STATE_KEYS}
    if any(state_digest(actual[k])!=hashes[k] for k in STATE_KEYS):raise ValueError('native full state restore mismatch')
    if runner.current_learning_iteration!=500:raise ValueError('native iteration not loaded')
    runner.current_learning_iteration=0
    # mjlab parent.load also restores env_state.common_step_counter. Reset only
    # after that native side effect; this is a new simulation with relative time.
    if hasattr(runner,'env'):runner.env.unwrapped.common_step_counter=0
    random.setstate(st['python_rng']);np.random.set_state(st['numpy_rng']);torch.set_rng_state(st['torch_rng'])
    if restore_cuda:
        if torch.cuda.device_count()!=len(st['cuda_rng']):raise ValueError('source CUDA RNG device count changed')
        torch.cuda.set_rng_state_all(st['cuda_rng'])
    fields=('python_rng','numpy_rng','torch_rng','cuda_rng') if restore_cuda else ('python_rng','numpy_rng','torch_rng')
    expected={k:st[k] for k in fields};observed=rng_state(torch,restore_cuda)
    if state_digest(observed)!=state_digest(expected):raise ValueError('source RNG restore mismatch')
    return dict(source_native_iteration=500,source_completed_updates=501,relative_iteration=0,
        state_hashes=hashes,learning_rate=runner.alg.learning_rate,RNG_sha256=state_digest(expected),
        source_RNG_sha256=state_digest({k:st[k] for k in ('python_rng','numpy_rng','torch_rng','cuda_rng')}),
        restored_cuda=restore_cuda,source_Adam_steps=10020)


def named_array_digest(arrays):
    h=hashlib.sha256()
    for key,value in sorted(arrays.items()):
        value=np.ascontiguousarray(value)
        if value.dtype.kind not in ('U','S') and not np.isfinite(value).all():raise ValueError('nonfinite named robot physics')
        h.update(key.encode());h.update(str(value.dtype).encode());h.update(str(value.shape).encode());h.update(value.tobytes())
    return h.hexdigest()


def robot_physics(raw):
    """Normalize all robot bodies/geoms/DOFs, including the floating base."""
    model=raw.sim.mj_model;robot=raw.scene['robot'];out={}
    for kind,fields in (('body',('body_mass','body_ipos','body_inertia')),('geom',('geom_friction',)),('dof',('dof_damping','dof_armature'))):
        if kind=='dof':
            ids=robot.indexing.free_joint_v_adr.tolist()+robot.indexing.joint_v_adr.tolist()
            names=[str(model.joint(int(model.dof_jntid[i])).name)+'['+str(int(i-model.jnt_dofadr[model.dof_jntid[i]]))+']' for i in ids]
        else:
            ids=getattr(robot.indexing,kind+'_ids').tolist()
            names=[];body_ordinals={}
            for i in ids:
                name=getattr(model,kind)(int(i)).name
                if kind=='geom':
                    body_name=model.body(int(model.geom_bodyid[i])).name
                    if not body_name:raise ValueError('robot geom parent body has no stable name')
                    ordinal=body_ordinals.get(body_name,0);body_ordinals[body_name]=ordinal+1
                    name=('named:'+name) if name else ('body:'+body_name+'/unnamed_geom['+str(ordinal)+']')
                names.append(name)
        if any(not name for name in names) or len(names)!=len(set(names)):raise ValueError('robot physical names missing/duplicated')
        order=np.argsort(names);ids=np.asarray(ids,dtype=int)[order];names=np.asarray(names)[order]
        out[kind+'_names']=names
        for field in fields:
            v=getattr(raw.sim.model,field).detach().cpu().numpy()
            out[field]=v[...,ids,:].copy() if field in ('body_ipos','body_inertia','geom_friction') else v[...,ids].copy()
    return out


def terrain_metadata(raw):
    import mujoco
    model=raw.sim.mj_model
    feet={i for i in range(model.ngeom) if any((model.geom(i).name or '').split('/')[-1]==f'{s}_foot{k}_collision' for s in ('left','right') for k in range(1,8))}
    if len(feet)!=14:raise ValueError('missing native foot geoms')
    terrain=set(int(x) for x in raw.scene.terrain.indexing.geom_ids.tolist())
    hfield=np.asarray([i for i in terrain if model.geom_type[i]==mujoco.mjtGeom.mjGEOM_HFIELD],dtype=int)
    origins=raw.scene.env_origins.cpu().numpy()
    assigned,bounds=r.assigned_patches(model.geom_pos,hfield,origins,bool(len(hfield)))
    return feet,terrain,hfield,assigned,bounds


def check_disk():
    if shutil.disk_usage(ROOT).free<10*1024**3:raise ValueError('disk below prospective 10GiB floor')


def aligned_environment_type(base,torch,seed):
    """Isolate terrain construction draws before all manager/timer/startup draws."""
    class AlignedEnvironment(base):
        def load_managers(self):
            r.seed_all(torch,seed)
            return super().load_managers()
    return AlignedEnvironment


def train(method,seed,smoke=False):
    verify_protocol();study,torch=a.setup()
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner,RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_rl_cfg,load_runner_cls
    from mjlab.utils.os import dump_yaml
    from src.tasks.velocity.mdp import observations
    from scripts import diagnose_g1_initial_update_20261026 as probe
    source=source_entry(method,seed);folder=OUT/('smoke_training' if smoke else 'training')/f'{method}_{seed}'
    folder.mkdir(parents=True,exist_ok=False)
    target=2 if smoke else c.UPDATES;points=(0,1,2) if smoke else c.CHECKPOINTS
    record=dict(status='RUNNING',started_at=a.now(),method=method,source_seed=seed,source=source,smoke=smoke,
        training_map=c.training_map(method,seed),updates=0,transitions=0,planned_updates=target,planned_transitions=target*1536,
        protocol_sha256=a.sha(OUT/'prospective_protocol.json'),encoder_sha256=a.ENCODER_SHA,checkpoints={},
        source_simulation_resume=False,learn_calls=0,native_minibatches_per_update=20)
    a.write(folder/'manifest.json',record);env=None
    try:
        check_disk()
        r.seed_all(torch,seed)
        cfg=c.make_training_cfg(study,method,seed)
        record['common_cfg_sha256']=study.common_cfg_digest(cfg)
        agent=load_rl_cfg('Unitree-G1-Flat');agent.seed=seed;agent.logger='tensorboard';agent.upload_model=False;agent.save_interval=10**9
        agent.algorithm.class_name=ANCHOR_CLASS;agent.algorithm.learning_rate=1e-5
        agent_dict=asdict(agent)
        if agent.num_steps_per_env!=24:raise ValueError('native rollout length changed')
        record['common_agent_sha256']=hashlib.sha256(json.dumps(study.canonical(agent_dict),sort_keys=True).encode()).hexdigest()
        dump_yaml(folder/'env.yaml',asdict(cfg));dump_yaml(folder/'agent.yaml',agent_dict)
        Env=aligned_environment_type(ManagerBasedRlEnv,torch,seed)
        raw=Env(cfg=cfg,device='cuda:0');env=RslRlVecEnvWrapper(raw,clip_actions=agent.clip_actions)
        record['construction_rng_alignment']=dict(point='load_managers entry after scene/terrain construction',seed=seed,scope='Python/NumPy/CPU-CUDA; all four arms; native manager and startup rules unchanged')
        if raw.num_envs!=64 or abs(raw.step_dt-.02)>1e-9:raise ValueError('training world/dt mismatch')
        bundle=observations._load_phasor_encoder(raw,str(a.ENCODER));bundle.encoder.eval().requires_grad_(False)
        encoder_before=study.base.state_sha(bundle.encoder);record['encoder_state_sha256']=encoder_before
        cls=load_runner_cls('Unitree-G1-Flat') or MjlabOnPolicyRunner
        runner=cls(env,agent_dict,str(folder),device='cuda:0')
        raw.common_step_counter=0
        # Read/validate construction observations before restoring the source RNG.
        obs=env.get_observations()
        restored=restore_source(runner,source['checkpoint'],torch,restore_cuda=True)
        record['restoration']=restored
        if tuple(obs['actor'].shape)!=(64,98):raise ValueError('actor input shape changed')
        expected_critic=next(v.shape[1] for v in runner.alg.critic.state_dict().values() if v.ndim==2)
        if obs['critic'].shape[1]!=expected_critic:raise ValueError('critic input shape changed')
        if {id(p) for p in bundle.encoder.parameters()} & {id(p) for g in runner.alg.optimizer.param_groups for p in g['params']}:raise ValueError('encoder entered optimizer')
        teacher_before=state_digest(runner.alg.source_teacher.state_dict())
        physical=r.physical_snapshot(raw);record_map(record['training_map'],physical);named=robot_physics(raw)
        np.savez_compressed(folder/'startup_physics.npz',**physical)
        np.savez_compressed(folder/'robot_physics.npz',**named)
        record.update(physical_sha256=c.r.array_digest(physical),robot_physics_sha256=named_array_digest(named),
            initial_RNG_sha256=state_digest(rng_state(torch)),teacher_state_sha256=teacher_before)
        if record['initial_RNG_sha256']!=restored['RNG_sha256']:raise ValueError('post-restore pre-sampling RNG consumed unexpectedly')
        # Same source/phase across domains must restore identical full state and RNG.
        r.pairing_guard(OUT/('smoke_training_pairing' if smoke else 'training_pairing')/f'phase_{source["phase"]}_{seed}.json',restored)
        r.pairing_guard(OUT/('smoke_training_pairing' if smoke else 'training_pairing')/f'robot_{seed}.json',dict(robot_physics_sha256=record['robot_physics_sha256']))
        domain='rough' if record['training_map'] else 'flat'
        r.pairing_guard(OUT/('smoke_training_pairing' if smoke else 'training_pairing')/f'common_{domain}_{seed}.json',
            dict(common_cfg_sha256=record['common_cfg_sha256'],common_agent_sha256=record['common_agent_sha256'],physical_sha256=record['physical_sha256']))
        feet,terrain,hfields,assigned,bounds=terrain_metadata(raw)
        if len(hfields)!=(5 if record['training_map'] else 0):raise ValueError('training physical hfield count mismatch')
        record.update(foot_geom_ids=sorted(feet),terrain_geom_ids=sorted(terrain),hfield_geom_ids=hfields.tolist(),
            assigned=[sorted(x) for x in assigned],bounds=bounds.tolist(),exposure_boundary_samples=0,assigned_contact_total=0)
        a.write(folder/'pre_sampling_pairing.json',dict(status='PASS',at=a.now(),transitions=0,PPO_updates=0,**restored,
            robot_physics_sha256=record['robot_physics_sha256'],actor_shape=list(obs['actor'].shape),critic_shape=list(obs['critic'].shape)))
        native_step=env.step
        def checked_step(action):
            if not torch.isfinite(action).all():raise ValueError('nonfinite training action')
            command=raw.command_manager.get_command('twist')
            if not bool(((command[:,0]>=.3-1e-6)&(command[:,0]<=.8+1e-6)&(command[:,1:].abs().amax(dim=1)<1e-6)).all()):raise ValueError('training command changed')
            result=native_step(action);record['transitions']+=64
            if not all(torch.isfinite(v).all() for v in (result[0]['actor'],result[0]['critic'],result[1],raw.scene['robot'].data.root_link_pos_w)):raise ValueError('nonfinite training state/reward')
            data=raw.sim.wp_data
            if int(data.nacon.numpy()[0])>=data.contact.geom.shape[0] or int(data.nefc.numpy().max())>=1500:raise ValueError('training native contact/constraint capacity saturation')
            return result
        env.step=checked_step
        native_save=lambda path,infos=None:MjlabOnPolicyRunner.save(runner,path,infos)
        def save_checkpoint(_path=None,infos=None):
            completed=record['updates']
            if completed not in points or str(completed) in record['checkpoints']:return
            if study.base.state_sha(bundle.encoder)!=encoder_before or state_digest(runner.alg.source_teacher.state_dict())!=teacher_before:raise ValueError('frozen teacher/encoder changed')
            if any(float(v['step'])!=10020+20*completed for v in runner.alg.optimizer.state.values()):raise ValueError('native cumulative Adam budget mismatch')
            check_disk()
            path=folder/f'model_{completed}.pt'
            details=dict(completed_updates=completed,source_completed_updates=501,learning_rate=runner.alg.learning_rate,
                method=method,source_seed=seed,protocol_sha256=record['protocol_sha256'],**rng_state(torch))
            native_save(str(path),dict(infos or {},domain_state=details))
            saved=torch.load(path,map_location='cpu',weights_only=False)
            if completed==0 and any(state_digest(saved[k])!=restored['state_hashes'][k] for k in STATE_KEYS):raise ValueError('checkpoint0 no longer matches D501')
            record['checkpoints'][str(completed)]=dict(path=str(path),sha256=a.sha(path))
            a.write(folder/'manifest.json',record)
        runner.save=save_checkpoint
        native_update=runner.alg.update
        def update():
            completed=record['updates']+1
            check_disk()
            if record['transitions']!=completed*1536:raise ValueError('continuous native rollout budget mismatch')
            # Boundary snapshot, explicitly not all-frame exposure coverage.
            exposure,points_raw,nacon=r.read_contacts(raw,feet,terrain,assigned,bounds)
            np.savez_compressed(folder/f'exposure_{completed}.npz',**points_raw,
                interior=exposure['interior_mask'],total=exposure['total'],interior_count=exposure['interior'])
            record['assigned_contact_total']+=int(exposure['interior'].sum());record['exposure_boundary_samples']+=1
            before=runner.alg.learning_rate
            losses,trace=a.record_update(runner.alg,native_update)
            source_trace=runner.alg.last_source_kl
            if len(trace)!=20 or len(source_trace)!=20 or any(x['beta']!=.1 for x in source_trace):raise ValueError('native KL/minibatch/source trace mismatch')
            record['updates']=completed
            row=dict(completed_update=completed,at=a.now(),lr_before=before,lr_after=runner.alg.learning_rate,
                minibatches=trace,source_kl=source_trace,losses=losses,transitions=record['transitions'],
                exposure_sha256=a.sha(folder/f'exposure_{completed}.npz'))
            with (folder/'updates.jsonl').open('a',encoding='utf8') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n')
            a.write(folder/'progress.json',dict(at=a.now(),updates=completed,transitions=record['transitions'],lr=runner.alg.learning_rate,PPO_updates=completed,Adam_steps=20*completed))
            return losses
        runner.alg.update=update
        old_log=runner.logger.log
        def log(*args,**kwargs):
            result=old_log(*args,**kwargs);save_checkpoint();return result
        runner.logger.log=log
        save_checkpoint();a.write(folder/'manifest.json',record)
        record['learn_calls']+=1
        runner.learn(num_learning_iterations=target,init_at_random_ep_len=True)
        if record['updates']!=target or record['transitions']!=target*1536 or set(record['checkpoints'])!=set(map(str,points)):raise ValueError('incomplete native training budget/checkpoints')
        if record['training_map'] and record['assigned_contact_total']==0:raise ValueError('rough training never contacted assigned hfield at boundary samples')
        for point,entry in record['checkpoints'].items():
            saved=torch.load(entry['path'],map_location='cpu',weights_only=False);policy=folder/f'policy_{point}.onnx'
            entry.update(probe.export_actor(saved['actor_state_dict'],policy,torch),policy=str(policy))
        record.update(status='COMPLETED',ended_at=a.now(),encoder_frozen=True,teacher_frozen=True,
            PPO_updates=target,Adam_steps=target*20,cumulative_Adam_steps=10020+target*20,learn_calls=1)
    except BaseException as error:
        record.update(status='FAILED',ended_at=a.now(),error=repr(error),traceback=traceback.format_exc());raise
    finally:
        a.write(folder/'manifest.json',record)
        if env is not None:env.close()


def claim_queue(folder=OUT):
    with (folder/'queue_started.json').open('x',encoding='utf8') as stream:json.dump(dict(at=a.now(),pid=os.getpid()),stream)


def verify_protocol():
    p=OUT/'prospective_protocol.json';value=a.read(p)
    if value['status']!='LOCKED_BEFORE_S_SAMPLING':raise ValueError('S protocol not locked')
    verify_files(value['identity']['files'])
    return value


def record_map(terrain_seed,arrays):
    if not terrain_seed:return
    if terrain_seed not in (*c.TRAIN_MAPS,*c.TERRAINS[1:]):raise ValueError('undeclared S map')
    digest=c.r.array_digest({k:arrays[k] for k in ('hfield_data','hfield_size','hfield_nrow','hfield_ncol','hfield_adr')})
    path=OUT/'map_identities.json';known=a.read(path) if path.exists() else {}
    for key,value in known.items():
        if int(key)!=terrain_seed and value==digest:raise ValueError('train/eval map arrays duplicated')
        if int(key)==terrain_seed and value!=digest:raise ValueError('same declared map changed')
    known[str(terrain_seed)]=digest;a.write(path,known)


def verify_training(method,seed,smoke=False):
    import torch
    folder=OUT/('smoke_training' if smoke else 'training')/f'{method}_{seed}'
    m=a.read(folder/'manifest.json');target=2 if smoke else 251;points=(0,1,2) if smoke else c.CHECKPOINTS
    if m['status']!='COMPLETED' or m['updates']!=target or m['transitions']!=target*1536 or m['learn_calls']!=1:raise ValueError('incomplete S training')
    if m['protocol_sha256']!=a.sha(OUT/'prospective_protocol.json') or m['source']!=source_entry(method,seed):raise ValueError('S training source/protocol changed')
    if not m['encoder_frozen'] or not m['teacher_frozen'] or m['cumulative_Adam_steps']!=10020+20*target:raise ValueError('S state/budget gate failed')
    lines=[json.loads(line) for line in (folder/'updates.jsonl').read_text(encoding='utf8').splitlines()]
    if len(lines)!=target or [x['completed_update'] for x in lines]!=list(range(1,target+1)):raise ValueError('noncontinuous S updates')
    total=0
    for i,row in enumerate(lines,1):
        if row['transitions']!=i*1536 or len(row['minibatches'])!=20 or len(row['source_kl'])!=20:raise ValueError('native update trace mismatch')
        if any(x['beta']!=.1 for x in row['source_kl']):raise ValueError('beta changed')
        p=folder/f'exposure_{i}.npz'
        if a.sha(p)!=row['exposure_sha256']:raise ValueError('exposure boundary changed')
        with np.load(p,allow_pickle=False) as saved:total+=int(saved['interior_count'].sum())
    if m['exposure_boundary_samples']!=target or total!=m['assigned_contact_total'] or m['training_map'] and total==0:raise ValueError('rough training exposure invalid')
    if set(m['checkpoints'])!=set(map(str,points)):raise ValueError('missing fixed checkpoint')
    for point in points:
        entry=m['checkpoints'][str(point)]
        if a.sha(entry['path'])!=entry['sha256'] or a.sha(entry['policy'])!=entry['policy_sha256']:raise ValueError('S checkpoint/ONNX changed')
        saved=torch.load(entry['path'],map_location='cpu',weights_only=False)
        if saved['infos']['domain_state']['completed_updates']!=point or saved['source_anchor_coefficient']!=.1:raise ValueError('checkpoint state binding failed')
        if state_digest(saved['source_teacher_state_dict'])!=m['teacher_state_sha256']:raise ValueError('checkpoint teacher drift')
        if any(float(v['step'])!=10020+point*20 for v in saved['optimizer_state_dict']['state'].values()):raise ValueError('checkpoint Adam count failed')
        if point==0 and any(state_digest(saved[k])!=m['restoration']['state_hashes'][k] for k in STATE_KEYS):raise ValueError('initial full source mismatch')
    return m


def cpu_runner(saved,torch):
    """Real native algorithm/runner load, without any simulation or storage update."""
    from tensordict import TensorDict
    from rsl_rl.models import MLPModel
    from mjlab.rl import MjlabOnPolicyRunner
    from types import SimpleNamespace
    from scripts.g1_source_anchor_ppo_20261027 import SourceAnchoredPPO
    obs=TensorDict({name:torch.zeros(1,next(v.shape[1] for v in saved[key].values() if v.ndim==2)) for name,key in [('actor','actor_state_dict'),('critic','critic_state_dict')]},batch_size=[1])
    groups={'actor':['actor'],'critic':['critic']};kw=dict(hidden_dims=(512,256,128),activation='elu',obs_normalization=True)
    actor=MLPModel(obs,groups,'actor',29,**kw,distribution_cfg=dict(class_name='GaussianDistribution',init_std=1.,std_type='scalar'))
    critic=MLPModel(obs,groups,'critic',1,**kw)
    runner=object.__new__(MjlabOnPolicyRunner);runner.env=SimpleNamespace(unwrapped=SimpleNamespace(common_step_counter=-1));runner.alg=SourceAnchoredPPO(actor,critic,None,device='cpu',learning_rate=1e-5)
    runner.current_learning_iteration=-1
    return runner


def cpu_robot_gold(study,torch):
    """Compile the native robot spec only; no environment, GPU, or simulation."""
    from types import SimpleNamespace
    import mujoco
    cfg=c.make_training_cfg(study,'FC',c.TRAIN_SEEDS[0])
    spec=cfg.scene.entities['robot'].spec_fn();model=spec.compile()
    tensor=lambda field:torch.from_numpy(np.asarray(getattr(model,field)).copy())
    joint=[];free=[]
    for j in range(model.njnt):
        start=int(model.jnt_dofadr[j]);end=int(model.jnt_dofadr[j+1]) if j+1<model.njnt else model.nv
        (free if model.jnt_type[j]==mujoco.mjtJoint.mjJNT_FREE else joint).extend(range(start,end))
    indexing=SimpleNamespace(body_ids=torch.arange(1,model.nbody),geom_ids=torch.arange(model.ngeom),
        joint_v_adr=torch.tensor(joint),free_joint_v_adr=torch.tensor(free))
    runtime=SimpleNamespace(**{k:tensor(k) for k in ('body_mass','body_ipos','body_inertia','geom_friction','dof_damping','dof_armature')})
    raw=SimpleNamespace(sim=SimpleNamespace(mj_model=model,model=runtime),scene={'robot':SimpleNamespace(indexing=indexing)})
    values=robot_physics(raw)
    if len(values['geom_names'])!=model.ngeom or len(values['dof_names'])!=model.nv:raise ValueError('native CPU robot physics coverage incomplete')
    return dict(status='PASS',environment_created=False,environment_steps=0,PPO_updates=0,
        body_count=model.nbody-1,geom_count=model.ngeom,unnamed_native_geoms=sum(not model.geom(i).name for i in range(model.ngeom)),
        dof_count=model.nv,named_physical_sha256=named_array_digest(values),coverage='all native robot bodies/geoms/DOFs, including unnamed visuals and free root')


def preflight():
    study,torch=a.setup();configs=[];restorations=[];onnx=[]
    import onnxruntime as ort
    for seed in c.TRAIN_SEEDS:
        cfgs={m:c.make_training_cfg(study,m,seed) for m in c.METHODS}
        for left,right in (('FC','FE'),('RC','RE')):
            if study.common_cfg_digest(cfgs[left])!=study.common_cfg_digest(cfgs[right]):raise ValueError('same-domain phase common config differs')
        without=[]
        for m,cfg in cfgs.items():
            value=deepcopy(cfg);value.scene.terrain=deepcopy(cfgs['FC'].scene.terrain);without.append(study.common_cfg_digest(value))
            configs.append(dict(method=m,seed=seed,common_cfg_sha256=study.common_cfg_digest(cfg),
                actor_terms=list(cfg.observations['actor'].terms),critic_terms=list(cfg.observations['critic'].terms),
                events=list(cfg.events),config=study.canonical(asdict(cfg))))
        if len(set(without))!=1:raise ValueError('cross-domain nonterrain common cfg differs')
    options=ort.SessionOptions();options.intra_op_num_threads=options.inter_op_num_threads=1
    for phase,method in (('clock','FC'),('learned','FE')):
        for seed in c.TRAIN_SEEDS:
            entry=source_entry(method,seed);saved=torch.load(entry['checkpoint'],map_location='cpu',weights_only=False)
            runner=cpu_runner(saved,torch);restored=restore_source(runner,entry['checkpoint'],torch,restore_cuda=False)
            restorations.append(dict(phase=phase,seed=seed,**restored))
            policy=ort.InferenceSession(entry['policy'],sess_options=options,providers=['CPUExecutionProvider'])
            if policy.get_inputs()[0].shape!=[1,98]:raise ValueError('source actor shape changed')
            action=policy.run(None,{policy.get_inputs()[0].name:np.zeros((1,98),np.float32)})[0]
            if action.shape!=(1,29) or not np.isfinite(action).all():raise ValueError('source ONNX invalid')
            onnx.append(dict(phase=phase,seed=seed,action_shape=list(action.shape),policy_sha256=entry['policy_sha256']))
    own=[Path(__file__),Path(c.__file__),ROOT/'scripts/g1_rough_domain_evaluation_20261028.py',OUT/'independent_audit.py']
    a.write(OUT/'cpu_preflight.json',dict(status='PASS',at=a.now(),environment_created=False,environment_steps=0,PPO_updates=0,
        configs=configs,restorations=restorations,source_ONNX=onnx,native_robot_cpu_gold=cpu_robot_gold(study,torch),code=[dict(path=str(p),sha256=a.sha(p)) for p in own]))


def sources():
    own=[Path(__file__),Path(c.__file__),ROOT/'scripts/g1_rough_domain_evaluation_20261028.py',
        ROOT/'tests/test_g1_rough_domain_20261028.py',SPEC,PLAN]
    own += [OUT/n for n in ('PROJECT_PLAN.md','experiment_matrix.md','preregistration.md','reproducibility-checklist.md',
        'plan_spec.json','target-chain.json','target_chain_report.json','failure-tree.json','failure_tree_report.json',
        'plan_findings.json','plan_package.json','research_package_report.json','prepare_plan_package.py',
        'builder_check.md','implementation_review.md','implementation_review.json','cpu_preflight.json','source_state_audit.json','tests_final_cpu.log',
        'independent_audit.py','independent_audit_plan.json','technical_amendment_01.json',
        'technical_failure_01/prospective_protocol.json','technical_failure_01/failure_summary.json')]
    paths=set(own)
    for directory,field in ((DOUT,'files'),(ROUT,'files')):
        p=directory/'prospective_protocol.json';paths.update((p,directory/'analysis.json'))
        entries=a.read(p)['identity'][field];verify_files(entries);paths.update(Path(f['path']) for f in entries)
    for method in ('FC','FE'):
        for seed in c.TRAIN_SEEDS:
            p=source_entry(method,seed)
            paths.update(Path(p[k]) for k in ('checkpoint','policy','manifest'))
            paths.update(Path(p['manifest']).parent/n for n in ('env.yaml','agent.yaml','updates.jsonl','pre_sampling_pairing.json'))
    return [dict(path=str(p),sha256=a.sha(p)) for p in sorted(paths,key=str)]


def lock():
    pre=a.read(OUT/'cpu_preflight.json')
    if pre['status']!='PASS' or pre['environment_steps'] or pre['PPO_updates']:raise ValueError('CPU preflight required')
    verify_files(pre['code'])
    if a.read(OUT/'research_package_report.json')['verdict'] not in ('PASS','WARN'):raise ValueError('research package unresolved')
    review=a.read(OUT/'implementation_review.json')
    if review['status']!='PASS' or review['unresolved_critical_important']!=0:raise ValueError('fresh implementation review required')
    verify_files(review['code'])
    identity=dict(files=sources(),sources=[source_entry(m,s) for m in c.METHODS for s in c.TRAIN_SEEDS],
        train_jobs=c.training_jobs(),evaluation_jobs=c.evaluation_jobs(),slots=c.slots(),checkpoints=c.CHECKPOINTS,
        formal_train_transitions=4626432,formal_PPO_updates=3012,formal_Adam_steps=60240,formal_eval_episodes=480,
        maximum_effective_eval_transitions=480000,smoke_train_transitions=12288,smoke_PPO_updates=8,smoke_Adam_steps=160,
        smoke_max_eval_transitions=80,training_maps=c.TRAIN_MAPS,evaluation_maps=c.TERRAINS,
        beta=.1,encoder_sha256=a.ENCODER_SHA,source_completed_updates=501,relative_updates=251,
        warm_start='full actor/critic/Adam/normalizers/original teacher/LR/RNG; new simulation, not exact physics resume',
        statistical_unit='three reused matched source training seeds; exploratory only; exact p minimum .25',
        family='RE-RC;RE-FE;RC-FC;interaction x completion/fall; Holm8',rough_guards=c.GUARDS,flat_guards=c.FLAT_GUARDS,
        packages={n:importlib.metadata.version(n) for n in ('torch','mjlab','mujoco','mujoco-warp','warp-lang','rsl-rl-lib','onnxruntime','numpy','scipy')})
    identity=json.loads(json.dumps(identity));path=OUT/'prospective_protocol.json'
    if path.exists():
        old=a.read(path)
        if old['identity']!=identity:raise ValueError('S protocol/source changed')
        return old
    if (OUT/'queue_started.json').exists() or (OUT/'training').exists() or (OUT/'smoke_training').exists():raise ValueError('sampling exists before lock; preserve and stop')
    value=dict(status='LOCKED_BEFORE_S_SAMPLING',at=a.now(),identity=identity);a.write(path,value)
    for item in identity['files']:
        p=Path(item['path'])
        if p.suffix in ('.py','.md','.json','.yaml') and p.stat().st_size<5*1024**2:
            relative=p.relative_to(ROOT) if p.is_relative_to(ROOT) else Path('dependencies')/item['sha256'][:12]/p.name
            dst=OUT/'source_snapshot'/relative;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(p.read_bytes())
    return value


def child_arguments(stage,*args):
    return [sys.executable,'-X','utf8','-c',PREFIX+"sys.argv=['g1_rough_domain_experiment_20261028','--stage',"+','.join(repr(x) for x in (stage,*args))+"];runpy.run_module('scripts.g1_rough_domain_experiment_20261028',run_name='__main__')"]


def run_child(stage,args,log):
    env=os.environ.copy();env.update(PYTHONHASHSEED='20264200',PYTHONUTF8='1',PYTHONIOENCODING='utf-8',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    with log.open('x',encoding='utf8') as stream:
        result=subprocess.run(child_arguments(stage,*args),cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    if result.returncode:raise RuntimeError('fresh S child failed; preserve and stop: '+str(log))


def _smoke_queue():
    verify_protocol()
    with (OUT/'smoke_started.json').open('x',encoding='utf8') as f:json.dump(dict(at=a.now(),pid=os.getpid()),f)
    for method in c.METHODS:
        run_child('train',['--method',method,'--source-seed',str(c.TRAIN_SEEDS[0]),'--smoke'],OUT/f'smoke_train_{method}.log')
        verify_training(method,c.TRAIN_SEEDS[0],smoke=True)
    from scripts import g1_rough_domain_evaluation_20261028 as ev
    for method in c.METHODS:
        for terrain in (0,c.TERRAINS[1]):
            run_child('evaluate',['--method',method,'--terrain-seed',str(terrain),'--smoke'],OUT/f'smoke_eval_{method}_{terrain}.log')
    manifests=[a.read(p) for p in (OUT/'smoke').glob('*/manifest.json')]
    if len(manifests)!=8 or any(m['status']!='COMPLETED' or m['transitions']!=10 for m in manifests):raise ValueError('incomplete S smoke eval')
    for p in (OUT/'smoke').glob('*/episodes.json'):
        row=a.read(p)[0];m=a.read(p.parent/'manifest.json');rawpath=p.parent/row['raw_path']
        if a.sha(rawpath)!=row['raw_sha256']:raise ValueError('smoke raw SHA mismatch')
        with np.load(rawpath,allow_pickle=False) as data:raw=dict(data)
        if c.r.termination_from_raw(raw):raise ValueError('smoke ended before engineering horizon')
        if m['terrain_seed'] and int(raw['foot_contact_interior'].sum())==0:raise ValueError('rough smoke actual assigned hfield foot exposure absent')
    a.write(OUT/'smoke_audit.json',dict(status='PASS',at=a.now(),protocol_sha256=a.sha(OUT/'prospective_protocol.json'),
        train_transitions=12288,eval_transitions=80,PPO_updates=8,Adam_steps=160,scope='engineering smoke, no efficacy or selection'))


def smoke_queue():
    try:return _smoke_queue()
    finally:update_costs()


def update_costs():
    groups={}
    for name in ('smoke_training','training'):
        rows=[]
        for p in (OUT/name).glob('*/manifest.json'):
            m=a.read(p);progress=a.read(p.parent/'progress.json') if (p.parent/'progress.json').exists() else m
            rows.append(dict(path=str(p),status=m['status'],known_transitions=max(m.get('transitions',0),progress.get('transitions',0)),known_PPO_updates=max(m.get('updates',0),progress.get('updates',0)),known_Adam_steps_lower_bound=20*max(m.get('updates',0),progress.get('updates',0))))
        groups[name]=rows
    for name in ('smoke','rollouts'):
        rows=[]
        for p in (OUT/name).glob('*/manifest.json'):
            m=a.read(p);progress=a.read(p.parent/'progress.json') if (p.parent/'progress.json').exists() else {}
            rows.append(dict(path=str(p),status=m['status'],known_effective_eval=max(m.get('transitions',0),progress.get('effective_transitions',0)),known_simulator_world_steps=max(m.get('simulator_world_steps',0),progress.get('simulator_world_steps',0))))
        groups[name]=rows
    a.write(OUT/'technical_cost_ledger.json',dict(at=a.now(),status='LIVE_LOWER_BOUND_WHILE_RUNNING',groups=groups,
        preserved_S_failures=[a.read(p) for p in sorted(OUT.glob('technical_failure_*/failure_summary.json'))],
        note='completed valid budgets separate from smoke and failed attempts; abrupt loss may exceed last progress, preserve uncertainty',
        historical_D_failure_train_bounds=[531456,1542144],historical_D_failure_PPO_bounds=[346,1004],historical_C_failure_eval_bounds=[6750,9000]))


def queue():
    verify_protocol();smoke=a.read(OUT/'smoke_audit.json')
    if smoke['status']!='PASS' or smoke['protocol_sha256']!=a.sha(OUT/'prospective_protocol.json'):raise ValueError('matching S smoke required')
    claim_queue();status=dict(status='RUNNING',at=a.now(),pid=os.getpid(),completed_train_jobs=0,completed_eval_jobs=0,total_train_jobs=12,total_eval_jobs=16)
    try:
        for j in c.training_jobs():
            status.update(active=dict(stage='train',**j),at=a.now());a.write(OUT/'queue_status.json',status)
            run_child('train',['--method',j['method'],'--source-seed',str(j['source_seed'])],OUT/f'train_{j["method"]}_{j["source_seed"]}.log')
            verify_training(j['method'],j['source_seed']);status['completed_train_jobs']+=1;update_costs()
        for j in c.evaluation_jobs():
            status.update(active=dict(stage='evaluate',**j),at=a.now());a.write(OUT/'queue_status.json',status)
            run_child('evaluate',['--method',j['method'],'--terrain-seed',str(j['terrain_seed'])],OUT/f'eval_{j["method"]}_{j["terrain_seed"]}.log')
            status['completed_eval_jobs']+=1;update_costs()
        from scripts import g1_rough_domain_evaluation_20261028 as ev
        ev.analyze();status.update(status='COMPLETED_PENDING_INDEPENDENT_AUDIT',ended_at=a.now(),analysis_sha256=a.sha(OUT/'analysis.json'))
    except BaseException as error:
        status.update(status='FAILED',at=a.now(),error=repr(error),traceback=traceback.format_exc());raise
    finally:a.write(OUT/'queue_status.json',status);update_costs()


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',required=True,choices=('preflight','lock','smoke','train','evaluate','queue','analyze','costs'))
    p.add_argument('--method',choices=c.METHODS);p.add_argument('--source-seed',type=int,choices=c.TRAIN_SEEDS);p.add_argument('--terrain-seed',type=int,choices=c.TERRAINS);p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    if args.stage=='train':train(args.method,args.source_seed,args.smoke)
    elif args.stage in ('evaluate','analyze'):
        from scripts import g1_rough_domain_evaluation_20261028 as ev
        if args.stage=='evaluate':ev.evaluate(args.method,args.terrain_seed,args.smoke)
        else:ev.analyze()
    else:globals()[{'smoke':'smoke_queue','costs':'update_costs'}.get(args.stage,args.stage)]()

if __name__=='__main__':main()
