"""Paired fixed-horizon evaluation and CPU batch=1 deployment timing."""
from __future__ import annotations

import copy
import json
import platform
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from scripts import g1_semantic_split_study as study
from scripts.run_g1_semantic_split import prepare, read, run_directory, environment_record


def validate_coverage(rows, protocol, split):
    expected = {(s,c,k) for s in protocol['evaluation'][split+'_seeds']
                for c in range(len(protocol['evaluation']['commands']))
                for k in protocol['evaluation']['scenarios']}
    actual = [(r['episode_seed'],r['command_id'],r['scenario']) for r in rows]
    if len(actual)!=len(expected) or set(actual)!=expected:
        raise ValueError('evaluation episodes missing, duplicated or outside the frozen matrix')
    return len(expected)


def cpu_session(path):
    import onnxruntime as ort
    options=ort.SessionOptions()
    options.intra_op_num_threads=options.inter_op_num_threads=1
    options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])


def observation_noise(layout, scenario, protocol, rng, n):
    widths=np.zeros(98,np.float32)
    for name,value in protocol['evaluation']['noise_uniform_half_width'].items():
        field=layout['fields'][name]
        widths[field['start']:field['stop']]=value
    return rng.uniform(-1.,1.,size=(n,98)).astype(np.float32)*widths*scenario.get('state_noise_scale',0.)


def body_lateral_velocity_delta(quaternion_wxyz, magnitude):
    """Rotate body [0, magnitude, 0] to world coordinates (wxyz quaternion)."""
    q = quaternion_wxyz
    v = torch.zeros((len(q), 3), dtype=q.dtype, device=q.device)
    v[:, 1] = magnitude
    twice_cross = 2 * torch.linalg.cross(q[:, 1:], v)
    return v + q[:, :1] * twice_cross + torch.linalg.cross(q[:, 1:], twice_cross)


def verify_rough_pairing(reference, snapshot):
    """Check actual spawn state, including patch selection, across policies."""
    reference = Path(reference)
    if reference.exists():
        if read(reference) != snapshot:
            raise ValueError('rough evaluation pairing differs in actual terrain or spawn state')
    else:
        study.write(reference, snapshot)


def load_batch(target, seed, protocol, identity):
    """Recompute cached scores from identity-bound raw data before accepting them."""
    target = Path(target)
    batch = read(target/f'episode_seed{seed}.json')
    expected_identity = dict(identity, episode_seed=seed)
    if batch.get('identity') != expected_identity:
        raise ValueError('cached batch identity differs')
    path = target/f'episode_seed{seed}.npz'
    if batch['trajectory_file'] != path.name or study.sha(path) != batch['trajectory_sha256']:
        raise ValueError('cached raw trajectory changed')
    jobs = [(c, name) for c in range(len(protocol['evaluation']['commands']))
            for name in protocol['evaluation']['scenarios']]
    with np.load(path, allow_pickle=False) as data:
        if json.loads(str(data['identity_json'])) != expected_identity:
            raise ValueError('raw trajectory identity differs')
        dt = float(data['dt'])
        steps = int(round(protocol['evaluation']['horizon_seconds']/dt))
        n = len(jobs)
        if (data['commands'].shape != (steps, n, 3) or data['velocities'].shape != (steps, n, 3)
                or data['actions'].shape != (steps, n, 29) or data['terminated'].shape != (steps, n)
                or data['alive_steps'].shape != (n,) or not np.isfinite(data['actions']).all()):
            raise ValueError('cached raw trajectory shape or action invalid')
        rows = []
        for i, (command_id, scenario) in enumerate(jobs):
            command = np.asarray(protocol['evaluation']['commands'][command_id], np.float32)
            if not np.array_equal(data['commands'][:, i], np.broadcast_to(command, (steps, 3))):
                raise ValueError('raw command differs from protocol')
            failed = np.flatnonzero(data['terminated'][:, i])
            alive_steps = int(failed[0]+1) if len(failed) else steps
            if len(failed) > 1 or int(data['alive_steps'][i]) != alive_steps:
                raise ValueError('raw termination and alive_steps disagree')
            metrics = study.episode_metrics(data['commands'][:, i], data['velocities'][:, i],
                        alive_steps=alive_steps, fell=bool(len(failed)), dt=dt)
            rows.append(dict(episode_seed=seed, command_id=command_id, scenario=scenario,
                             command=command.tolist(), slot=i, **metrics))
    if batch['rows'] != rows or batch['dt'] != dt or batch['horizon_steps'] != steps:
        raise ValueError('cached metrics differ from raw trajectory recomputation')
    return batch


def verified_evaluation_rows(target, record, protocol, identity):
    target = Path(target)
    if record['identity'] != identity or record['status'] != 'COMPLETED':
        raise ValueError('evaluation identity or completion failed')
    rows = []
    entries = []
    for seed in protocol['evaluation'][identity['split']+'_seeds']:
        batch = load_batch(target, seed, protocol, identity)
        batch_path = target/f'episode_seed{seed}.json'
        entries.append(dict(file=batch['trajectory_file'], sha256=batch['trajectory_sha256'],
                            batch_file=batch_path.name, batch_sha256=study.sha(batch_path)))
        rows.extend(dict(row, arm=identity['arm'], training_seed=identity['training_seed']) for row in batch['rows'])
    if entries != record['raw_trajectories']:
        raise ValueError('raw trajectory or batch manifest changed')
    if study.sha(target/'episodes.json') != record['episodes_sha256'] or read(target/'episodes.json') != rows:
        raise ValueError('episode table differs from raw trajectory recomputation')
    validate_coverage(rows, protocol, identity['split'])
    return rows


def rollout_seed(raw, session, layout, protocol, episode_seed, device, target, identity):
    """One paired batch: command × scenario slots, reseeded identically per actor.

    Startup domain randomization uses a common environment-construction seed;
    reset randomization uses episode_seed. Slots match exactly across policies.
    Different scenarios are distinct slots, so their reset draws need not match.
    """
    jobs=[(c,name,settings) for c in range(len(protocol['evaluation']['commands']))
          for name,settings in protocol['evaluation']['scenarios'].items()]
    count=len(jobs)
    dt=float(raw.step_dt)
    steps=int(round(protocol['evaluation']['horizon_seconds']/dt))
    if steps<1:raise ValueError('evaluation horizon shorter than one control step')
    command=np.asarray([protocol['evaluation']['commands'][c] for c,_,_ in jobs],np.float32)
    commands=np.broadcast_to(command[None],(steps,count,3)).copy()
    velocities=np.zeros((steps,count,3),np.float32)
    actions=np.zeros((steps,count,29),np.float32)
    terminations=np.zeros((steps,count),bool)
    alive=np.ones(count,bool);alive_steps=np.full(count,steps,np.int64)
    rngs=[np.random.default_rng(np.random.SeedSequence([episode_seed,c,20261026])) for c,_,_ in jobs]
    delays=[study.ActionDelay(settings.get('action_delay_steps',0),1,29) for _,_,settings in jobs]
    study.seed_all(episode_seed)
    raw.reset(seed=episode_seed)
    robot=raw.scene['robot']
    if protocol.get('terrain_mode') == 'blind_rough':
        terrain = raw.scene.terrain
        snapshot = {name: getattr(terrain, name).detach().cpu().tolist()
                    for name in ('terrain_levels', 'terrain_types', 'env_origins')}
        snapshot.update({name: getattr(robot.data, field).detach().cpu().tolist()
                         for name, field in (('initial_root_pos_w', 'root_link_pos_w'),
                                             ('initial_root_quat_w', 'root_link_quat_w'),
                                             ('initial_root_vel_w', 'root_link_vel_w'))})
        verify_rough_pairing(Path(target).parent/'_rough_pairing'/f'episode_seed{episode_seed}.json', snapshot)
        study.write(Path(target)/f'episode_seed{episode_seed}_spawn.json', snapshot)
    for step in range(steps):
        # Controlled velocity impulse, once at the specified instant; never rely
        # on play=True's (disabled) push event for robustness.
        for i,(_,_,settings) in enumerate(jobs):
            if ('lateral_delta_v_mps' in settings
                    and step==round(settings['at_seconds']/dt) and alive[i]):
                ids=torch.tensor([i],device=device,dtype=torch.long)
                v=robot.data.root_link_vel_w[ids].clone()
                sign=1. if episode_seed%2 else -1.
                v[:,:3]+=body_lateral_velocity_delta(robot.data.root_link_quat_w[ids],
                                                      sign*settings['lateral_delta_v_mps'])
                robot.write_root_link_velocity_to_sim(v,env_ids=ids)
        if any('lateral_delta_v_mps' in settings and step==round(settings['at_seconds']/dt)
               for _,_,settings in jobs):
            raw.scene.write_data_to_sim();raw.sim.forward();raw.sim.sense()
        raw.command_manager.get_command('twist').copy_(torch.as_tensor(command,device=device))
        obs=raw.observation_manager.compute(update_history=False)['actor'].detach().cpu().numpy().astype(np.float32)
        current=torch.cat((robot.data.root_link_lin_vel_b[:,:2],robot.data.root_link_ang_vel_b[:,2:3]),dim=-1)
        velocities[step]=current.detach().cpu().numpy()
        batch=[]
        for i,(_,_,settings) in enumerate(jobs):
            perturbed=obs[i:i+1]+observation_noise(layout,settings,protocol,rngs[i],1)
            value=session.run(None,{'obs':perturbed})[0]
            if not np.isfinite(value).all():raise ValueError('nonfinite inference action')
            batch.append(delays[i](value)[0])
        action=np.asarray(batch,np.float32)
        action[~alive]=0.
        actions[step]=action
        _,_,terminated,timed_out,_=raw.step(torch.as_tensor(action,device=device))
        if protocol.get('terrain_mode') == 'blind_rough':
            study.check_rough_simulation(raw)
        failed=terminated.detach().cpu().numpy().astype(bool)&alive
        # The configured time limit exceeds the evaluation horizon. An early
        # timeout is a protocol failure, not a completed or silently dropped run.
        if np.any(timed_out.detach().cpu().numpy().astype(bool)&alive):
            raise ValueError('unexpected timeout before fixed evaluation horizon')
        alive_steps[failed]=step+1;terminations[step]=failed;alive[failed]=False
    target=Path(target);target.mkdir(parents=True,exist_ok=True)
    trajectory=target/f'episode_seed{episode_seed}.npz'
    if trajectory.exists():raise FileExistsError('raw trajectory already exists without a validated result')
    np.savez_compressed(trajectory,commands=commands,velocities=velocities,actions=actions,
                        terminated=terminations,alive_steps=alive_steps,dt=dt,
                        identity_json=json.dumps(dict(identity, episode_seed=episode_seed), sort_keys=True))
    rows=[]
    for i,(command_id,scenario,_) in enumerate(jobs):
        metrics=study.episode_metrics(commands[:,i],velocities[:,i],alive_steps=int(alive_steps[i]),
                                      fell=not bool(alive[i]),dt=dt)
        rows.append(dict(episode_seed=episode_seed,command_id=command_id,scenario=scenario,
                         command=command[i].tolist(),slot=i,**metrics))
    batch_record={'identity':dict(identity, episode_seed=episode_seed),
                  'episode_seed':episode_seed,'dt':dt,'horizon_steps':steps,'rows':rows,
                  'trajectory_file':trajectory.name,'trajectory_sha256':study.sha(trajectory)}
    study.write(target/f'episode_seed{episode_seed}.json',batch_record)
    return rows


def evaluate(output,arm,training_seed,split='holdout',*,device='cuda:0',resume=False):
    output=Path(output);protocol=prepare(output)
    seal=read(output/'formal_policies_sealed.json')
    # Development uses the same final-policy identity and separate episode seeds;
    # no held-out episode is permitted before all 12 training runs are sealed.
    study.validate_policy_seal(seal,protocol)
    policy=next(r for r in seal['policies'] if r['arm']==arm and r['seed']==training_seed)
    target=output/'evaluation'/split/f'{arm}_seed{training_seed}'
    manifest=target/'manifest.json'
    identity={'arm':arm,'training_seed':training_seed,'split':split,
              'policy_sha256':policy['policy_sha256'],'protocol_sha256':study.protocol_digest(protocol),
              'source_sha256':seal['source_sha256']}
    if manifest.exists():
        old=read(manifest)
        if old['identity']!=identity:raise ValueError('existing evaluation has a different identity')
        if old['status']=='COMPLETED':
            verified_evaluation_rows(target, old, protocol, identity)
            return old
        if not resume:raise FileExistsError('incomplete evaluation preserved; explicit --resume required')
        study.write(target/'attempts'/f'manifest_{time.time_ns()}.json',old)
    record={'schema':'g1.semantic_split.evaluation.v1','status':'RUNNING','started_at':study.now(),
            'identity':identity,'policy_freeze_verified_at':study.now(),'device':device}
    study.write(manifest,record)
    raw=None
    try:
        study.bootstrap();study.seed_all(20262400)
        from mjlab.envs import ManagerBasedRlEnv
        count=len(protocol['evaluation']['commands'])*len(protocol['evaluation']['scenarios'])
        cfg=study.make_env_config(protocol,20262400,count,evaluation=True)
        raw=ManagerBasedRlEnv(cfg=cfg,device=device)
        layout=study.runtime_layout(raw)
        study.write(target/'environment.json',environment_record())
        study.write(target/'observation_layout.json',layout)
        session=cpu_session(policy['policy'])
        rows=[];trajectories=[]
        for seed in protocol['evaluation'][split+'_seeds']:
            batch_path=target/f'episode_seed{seed}.json'
            raw_path=target/f'episode_seed{seed}.npz'
            if batch_path.exists():
                batch=load_batch(target, seed, protocol, identity)
                rows.extend(batch['rows'])
            else:
                if raw_path.exists():
                    # Preserve an uncommitted partial batch, then rerun that seed.
                    raw_path.rename(target/f'partial_{time.time_ns()}_{raw_path.name}')
                rows.extend(rollout_seed(raw,session,layout,protocol,seed,device,target,identity))
                batch=load_batch(target, seed, protocol, identity)
            trajectories.append({'file':batch['trajectory_file'],'sha256':batch['trajectory_sha256'],
                                 'batch_file':batch_path.name,'batch_sha256':study.sha(batch_path)})
            record['completed_episode_seed_batches']=len(trajectories);record['updated_at']=study.now()
            study.write(manifest,record)
        for row in rows:row.update(arm=arm,training_seed=training_seed)
        count=validate_coverage(rows,protocol,split)
        study.write(target/'episodes.json',rows)
        record.update(status='COMPLETED',ended_at=study.now(),episode_count=count,
                      episodes_sha256=study.sha(target/'episodes.json'),raw_trajectories=trajectories,
                      pairing='same construction/reset seed and command-scenario slot across all actors')
        return record
    except BaseException as error:
        record.update(status='FAILED',ended_at=study.now(),error=repr(error),traceback=traceback.format_exc())
        raise
    finally:
        study.write(manifest,record)
        if raw is not None:raw.close()


def smoke_evaluate(output,arm,seed,*,device='cpu'):
    """Short engineering rollout; cannot enter the scientific holdout results."""
    protocol=prepare(output)
    directory=run_directory(output,arm,seed,'smoke')
    training=read(directory/'manifest.json')
    if training['status']!='COMPLETED':raise ValueError('smoke policy incomplete')
    protocol=copy.deepcopy(protocol)
    protocol['evaluation']['horizon_seconds']=.2
    protocol['evaluation']['commands']=[[.5,0.,0.]]
    for settings in protocol['evaluation']['scenarios'].values():
        if 'at_seconds' in settings:settings['at_seconds']=.04
    target=Path(output)/'smoke_evaluation'/arm
    if target.exists():raise FileExistsError('preserve previous smoke evaluation')
    study.bootstrap();study.seed_all(20262499)
    from mjlab.envs import ManagerBasedRlEnv
    cfg=study.make_env_config(protocol,20262499,len(protocol['evaluation']['scenarios']),evaluation=True)
    raw=ManagerBasedRlEnv(cfg=cfg,device=device)
    try:
        identity={'arm':arm,'training_seed':seed,'split':'engineering_smoke',
                  'policy_sha256':study.sha(directory/'policy.onnx')}
        rows=rollout_seed(raw,cpu_session(directory/'policy.onnx'),study.runtime_layout(raw),
                          protocol,20262499,device,target,identity)
        load_batch(target,20262499,protocol,identity)
        record={'status':'COMPLETED','scope':'engineering smoke, not scientific outcome',
                'episodes':len(rows),'policy_sha256':study.sha(directory/'policy.onnx'),'at':study.now()}
        study.write(target/'manifest.json',record)
        return record
    finally:raw.close()


def benchmark_policy(path,protocol):
    import psutil
    session=cpu_session(path)
    x=np.random.default_rng(20261026).normal(size=(1,98)).astype(np.float32)
    cfg=protocol['latency']
    for _ in range(cfg['warmup']):session.run(None,{'obs':x})
    started_cpu_load=psutil.cpu_percent(interval=.1)
    measurements=[]
    for _ in range(cfg['samples']):
        start=time.perf_counter_ns();session.run(None,{'obs':x})
        measurements.append((time.perf_counter_ns()-start)/1000.)
    return {'latency_us':{f'p{q}':float(np.percentile(measurements,q)) for q in cfg['quantiles']},
            'raw_latency_us':measurements,'samples':cfg['samples'],'warmup':cfg['warmup'],
            'cpu':platform.processor(),'logical_cpus':psutil.cpu_count(),'system_load_percent_before':started_cpu_load,
            'backend':cfg['backend'],'threads':cfg['threads'],'batch':1,'scope':cfg['scope'],
            'policy_sha256':study.sha(path),'at':study.now(),
            'limitation':'shared-machine CPU timing; repeat on target robot hardware before deployment claims'}


def benchmark_all(output,*,smoke=False):
    output=Path(output);protocol=prepare(output)
    if smoke:
        policies=[{'arm':arm,'seed':protocol['training']['pilot_seed'],
                   'policy':str(run_directory(output,arm,protocol['training']['pilot_seed'],'smoke')/'policy.onnx')}
                  for arm in study.ARMS]
    else:
        seal=read(output/'formal_policies_sealed.json');study.validate_policy_seal(seal,protocol)
        policies=seal['policies']
    rows=[]
    # Interleave arms by training seed rather than timing all copies of one arm.
    for policy in policies:
        rows.append(dict(arm=policy['arm'],training_seed=policy['seed'],
                         **benchmark_policy(policy['policy'],protocol)))
    study.write(output/('smoke_latency.json' if smoke else 'latency.json'),
                {'status':'COMPLETED','scope':'untrained smoke policies' if smoke else 'final trained policies',
                 'protocol_sha256':study.protocol_digest(protocol),'results':rows})
    return rows
