"""Prospective CPU contract for the finite, zero-PPO rough transfer study."""
import hashlib
import itertools
import math
import random
import numpy as np

TRAIN_SEEDS=(20261901,20261902,20261903)
EVAL_SEEDS=tuple(range(20264311,20264316))
TERRAINS=(0,20264201,20264202,20264203)
METHODS=('AC','AE','DC','DE')
SCENARIOS=('steady','push')
GUARDS={'mae_forward_velocity_mps':.02,'forward_displacement_m':-.5,
        'tilt_rms_rad':.02,'abs_heading_change_rad':.2}


def build_jobs():
    jobs=[dict(method=m,terrain_seed=t) for m in METHODS for t in TERRAINS]
    random.Random(20264290).shuffle(jobs)
    return jobs


def world_slots():
    return [dict(train_seed=s,eval_seed=e,patch_col=i) for s in TRAIN_SEEDS for i,e in enumerate(EVAL_SEEDS)]


def route_actions(obs,sessions,names):
    if obs.ndim!=2 or obs.shape[1]!=98 or len(obs)!=len(sessions) or len(names)!=len(obs) or not np.isfinite(obs).all():
        raise ValueError('aligned 98-dimensional actor worlds required')
    action=np.concatenate([s.run(None,{name:obs[i:i+1]})[0] for i,(s,name) in enumerate(zip(sessions,names))])
    if action.shape!=(len(obs),29) or not np.isfinite(action).all():raise ValueError('invalid routed actions')
    return action


def array_digest(arrays):
    h=hashlib.sha256()
    for k,v in sorted(arrays.items()):
        a=np.ascontiguousarray(v)
        if a.dtype.hasobject or not np.isfinite(a).all():raise ValueError('invalid physical array')
        h.update(k.encode());h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()


def contact_exposure(geom,world,positions,dist,margin,efc,foot_ids,terrain_ids,assigned,bounds,num_envs):
    geom=np.asarray(geom);world=np.asarray(world);positions=np.asarray(positions)
    dist=np.asarray(dist);margin=np.asarray(margin);efc=np.asarray(efc);bounds=np.asarray(bounds)
    n=len(world)
    if geom.shape!=(n,2) or positions.shape!=(n,3) or dist.shape!=(n,) or margin.shape!=(n,) or efc.ndim!=2 or efc.shape[0]!=n:
        raise ValueError('invalid contact shapes')
    if bounds.shape!=(num_envs,4) or len(assigned)!=num_envs or np.any(world<0) or np.any(world>=num_envs):
        raise ValueError('invalid contact world assignment')
    if not all(np.isfinite(x).all() for x in (geom,world,positions,dist,margin,efc,bounds)):raise ValueError('nonfinite contacts')
    foot0=np.isin(geom[:,0],list(foot_ids));foot1=np.isin(geom[:,1],list(foot_ids))
    terrain0=np.isin(geom[:,0],list(terrain_ids));terrain1=np.isin(geom[:,1],list(terrain_ids))
    valid=((foot0&terrain1)|(foot1&terrain0))&(efc>=0).any(axis=1)&(dist<margin)
    indices=np.flatnonzero(valid);selected_world=world[indices].astype(int)
    terrain_geom=np.where(terrain0,geom[:,0],geom[:,1])[indices]
    interior=np.zeros(len(indices),bool)
    for k,(i,w,g) in enumerate(zip(indices,selected_world,terrain_geom)):
        xlo,xhi,ylo,yhi=bounds[w];x,y=positions[i,:2]
        interior[k]=g in assigned[w] and x>xlo+.25 and x<xhi-.25 and y>ylo+.25 and y<yhi-.25
    return dict(indices=indices,world=selected_world,terrain_geom=terrain_geom,interior_mask=interior,
        total=np.bincount(selected_world,minlength=num_envs),
        interior=np.bincount(selected_world[interior],minlength=num_envs))


def episode_metrics(raw,horizon,scenario,fell,rough):
    from scripts.g1_closed_loop_metrics import summarize_episode
    n=len(raw['time']);steps=round(horizon/.02)
    if not 0<n<=steps:raise ValueError('invalid observation length')
    for k in ('time','position','quaternion','velocity','gravity','angle','contact','foot_contact_total','foot_contact_interior'):
        if len(raw[k])!=n or not np.isfinite(raw[k]).all():raise ValueError('invalid raw telemetry: '+k)
    np.testing.assert_allclose(raw['time'],np.arange(n)*.02,atol=1e-7)
    if n<steps and not fell:raise ValueError('unexplained missing tail')
    total=int(np.sum(raw['foot_contact_total']));inside=int(np.sum(raw['foot_contact_interior']))
    if total<0 or inside<0 or inside>total:raise ValueError('invalid exposure counts')
    fraction=inside/total if total else None
    m=summarize_episode(dict(raw,scenario=scenario),np.full(steps,.5),.02,fell)
    q=np.asarray(raw['quaternion']);yaw=np.unwrap(np.arctan2(2*(q[:,0]*q[:,3]+q[:,1]*q[:,2]),1-2*(q[:,2]**2+q[:,3]**2)))
    delta=raw['position'][-1,:2]-raw['position'][0,:2]
    distance=float(delta@np.array([np.cos(yaw[0]),np.sin(yaw[0])]))
    tilt=np.arccos(np.clip(-raw['gravity'][:,2],-1.,1.))
    moving=float(np.count_nonzero(raw['velocity'][:,0]>.1)/steps)
    m.update(forward_displacement_m=distance,moving_fraction=moving,tilt_rms_rad=float(np.sqrt(np.mean(tilt**2))),
        abs_heading_change_rad=float(abs(yaw[-1]-yaw[0])),rough_contact_fraction=fraction,
        foot_contact_total=total,foot_contact_interior=inside,zero_contact_frames=int(np.sum(raw['foot_contact_total']==0)),
        exposure_passed=bool(not rough or (fraction is not None and fraction>=.95)),
        completed=bool(not fell and n==steps and distance>=5. and moving>=.8 and (not rough or (fraction is not None and fraction>=.95))))
    return m


def termination_from_raw(raw):
    n=len(raw['time']);terminated=np.asarray(raw['terminated']);timed=np.asarray(raw['timed_out'])
    if terminated.shape!=(n,) or timed.shape!=(n,) or terminated.dtype!=bool or timed.dtype!=bool:
        raise ValueError('native terminal flags required for each effective step')
    if timed.any() or terminated[:-1].any():raise ValueError('timeout or concatenated terminal episode')
    return bool(terminated[-1])


def seed_comparison(values):
    from scipy.stats import t
    x=np.asarray(values,dtype=float)
    if x.shape!=(3,) or not np.isfinite(x).all():raise ValueError('exactly three finite training-seed differences required')
    mean=float(x.mean());sem=float(x.std(ddof=1)/np.sqrt(3));half=float(t.ppf(.975,2))*sem
    possible=[abs(np.mean(x*np.array(s))) for s in itertools.product((-1,1),repeat=3)]
    p=float(np.mean(np.asarray(possible)>=abs(mean)-1e-12))
    return dict(per_seed=x.tolist(),mean=mean,ci95=[mean-half,mean+half],p_exact=p,n=3)


def holm(pvalues):
    p=np.asarray(pvalues,float);order=np.argsort(p);out=np.empty(len(p));last=0.
    for i,k in enumerate(order):last=max(last,min(1.,float(p[k])*(len(p)-i)));out[k]=last
    return out.tolist()


def paired_analysis(rows):
    expected={(j['method'],j['terrain_seed'],s['train_seed'],s['eval_seed'],q) for j in build_jobs() for s in world_slots() for q in SCENARIOS}
    by={}
    metrics=('completed','fell',*GUARDS)
    for r in rows:
        key=tuple(r[k] for k in ('method','terrain_seed','train_seed','eval_seed','scenario'))
        if key in by or not all(math.isfinite(float(r[m])) for m in metrics):raise ValueError('duplicate or nonfinite episode')
        by[key]=r
    if set(by)!=expected:raise ValueError('missing or extra factorial episodes')
    means={}
    for method in METHODS:
        means[method]={m:[float(np.mean([by[method,t,s,e,q][m] for t in TERRAINS[1:] for e in EVAL_SEEDS for q in SCENARIOS])) for s in TRAIN_SEEDS] for m in metrics}
    contrasts={'DE_minus_DC':{'DE':1,'DC':-1},'AE_minus_AC':{'AE':1,'AC':-1},
        'DE_minus_AE':{'DE':1,'AE':-1},'interaction':{'DE':1,'AE':-1,'DC':-1,'AC':1}}
    family={}
    for name,weights in contrasts.items():
        for metric in ('completed','fell'):
            x=sum(w*np.asarray(means[m][metric]) for m,w in weights.items())
            family[name+':'+metric]=seed_comparison(x)
    adjusted=holm([v['p_exact'] for v in family.values()])
    for v,p in zip(family.values(),adjusted):v['p_holm']=p
    candidate={}
    for other in ('DC','AC'):
        diffs={m:np.asarray(means['DE'][m])-means[other][m] for m in metrics}
        gates=dict(completion_margin=bool(diffs['completed'].mean()>=.10-1e-12),fall_margin=bool(diffs['fell'].mean()<=-.10+1e-12),
            completion_seed_direction=bool(np.sum(diffs['completed']>0)>=2),fall_seed_direction=bool(np.sum(diffs['fell']<0)>=2))
        for k,limit in GUARDS.items():gates[k]=bool(diffs[k].mean()>=limit-1e-12 if limit<0 else diffs[k].mean()<=limit+1e-12)
        candidate[other]=dict(gates=gates,passed=all(gates.values()),differences={k:seed_comparison(v) for k,v in diffs.items()})
    return dict(independent_training_seeds=3,rough_seed_means=means,family=family,
        candidate=dict(comparators=candidate,passed=all(v['passed'] for v in candidate.values()),
            meaning='exploratory point-effect gate only; new training seeds/maps and causal ablation still required'),
        inference='conditional on 15 shared rough patches; exact seed sign-flip minimum two-sided p=.25; no confirmed superiority')
