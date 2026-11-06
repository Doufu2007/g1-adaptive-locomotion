"""Independent CPU arithmetic audit for the confirmation evaluation.
No production scorer or production analysis module is imported.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, itertools, json, math, statistics
import numpy as np
from scipy.stats import t

O=Path(__file__).resolve().parent
METHODS=('RE_live','RE_frozen','RC_clock','FC_clock')
BASE={'RE_live':'RE','RE_frozen':'RE','RC_clock':'RC','FC_clock':'FC'}
SEEDS=(20261901,20261902,20261903)
MAPS=(20264901,20264902,20264903)
EVAL=(20264911,20264912,20264913,20264914,20264915)
SCENARIOS=('steady','push')
LIMITS={'mae_forward_velocity_mps':.02,'forward_displacement_m':-.5,'tilt_rms_rad':.02,'abs_heading_change_rad':.2}

def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):
    h=hashlib.sha256(); h.update(Path(path).read_bytes()); return h.hexdigest()
def close(x,y,tol=2e-6):
    if isinstance(x,(float,int,np.floating,np.integer)) or isinstance(y,(float,int,np.floating,np.integer)):
        if not math.isclose(float(x),float(y),rel_tol=tol,abs_tol=tol): raise AssertionError((x,y))
    else: np.testing.assert_allclose(x,y,rtol=tol,atol=tol)
def digest(arrays):
    h=hashlib.sha256()
    for key,value in sorted(arrays.items()):
        a=np.ascontiguousarray(value)
        if a.dtype.hasobject or not np.isfinite(a).all(): raise AssertionError(f'nonfinite physical array {key}')
        h.update(key.encode()); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()
def contrast(x):
    x=np.asarray(x,dtype=float); assert x.shape==(3,) and np.isfinite(x).all()
    mean=float(x.mean()); half=float(t.ppf(.975,2)*x.std(ddof=1)/math.sqrt(3))
    possible=[abs(float(np.mean(x*np.asarray(sign)))) for sign in itertools.product((-1,1),repeat=3)]
    return dict(per_seed=x.tolist(),mean=mean,ci95=[mean-half,mean+half],p_exact=float(np.mean(np.asarray(possible)>=abs(mean)-1e-12)),n=3)
def holm(values):
    p=np.asarray(values,float); order=np.argsort(p); out=np.empty(len(p)); last=0.
    for i,k in enumerate(order): last=max(last,min(1.,float(p[k])*(len(p)-i))); out[k]=last
    return out.tolist()
def metric(raw,terrain,fell):
    n=len(raw['time']); assert 0<n<=1000
    np.testing.assert_allclose(raw['time'],np.arange(n)*.02,rtol=0,atol=1e-7)
    assert not raw['timed_out'].any() and not raw['terminated'][:-1].any() and (n==1000 or fell)
    padded=np.zeros(1000); padded[:n]=raw['velocity'][:,0]; mae=float(np.abs(padded-.5).mean())
    q=raw['quaternion']; yaw=np.unwrap(np.arctan2(2*(q[:,0]*q[:,3]+q[:,1]*q[:,2]),1-2*(q[:,2]**2+q[:,3]**2)))
    displacement=float((raw['position'][-1,:2]-raw['position'][0,:2])@np.array([np.cos(yaw[0]),np.sin(yaw[0])]))
    tilt=float(np.sqrt(np.mean(np.arccos(np.clip(-raw['gravity'][:,2],-1,1))**2)))
    heading=float(abs(yaw[-1]-yaw[0])); moving=float(np.sum(raw['velocity'][:,0]>.1)/1000)
    fraction=float(raw['points_interior'].mean()) if len(raw['points_interior']) else None
    completed=bool(not fell and n==1000 and displacement>=5 and moving>=.8 and (not terrain or fraction is not None and fraction>=.95))
    return dict(fell=bool(fell),completed=completed,mae_forward_velocity_mps=mae,forward_displacement_m=displacement,tilt_rms_rad=tilt,abs_heading_change_rad=heading,moving_fraction=moving,rough_contact_fraction=fraction,foot_contact_total=int(raw['foot_contact_total'].sum()),foot_contact_interior=int(raw['foot_contact_interior'].sum()))
def main():
    protocol=read(O/'prospective_protocol.json'); protocol_sha=sha(O/'prospective_protocol.json')
    analysis=read(O/'analysis.json'); assert analysis['status']=='COMPLETED' and analysis['protocol_sha256']==protocol_sha
    status=read(O/'queue_status.json'); assert status['status']=='COMPLETED' and status['completed_jobs']==12 and status['PPO_updates']==status['Adam_steps']==0
    assert status['analysis_sha256']==sha(O/'analysis.json')
    assert protocol['identity']['formal_episodes']==360 and protocol['identity']['maximum_effective_eval_transitions']==360000
    for entry in protocol['identity']['files']: assert sha(entry['path'])==entry['sha256'], entry['path']
    for entry in protocol['identity']['policies']:
        for key in ('checkpoint','policy','training_manifest'): assert sha(entry[key])==entry[key+'_sha256'], key
    rows=[]; raw_checks=[]; total_steps=0; world_steps=0; starts={}; physical={}
    for method in METHODS:
        for terrain in MAPS:
            folder=O/'rollouts'/f'{method}_terrain{terrain}'; manifest=read(folder/'manifest.json'); episodes=read(folder/'episodes.json')
            assert manifest['status']=='COMPLETED' and manifest['protocol_sha256']==protocol_sha and len(episodes)==manifest['episode_count']==30
            assert manifest['PPO_updates']==manifest['Adam_steps']==0
            assert manifest['method']==method and manifest['terrain_seed']==terrain
            expected_slots=[(s,e,i) for s in SEEDS for i,e in enumerate(EVAL)]
            assert [(x['train_seed'],x['eval_seed'],x['patch_col']) for x in manifest['slots']]==expected_slots
            for scenario in SCENARIOS:
                initial=read(folder/f'{scenario}_initial.json'); key=(terrain,scenario)
                if key in starts: assert starts[key]==initial
                starts[key]=initial
                phys=manifest[scenario+'_physics']; p=folder/phys['path']; assert sha(p)==phys['sha256']
                with np.load(p,allow_pickle=False) as saved: ph=digest(dict(saved))
                assert ph==phys['array_sha256']==initial['physical_sha256']
                if key in physical: assert physical[key]==ph
                physical[key]=ph
            for row in episodes:
                path=folder/row['raw_path']; assert sha(path)==row['raw_sha256']; raw_checks.append(dict(path=str(path),sha256=sha(path)));
                with np.load(path,allow_pickle=False) as saved: raw=dict(saved)
                n=len(raw['time']); assert n>0 and n<=1000
                assert raw['terminated'].shape==(n,) and raw['timed_out'].shape==(n,)
                fell=bool(raw['terminated'][-1]); scores=metric(raw,bool(terrain),fell)
                for key,value in scores.items(): close(value,row[key])
                assert raw['observation'].shape[1]==98 and np.isfinite(raw['observation']).all()
                np.testing.assert_allclose(raw['observation'][:,6:9],np.tile([.5,0.,0.],(n,1)),rtol=0,atol=1e-7)
                np.testing.assert_allclose(raw['observation'][:,9:11],np.stack([np.sin(raw['angle']),np.cos(raw['angle'])],axis=1),rtol=0,atol=2e-6)
                assert raw['phase_live'].shape==raw['phase_used'].shape==(n,2) and np.isfinite(raw['phase_live']).all() and np.isfinite(raw['phase_used']).all()
                np.testing.assert_allclose(raw['phase_used'],np.stack([np.sin(raw['angle']),np.cos(raw['angle'])],axis=1),rtol=0,atol=2e-6)
                if method=='RE_frozen': np.testing.assert_allclose(raw['phase_used'],np.broadcast_to(raw['phase_used'][0],raw['phase_used'].shape),rtol=0,atol=2e-6)
                slot=next(i for i,s in enumerate(manifest['slots']) if s['train_seed']==row['train_seed'] and s['eval_seed']==row['eval_seed'])
                initial=read(folder/f'{row["scenario"]}_initial.json'); close(raw['position'][0],initial['initial']['position'][slot]); close(raw['quaternion'][0],initial['initial']['quaternion'][slot])
                feet=set(manifest['foot_geom_ids']); terrain_geoms=set(manifest['terrain_geom_ids']); assigned=set(initial['assigned'][slot]); xlo,xhi,ylo,yhi=initial['bounds'][slot]
                flags=[]
                for k,(g0,g1) in enumerate(raw['points_geom']):
                    assert ((g0 in feet and g1 in terrain_geoms) or (g1 in feet and g0 in terrain_geoms)) and raw['points_dist'][k]<raw['points_includemargin'][k] and np.any(raw['points_efc_address'][k]>=0)
                    geom=g1 if g0 in feet else g0; x,y=raw['points_pos'][k,:2]; flags.append(bool(geom in assigned and xlo+.25<x<xhi-.25 and ylo+.25<y<yhi-.25))
                flags=np.asarray(flags,bool); np.testing.assert_array_equal(flags,raw['points_interior'])
                offsets=raw['contact_offsets']; assert offsets.shape==(n+1,) and offsets[0]==0 and offsets[-1]==len(flags) and np.all(np.diff(offsets)>=0)
                np.testing.assert_array_equal(np.diff(offsets),raw['foot_contact_total']); cumulative=np.r_[0,np.cumsum(flags.astype(int))]; np.testing.assert_array_equal(np.diff(cumulative[offsets]),raw['foot_contact_interior'])
                rows.append(dict(row,**scores)); total_steps+=n
            assert sum(x['observed_steps'] for x in episodes)==manifest['transitions']; world_steps+=manifest['simulator_world_steps']
    expected={(m,t,s,e,q) for m in METHODS for t in MAPS for s in SEEDS for e in EVAL for q in SCENARIOS}; keys=[tuple(x[k] for k in ('method','terrain_seed','train_seed','eval_seed','scenario')) for x in rows]
    assert len(keys)==len(set(keys))==360 and set(keys)==expected and total_steps==analysis['effective_eval_transitions']
    metrics=('completed','fell','mae_forward_velocity_mps','forward_displacement_m','tilt_rms_rad','abs_heading_change_rad')
    means={m:{metric:[statistics.mean(float(x[metric]) for x in rows if x['method']==m and x['train_seed']==seed) for seed in SEEDS] for metric in metrics} for m in METHODS}
    family={}
    for name,weights in {'RE_live_minus_RC_clock':{'RE_live':1,'RC_clock':-1},'RE_live_minus_RE_frozen':{'RE_live':1,'RE_frozen':-1}}.items():
        for metric_name in ('completed','fell'):
            family[name+':'+metric_name]=contrast([sum(weight*means[m][metric_name][i] for m,weight in weights.items()) for i in range(3)])
    for value,p in zip(family.values(),holm([value['p_exact'] for value in family.values()])): value['p_holm']=p
    for key,value in family.items():
        for sub,val in value.items(): close(val,analysis['statistics']['family'][key][sub])
    def diffs(other): return {metric_name:np.asarray(means['RE_live'][metric_name])-means[other][metric_name] for metric_name in metrics}
    candidates={}
    for other in ('RC_clock','RE_frozen'):
        d=diffs(other); gates={'completion_margin':bool(d['completed'].mean()>=.10-1e-12),'fall_margin':bool(d['fell'].mean()<=-.10+1e-12),'completion_seed_direction':bool(np.sum(d['completed']>0)>=2),'fall_seed_direction':bool(np.sum(d['fell']<0)>=2)}
        gates.update({k:bool(d[k].mean()>=v-1e-12 if v<0 else d[k].mean()<=v+1e-12) for k,v in LIMITS.items()})
        candidates[other]={'gates':gates,'passed':all(gates.values())}
        assert gates==analysis['statistics']['candidate']['comparators'][other]['gates']
    result=dict(status='PASS',at=datetime.now(timezone.utc).isoformat(),analysis_sha256=sha(O/'analysis.json'),raw_count=360,effective_eval_transitions=total_steps,simulator_world_steps=world_steps,evaluation_PPO_updates=0,evaluation_Adam_steps=0,raw_checks=raw_checks,candidate_comparator_pass={k:v['passed'] for k,v in candidates.items()},family=family,independent_arithmetic='same operator; separate CPU raw/contact/phase/seed/statistics implementation; no production scorer or analysis imports; not a second reviewer or new seed',scope='fixed S checkpoints on three fresh rough maps and five fresh eval seeds; matched live-vs-frozen feedback ablation')
    (O/'independent_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('raw_checks','family')},ensure_ascii=False))
if __name__=='__main__': main()
