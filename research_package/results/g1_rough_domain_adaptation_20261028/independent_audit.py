"""Independent CPU arithmetic, no production scorer or analysis imports."""
from pathlib import Path
import hashlib,itertools,json,math,statistics
import numpy as np
from scipy.stats import t
from datetime import datetime,timezone
O=Path(__file__).resolve().parent
SEEDS=(20261901,20261902,20261903);EVAL=tuple(range(20264811,20264816));MAPS=(0,20264701,20264702,20264703);METHODS=('FC','FE','RC','RE')
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def close(a,b):
 if a is None or b is None:assert a is b;return
 np.testing.assert_allclose(a,b,atol=1e-10,rtol=1e-8)
def digest(arrays):
 h=hashlib.sha256()
 for k,v in sorted(arrays.items()):
  v=np.ascontiguousarray(v);h.update(k.encode());h.update(str(v.dtype).encode());h.update(str(v.shape).encode());h.update(v.tobytes())
 return h.hexdigest()
def contrast(x):
 mean=statistics.mean(x);half=float(t.ppf(.975,2))*statistics.stdev(x)/math.sqrt(3)
 p=sum(abs(sum(z*v for z,v in zip(sign,x))/3)>=abs(mean)-1e-12 for sign in itertools.product((-1,1),repeat=3))/8
 return dict(per_seed=x,mean=mean,ci95=[mean-half,mean+half],p_exact=p,n=3)
def holm(ps):
 order=sorted(range(len(ps)),key=lambda k:ps[k]);out=[0.]*len(ps);last=0.
 for i,k in enumerate(order):last=max(last,min(1.,ps[k]*(len(ps)-i)));out[k]=last
 return out

def recursive_equal(x,y):
 import torch
 assert type(x)==type(y)
 if isinstance(x,torch.Tensor):torch.testing.assert_close(x.cpu(),y.cpu(),atol=0,rtol=0)
 elif isinstance(x,np.ndarray):np.testing.assert_array_equal(x,y)
 elif isinstance(x,dict):
  assert set(x)==set(y)
  for k in x:recursive_equal(x[k],y[k])
 elif isinstance(x,(tuple,list)):
  assert len(x)==len(y)
  for a,b in zip(x,y):recursive_equal(a,b)
 else:assert x==y

def training_audit(protocol):
 import torch
 robot={};physical={};trainmaps={};total=0
 sources={(p['method'],p['source_seed']):p for p in protocol['identity']['sources']}
 for method in METHODS:
  for seed in SEEDS:
   folder=O/'training'/f'{method}_{seed}';m=read(folder/'manifest.json');source=sources[method,seed]
   assert m['source']==source and m['updates']==251 and m['transitions']==385536 and m['status']=='COMPLETED'
   assert m['PPO_updates']==251 and m['Adam_steps']==5020 and m['learn_calls']==1 and m['cumulative_Adam_steps']==15040
   assert m['encoder_frozen'] and m['teacher_frozen'] and m['protocol_sha256']==sha(O/'prospective_protocol.json')
   src=torch.load(source['checkpoint'],map_location='cpu',weights_only=False)
   assert src['iter']==500 and src['infos']['factorial_state']['completed_updates']==501
   for point in (0,1,25,100,251):
    entry=m['checkpoints'][str(point)];assert sha(entry['path'])==entry['sha256'] and sha(entry['policy'])==entry['policy_sha256']
    cp=torch.load(entry['path'],map_location='cpu',weights_only=False);st=cp['infos']['domain_state']
    assert st['completed_updates']==point and st['source_completed_updates']==501 and cp['source_anchor_coefficient']==.1
    recursive_equal(cp['source_teacher_state_dict'],src['source_teacher_state_dict'])
    assert len(cp['optimizer_state_dict']['state'])==17
    assert all(float(v['step'])==10020+20*point for v in cp['optimizer_state_dict']['state'].values())
    assert all(g['lr']==st['learning_rate'] for g in cp['optimizer_state_dict']['param_groups'])
    if point==0:
     for key in ('actor_state_dict','critic_state_dict','optimizer_state_dict','source_teacher_state_dict'):recursive_equal(cp[key],src[key])
     for key in ('python_rng','numpy_rng','torch_rng','cuda_rng'):recursive_equal(st[key],src['infos']['factorial_state'][key])
   with np.load(folder/'robot_physics.npz',allow_pickle=False) as data:ph=digest(dict(data))
   if seed in robot:assert robot[seed]==ph
   robot[seed]=ph
   with np.load(folder/'startup_physics.npz',allow_pickle=False) as data:arrays=dict(data)
   ph=digest(arrays);assert ph==m['physical_sha256']
   key=(method[0],seed)
   if key in physical:assert physical[key]==ph
   physical[key]=ph
   wanted=(20264601+SEEDS.index(seed)) if method.startswith('R') else 0
   assert m['training_map']==wanted
   if wanted:
    hh=digest({k:arrays[k] for k in ('hfield_data','hfield_size','hfield_nrow','hfield_ncol','hfield_adr')})
    if wanted in trainmaps:assert trainmaps[wanted]==hh
    trainmaps[wanted]=hh
   lines=[json.loads(line) for line in (folder/'updates.jsonl').read_text(encoding='utf8').splitlines()];assert len(lines)==251
   interiors=0
   for i,row in enumerate(lines,1):
    assert row['completed_update']==i and row['transitions']==1536*i and len(row['minibatches'])==len(row['source_kl'])==20
    assert all(v['beta']==.1 for v in row['source_kl'])
    p=folder/f'exposure_{i}.npz';assert sha(p)==row['exposure_sha256']
    with np.load(p,allow_pickle=False) as data:points=dict(data)
    flags=[];counts=np.zeros(64,int);inside=np.zeros(64,int)
    for k,(g0,g1) in enumerate(points['geom']):
     world=int(points['worldid'][k]);feet=set(m['foot_geom_ids']);ground=set(m['terrain_geom_ids'])
     assert (g0 in feet and g1 in ground) or (g1 in feet and g0 in ground)
     assert points['dist'][k]<points['includemargin'][k] and np.any(points['efc_address'][k]>=0)
     g=g1 if g0 in feet else g0;x,y=points['pos'][k,:2];xlo,xhi,ylo,yhi=m['bounds'][world]
     flag=g in m['assigned'][world] and xlo+.25<x<xhi-.25 and ylo+.25<y<yhi-.25
     flags.append(flag);counts[world]+=1;inside[world]+=int(flag)
    np.testing.assert_array_equal(flags,points['interior']);np.testing.assert_array_equal(counts,points['total']);np.testing.assert_array_equal(inside,points['interior_count'])
    interiors+=int(inside.sum())
   assert m['exposure_boundary_samples']==251 and interiors==m['assigned_contact_total']
   if wanted:assert interiors>0
   total+=m['transitions']
 assert total==4626432
 # Full generated arrays must be disjoint between training and all eval maps.
 evalhash=[]
 for terrain in MAPS[1:]:
  folder=O/'rollouts'/f'FC_terrain{terrain}';m=read(folder/'manifest.json')
  with np.load(folder/m['steady_physics']['path'],allow_pickle=False) as data:arrays=dict(data)
  hh=digest({k:arrays[k] for k in ('hfield_data','hfield_size','hfield_nrow','hfield_ncol','hfield_adr')});evalhash.append(hh)
 assert len(set(evalhash+list(trainmaps.values())))==6

def run():
 protocol=read(O/'prospective_protocol.json');analysis=read(O/'analysis.json')
 assert analysis['status']=='COMPLETED' and analysis['protocol_sha256']==sha(O/'prospective_protocol.json')
 for f in protocol['identity']['files']:assert sha(f['path'])==f['sha256']
 for p in protocol['identity']['sources']:
  for key in ('checkpoint','policy','manifest'):assert sha(p[key])==p[key+'_sha256']
 training_audit(protocol)
 rows=[];starts={};physics={};sampling=0;worldsteps=0;raw_checks=[]
 for method in METHODS:
  for terrain in MAPS:
   folder=O/'rollouts'/f'{method}_terrain{terrain}';m=read(folder/'manifest.json');rr=read(folder/'episodes.json')
   assert m['status']=='COMPLETED' and m['protocol_sha256']==sha(O/'prospective_protocol.json')
   assert len(rr)==m['episode_count']==30 and m['PPO_updates']==m['Adam_steps']==0
   assert [(s['train_seed'],s['eval_seed'],s['patch_col']) for s in m['slots']]==[(s,e,i) for s in SEEDS for i,e in enumerate(EVAL)]
   for slot,p in zip(m['slots'],m['policies']):
    sm=read(O/'training'/f'{method}_{slot["train_seed"]}'/'manifest.json');entry=sm['checkpoints']['251']
    assert p['method']==method and p['train_seed']==slot['train_seed'] and p['completed_updates']==251
    assert p['checkpoint']==entry['path'] and p['checkpoint_sha256']==entry['sha256']
    assert p['policy']==entry['policy'] and p['policy_sha256']==entry['policy_sha256']
    for key in ('checkpoint','policy','training_manifest'):assert sha(p[key])==p[key+'_sha256']
   for scenario in ('steady','push'):
    initial=read(folder/f'{scenario}_initial.json');key=(terrain,scenario)
    assert initial==read(O/'pairing'/f'terrain{terrain}_{scenario}.json')
    if key in starts:assert starts[key]==initial
    starts[key]=initial;entry=m[scenario+'_physics'];p=folder/entry['path'];assert sha(p)==entry['sha256']
    with np.load(p,allow_pickle=False) as data:arrays=dict(data)
    ph=digest(arrays);assert ph==entry['array_sha256']==initial['physical_sha256']
    if key in physics:assert physics[key]==ph
    physics[key]=ph
   for row in rr:
    p=folder/row['raw_path'];assert sha(p)==row['raw_sha256'];raw_checks.append(dict(path=str(p),sha256=sha(p)))
    with np.load(p,allow_pickle=False) as data:r=dict(data)
    n=len(r['time']);assert 0<n<=1000
    np.testing.assert_allclose(r['time'],np.arange(n)*.02,atol=1e-7)
    assert r['terminated'].shape==(n,) and r['timed_out'].shape==(n,)
    assert not r['timed_out'].any() and not r['terminated'][:-1].any()
    fell=bool(r['terminated'][-1]);assert n==1000 or fell
    v=np.zeros(1000);v[:n]=r['velocity'][:,0];mae=float(abs(v-.5).mean())
    q=r['quaternion'];yaw=np.unwrap(np.arctan2(2*(q[:,0]*q[:,3]+q[:,1]*q[:,2]),1-2*(q[:,2]**2+q[:,3]**2)))
    displacement=float((r['position'][-1,:2]-r['position'][0,:2])@np.array([np.cos(yaw[0]),np.sin(yaw[0])]))
    tilt=float(np.sqrt(np.mean(np.arccos(np.clip(-r['gravity'][:,2],-1,1))**2)));heading=float(abs(yaw[-1]-yaw[0]))
    moving=float(np.sum(r['velocity'][:,0]>.1)/1000)
    np.testing.assert_allclose(r['observation'][:,6:9],np.tile([.5,0.,0.],(n,1)),atol=1e-7)
    np.testing.assert_allclose(r['observation'][:,9:11],np.stack([np.sin(r['angle']),np.cos(r['angle'])],1),atol=2e-6)
    slot=next(i for i,s in enumerate(m['slots']) if s['train_seed']==row['train_seed'] and s['eval_seed']==row['eval_seed'])
    initial=read(folder/f'{row["scenario"]}_initial.json')
    close(r['position'][0],initial['initial']['position'][slot]);close(r['quaternion'][0],initial['initial']['quaternion'][slot])
    geoms=r['points_geom'];points=r['points_pos'];aids=set(initial['assigned'][slot]);bounds=initial['bounds'][slot]
    feet=set(m['foot_geom_ids']);ground=set(m['terrain_geom_ids']);valid=[];interior=[]
    for k,(g0,g1) in enumerate(geoms):
     valid.append(((g0 in feet and g1 in ground) or (g1 in feet and g0 in ground)) and r['points_dist'][k]<r['points_includemargin'][k] and np.any(r['points_efc_address'][k]>=0))
     g=g1 if g0 in feet else g0;x,y=points[k,:2];xlo,xhi,ylo,yhi=bounds
     interior.append(g in aids and xlo+.25<x<xhi-.25 and ylo+.25<y<yhi-.25)
    assert all(valid);interior=np.asarray(interior,bool);assert np.array_equal(interior,r['points_interior'])
    offsets=r['contact_offsets'];assert offsets.shape==(n+1,) and offsets[0]==0 and offsets[-1]==len(points) and np.all(np.diff(offsets)>=0)
    assert np.array_equal(np.diff(offsets),r['foot_contact_total'])
    cum=np.r_[0,np.cumsum(interior.astype(int))];assert np.array_equal(np.diff(cum[offsets]),r['foot_contact_interior'])
    fraction=float(interior.mean()) if len(interior) else None
    completed=not fell and n==1000 and displacement>=5 and moving>=.8 and (not terrain or fraction is not None and fraction>=.95)
    scores=dict(fell=fell,completed=bool(completed),mae_forward_velocity_mps=mae,forward_displacement_m=displacement,tilt_rms_rad=tilt,abs_heading_change_rad=heading,moving_fraction=moving,rough_contact_fraction=fraction,foot_contact_total=len(points),foot_contact_interior=int(interior.sum()))
    for k,vv in scores.items():close(vv,row[k])
    rows.append(dict(row,**scores));sampling+=n
   assert sum(x['observed_steps'] for x in rr)==m['transitions'];worldsteps+=m['simulator_world_steps']
 expected={(m,z,s,e,q) for m in METHODS for z in MAPS for s in SEEDS for e in EVAL for q in ('steady','push')}
 keys=[tuple(r[k] for k in ('method','terrain_seed','train_seed','eval_seed','scenario')) for r in rows]
 assert len(keys)==len(set(keys))==480 and set(keys)==expected
 metrics=('completed','fell','mae_forward_velocity_mps','forward_displacement_m','tilt_rms_rad','abs_heading_change_rad')
 domains={label:{m:{k:[statistics.mean(float(r[k]) for r in rows if r['method']==m and r['train_seed']==s and (r['terrain_seed']!=0)==rough) for s in SEEDS] for k in metrics} for m in METHODS} for label,rough in [('rough',True),('flat',False)]}
 means=domains['rough'];family={}
 for name,weights in {'RE_minus_RC':{'RE':1,'RC':-1},'RE_minus_FE':{'RE':1,'FE':-1},'RC_minus_FC':{'RC':1,'FC':-1},'interaction':{'RE':1,'FE':-1,'RC':-1,'FC':1}}.items():
  for metric in ('completed','fell'):family[name+':'+metric]=contrast([sum(w*means[m][metric][i] for m,w in weights.items()) for i in range(3)])
 for v,p in zip(family.values(),holm([v['p_exact'] for v in family.values()])):v['p_holm']=p
 for key,vals in family.items():
  for k,v in vals.items():close(v,analysis['statistics']['family'][key][k])
 for label in domains:
  for m in METHODS:
   for k in metrics:close(domains[label][m][k],analysis['statistics']['domain_seed_means'][label][m][k])
 def diff(label,left,right):return {k:np.array(domains[label][left][k])-domains[label][right][k] for k in metrics}
 def point(d):return dict(completion_margin=bool(d['completed'].mean()>=.10-1e-12),fall_margin=bool(d['fell'].mean()<=-.10+1e-12),completion_seed_direction=bool(sum(d['completed']>0)>=2),fall_seed_direction=bool(sum(d['fell']<0)>=2))
 limits={'mae_forward_velocity_mps':.02,'forward_displacement_m':-.5,'tilt_rms_rad':.02,'abs_heading_change_rad':.2}
 def protection(d,lim):return {k:bool(d[k].mean()>=v-1e-12 if v<0 else d[k].mean()<=v+1e-12) for k,v in lim.items()}
 gates={}
 for other in ('RC','FC'):
  d=diff('rough','RE',other);parts={**point(d),**protection(d,limits)}
  assert parts==analysis['statistics']['candidate']['comparators'][other]['gates']
  gates[other]=all(parts.values())
 flat=protection(diff('flat','RE','FE'),dict(completed=-.10,fell=.10,**limits))
 assert flat==analysis['statistics']['candidate']['flat_guards']
 assert (all(gates.values()) and all(flat.values()))==analysis['statistics']['candidate']['passed']
 for phase,left,right in [('learned','RE','FE'),('clock','RC','FC')]:
  assert all(point(diff('rough',left,right)).values())==analysis['statistics']['training_domain_effect'][phase]['point_gate_passed']
 assert sampling==analysis['effective_eval_transitions']<=480000
 result=dict(status='PASS',at=datetime.now(timezone.utc).isoformat(),analysis_sha256=sha(O/'analysis.json'),raw_count=480,raw_checks=raw_checks,effective_eval_transitions=sampling,simulator_world_steps=worldsteps,formal_training_transitions=4626432,formal_PPO_updates=3012,formal_Adam_steps=60240,evaluation_PPO_updates=0,evaluation_Adam_steps=0,independent_arithmetic='same operator; independent CPU raw/contact/seed/state budget arithmetic; no production scorer/analysis imports; not new seeds or a second reviewer',candidate_comparator_pass=gates,flat_guards=flat,scope='finite 2x2 adaptation of three reused source lineages; not independent or feedback causal confirmation')
 (O/'independent_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
 print(json.dumps({k:v for k,v in result.items() if k!='raw_checks'},ensure_ascii=False))
 return result


if __name__=='__main__':run()
