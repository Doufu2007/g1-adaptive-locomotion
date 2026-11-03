"""Independent CPU arithmetic, no production scorer or analysis imports."""
from pathlib import Path
import hashlib,itertools,json,math,statistics
import numpy as np
from scipy.stats import t
from datetime import datetime,timezone
O=Path(__file__).resolve().parent
SEEDS=(20261901,20261902,20261903);EVAL=tuple(range(20264311,20264316));MAPS=(0,20264201,20264202,20264203);METHODS=('AC','AE','DC','DE')
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
protocol=read(O/'prospective_protocol.json');analysis=read(O/'analysis.json')
assert analysis['status']=='COMPLETED' and analysis['protocol_sha256']==sha(O/'prospective_protocol.json')
for f in protocol['identity']['files']:assert sha(f['path'])==f['sha256']
for p in protocol['identity']['policies']:
 for key in ('checkpoint','policy','training_manifest'):assert sha(p[key])==p[key+'_sha256']
rows=[];starts={};physics={};sampling=0;worldsteps=0;raw_checks=[]
for method in METHODS:
 for terrain in MAPS:
  folder=O/'rollouts'/f'{method}_terrain{terrain}';m=read(folder/'manifest.json');rr=read(folder/'episodes.json')
  assert m['status']=='COMPLETED' and m['protocol_sha256']==sha(O/'prospective_protocol.json')
  assert len(rr)==m['episode_count']==30 and m['PPO_updates']==m['Adam_steps']==0
  assert [(s['train_seed'],s['eval_seed'],s['patch_col']) for s in m['slots']]==[(s,e,i) for s in SEEDS for i,e in enumerate(EVAL)]
  expected_policies={(p['method'],p['train_seed']):p for p in protocol['identity']['policies']}
  for slot,p in zip(m['slots'],m['policies']):assert p==expected_policies[method,slot['train_seed']]
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
means={m:{k:[statistics.mean(float(r[k]) for r in rows if r['method']==m and r['train_seed']==s and r['terrain_seed']!=0) for s in SEEDS] for k in metrics} for m in METHODS}
family={}
for name,weights in {'DE_minus_DC':{'DE':1,'DC':-1},'AE_minus_AC':{'AE':1,'AC':-1},'DE_minus_AE':{'DE':1,'AE':-1},'interaction':{'DE':1,'AE':-1,'DC':-1,'AC':1}}.items():
 for metric in ('completed','fell'):family[name+':'+metric]=contrast([sum(w*means[m][metric][i] for m,w in weights.items()) for i in range(3)])
for v,p in zip(family.values(),holm([v['p_exact'] for v in family.values()])):v['p_holm']=p
for key,vals in family.items():
 for k,v in vals.items():close(v,analysis['statistics']['family'][key][k])
for m in METHODS:
 for k in metrics:close(means[m][k],analysis['statistics']['rough_seed_means'][m][k])
gates={}
for other in ('DC','AC'):
 d={k:np.array(means['DE'][k])-means[other][k] for k in metrics}
 ok=d['completed'].mean()>=.10-1e-12 and d['fell'].mean()<=-.10+1e-12 and sum(d['completed']>0)>=2 and sum(d['fell']<0)>=2
 ok=ok and d['mae_forward_velocity_mps'].mean()<=.02+1e-12 and d['forward_displacement_m'].mean()>=-.5-1e-12 and d['tilt_rms_rad'].mean()<=.02+1e-12 and d['abs_heading_change_rad'].mean()<=.2+1e-12
 gates[other]=bool(ok)
assert all(gates.values())==analysis['statistics']['candidate']['passed']
assert sampling==analysis['effective_eval_transitions']<=480000
result=dict(status='PASS',at=datetime.now(timezone.utc).isoformat(),analysis_sha256=sha(O/'analysis.json'),raw_count=480,raw_checks=raw_checks,effective_eval_transitions=sampling,simulator_world_steps=worldsteps,PPO_updates=0,Adam_steps=0,independent_arithmetic='same operator; own CPU raw scorer/contact/seed arithmetic, no production scoring/analysis imports; not new training seeds or second reviewer',candidate_comparator_pass=gates,scope='15 shared rough patches, three training seeds, exploratory only; no causal feedback benefit confirmation')
(O/'independent_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps({k:v for k,v in result.items() if k!='raw_checks'},ensure_ascii=False))
