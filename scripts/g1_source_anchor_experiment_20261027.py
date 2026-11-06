"""Frozen source-KL intervention; reuse audited stage-A controls without editing them."""
from dataclasses import asdict
from pathlib import Path
import argparse,concurrent.futures,copy,gc,hashlib,json,math,os,random,statistics,subprocess,sys,time,traceback
from scripts import g1_phase_lr_factorial_20261027 as a
from scripts import g1_contact_reference_experiment_20261027 as b
from scripts import g1_phase_policy_cross_20261027 as c
ROOT=a.ROOT;AOUT=b.AOUT;OUT=ROOT/'results/g1_source_anchor_20261027'
SPEC=ROOT/'docs/superpowers/specs/2026-10-27-g1-source-anchor-design.md'
PLAN=ROOT/'docs/superpowers/plans/2026-10-27-g1-source-anchor.md'
ANCHOR_CLASS='scripts.g1_source_anchor_ppo_20261027:SourceAnchoredPPO'
CAPTURE=ROOT/'results/g1_ppo_initialization_probe_20261026/local_reconstruction_v3/captures'
ORIGINAL_SETUP=a.setup;BASE_CFG=None;STUDY=None;PRE_CFG={}
PREFIX="import site;site.addsitedir('<user-profile>/AppData/Roaming/Python/Python313/site-packages');import runpy,sys;"

def common_agent_digest(config):
 value=copy.deepcopy(config);name=value['algorithm']['class_name']
 if name not in ('PPO',ANCHOR_CLASS):raise ValueError('undeclared algorithm difference')
 value['algorithm']['class_name']='PPO';value['algorithm'].pop('learning_rate')
 return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()

def check_launch_seed(launch,control):
 if launch['startup_pythonhashseed']!=control['startup_pythonhashseed'] or launch['startup_pythonhashseed']!='20263300':raise ValueError('unpaired interpreter startup seed')
 if launch['launcher_hash_probe']!=control['launcher_hash_probe']:raise ValueError('unpaired interpreter hash probe')
 return True

def build_jobs():
 jobs=[]
 for seed in a.SEEDS:
  arms=['clock','learned'];random.Random(20263890+seed).shuffle(arms)
  jobs.extend(dict(seed=seed,arm=arm,updates=501,transitions=769536) for arm in arms)
 return jobs

def paired_contrasts(rows):
 groups={}
 for r in rows:
  key=(r['seed'],r['arm'],r['variant'],r['eval_seed']);value=float(r['mae_forward_velocity_mps'])
  if key in groups or not math.isfinite(value):raise ValueError('duplicate/nonfinite paired row')
  groups[key]=value
 seeds=sorted({r['seed'] for r in rows});out={k:[] for k in ('E_anchor_minus_clock_anchor','E_anchor_minus_E_base','clock_anchor_minus_clock_base','interaction')}
 if not seeds:raise ValueError('empty rows')
 for seed in seeds:
  cells=[{e:v for (s,arm,variant,e),v in groups.items() if s==seed and arm==aa and variant==vv} for aa,vv in [('clock','base'),('learned','base'),('clock','anchor'),('learned','anchor')]]
  if not cells[0] or any(set(x)!=set(cells[0]) for x in cells):raise ValueError('missing paired episode')
  cb,eb,ca,ea=[statistics.mean(x.values()) for x in cells]
  for k,v in zip(out,(ea-ca,ea-eb,ca-cb,(ea-eb)-(ca-cb))):out[k].append(v)
 return out

def dual_candidate(anchor_differences,anchor_guards,base_differences,base_guards):
 anchored=a.candidate_gate(anchor_differences,anchor_guards);original=a.candidate_gate(base_differences,base_guards)
 return dict(proceed=anchored['proceed'] and original['proceed'],versus_anchor_clock=dict(anchored,guards=anchor_guards),versus_original_clock=dict(original,guards=base_guards))

def lock():
 c.lock();assert a.read(c.OUT/'independent_audit.json')['status']=='PASS'
 assert a.sha(c.OUT/'analysis.json')==a.read(c.OUT/'queue_status.json')['analysis_sha256']
 paths=[Path(__file__),ROOT/'scripts/g1_source_anchor_ppo_20261027.py',ROOT/'tests/test_g1_source_anchor_20261027.py',SPEC,PLAN,
        c.OUT/'prospective_protocol.json',c.OUT/'analysis.json',c.OUT/'independent_audit.json',AOUT/'prospective_protocol.json',AOUT/'analysis.json',
        OUT/'technical_amendment_01.json',OUT/'technical_failure_01/prospective_protocol.json',
        OUT/'technical_amendment_02.json',OUT/'technical_failure_02/prospective_protocol.json']
 for module in ('rsl_rl.algorithms.ppo','rsl_rl.models.mlp_model','rsl_rl.modules.distribution','mjlab.utils.random'):paths.append(Path(__import__(module,fromlist=['x']).__file__))
 controls=[];baseline_raw={r['raw_path']:r['raw_sha256'] for r in a.read(AOUT/'analysis.json')['episodes']}
 for arm in ('clock','learned'):
  for seed in a.SEEDS:
   folder=AOUT/'training'/f'{arm}_low_{seed}';manifest=a.read(folder/'manifest.json')
   assert manifest['status']=='COMPLETED' and manifest['updates']==501 and manifest['transitions']==769536
   paths.append(folder/'manifest.json')
   for entry in manifest['checkpoints'].values():
    assert a.sha(entry['path'])==entry['sha256'] and a.sha(entry['policy'])==entry['policy_sha256']
   for point in a.CHECKPOINTS:
    evaluation=AOUT/'evaluation'/f'{arm}_low_{seed}_u{point}'
    paths.extend([evaluation/'manifest.json',evaluation/'episodes.json'])
    for row in a.read(evaluation/'episodes.json'):
     raw=evaluation/row['raw_path'];assert baseline_raw[str(raw.relative_to(ROOT))]==row['raw_sha256']==a.sha(raw),'original stage-A telemetry changed'
   controls.append(dict(arm=arm,seed=seed,manifest_sha256=a.sha(folder/'manifest.json'),checkpoints=manifest['checkpoints']))
   cap=CAPTURE/f'{arm}_{seed}';cm=a.read(cap/'manifest.json');assert cm['status']=='COMPLETED' and a.sha(cap/'batch.pt')==cm['batch_sha256']
   paths.extend([cap/'batch.pt',cap/'manifest.json'])
 identity=dict(files=[dict(path=str(p),sha256=a.sha(p)) for p in paths],controls=controls,jobs=build_jobs(),beta=.1,
   teacher='frozen source9900; own phase observations; no random samples',formal_transitions=4617216,formal_episodes=270,
   smoke_train_transitions=6144,smoke_evaluation_transitions=20,smoke_PPO_updates=4,smoke_Adam_steps=80,initial_lr=.00001,adaptive_schedule=True,
   checkpoints=list(a.CHECKPOINTS),primary='501 steady; 10 eval starts per training seed; body-forward full20s MAE',
   family=['E_anchor_minus_clock_anchor','E_anchor_minus_E_base','clock_anchor_minus_clock_base','interaction'],statistical_unit='3 paired training seeds',
   source_sha256=a.SOURCE_SHA,encoder_sha256=a.ENCODER_SHA,maximum_concurrent_workers=2,seed_shards=[list(a.SEEDS[i::2]) for i in range(2)],confirmatory=False)
 path=OUT/'prospective_protocol.json'
 if path.exists():
  old=a.read(path);assert old['identity']==identity,'frozen stage-D source changed';return old
 value=dict(status='LOCKED_BEFORE_SOURCE_ANCHOR_SAMPLING',at=a.now(),identity=identity);a.write(path,value)
 for p in paths:
  if p.is_relative_to(ROOT) and p.suffix in ('.py','.md'):
   target=OUT/'source_snapshot'/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(p.read_bytes())
 return value

def setup():
 global STUDY,BASE_CFG
 STUDY,torch=ORIGINAL_SETUP()
 if BASE_CFG is None:BASE_CFG=STUDY.make_cfg
 def config(terrain,arm,seed,**kw):
  cfg=BASE_CFG(terrain,arm,seed,**kw)
  if not kw['play'] and kw['num_envs']==64:PRE_CFG[seed]=STUDY.common_cfg_digest(cfg)
  return cfg
 STUDY.make_cfg=config
 return STUDY,torch

def install():a.OUT=OUT;a.setup=setup;a.lock=lock

def train(arm,seed,smoke=False):
 install()
 folder=OUT/('smoke' if smoke else 'training')/f'{arm}_low_{seed}'
 if (folder/'manifest.json').exists():
  lock();done=a.read(folder/'manifest.json')
  assert done['status']=='COMPLETED' and done['protocol_sha256']==a.sha(OUT/'prospective_protocol.json'),'incomplete attempt preserved'
  assert done['teacher_state_sha256'] and done['normalized_common_agent_sha256']
  return
 from mjlab.rl import MjlabOnPolicyRunner
 from mjlab.tasks import registry
 from scripts.g1_source_anchor_ppo_20261027 import SourceAnchoredPPO
 from scripts import diagnose_g1_initial_update_20261026 as probe
 native_cfg=registry.load_rl_cfg;native_save=MjlabOnPolicyRunner.save;native_update=SourceAnchoredPPO.update
 teacher_digest=None;actual_agent=None
 def agent_cfg(name):
  nonlocal actual_agent
  cfg=native_cfg(name);cfg.algorithm.class_name=ANCHOR_CLASS;actual_agent=cfg;return cfg
 def save(runner,path,infos=None):
  nonlocal teacher_digest
  state=(infos or {}).get('factorial_state',{});completed=state.get('completed_updates');raw=runner.env.unwrapped
  current=STUDY.base.state_sha(runner.alg.source_teacher)
  if completed==0:
   import torch
   teacher_digest=current;assert current==STUDY.base.state_sha(runner.alg.actor) and not runner.alg.optimizer.state
   assert actual_agent is not None
   common=common_agent_digest(STUDY.canonical(asdict(actual_agent)))
   observed=dict(initial_actor_sha256=STUDY.base.state_sha(runner.alg.actor),initial_critic_sha256=STUDY.base.state_sha(runner.alg.critic),
    common_cfg_sha256=PRE_CFG[seed],common_agent_sha256=common,startup_model_hashes={k:probe.digest_tensor(getattr(raw.sim.model,k)) for k in ('body_mass','body_ipos','geom_friction')})
   launch=a.read(folder/'manifest.json');assert launch['launcher_hash_probe']==hash('g1_phase_lr_hash_probe')
   for control in ('clock','learned'):
    m=a.read(AOUT/'training'/f'{control}_low_{seed}'/'manifest.json');b.check_initialization(observed,m)
    saved=torch.load(m['checkpoints']['0']['path'],map_location='cpu',weights_only=False)['infos']['factorial_state']
    assert torch.equal(saved['torch_rng'],state['torch_rng']) and len(saved['cuda_rng'])==len(state['cuda_rng'])
    assert all(torch.equal(x,y) for x,y in zip(saved['cuda_rng'],state['cuda_rng']))
    check_launch_seed(launch,m)
    assert m['backend']==dict(tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_deterministic=torch.backends.cudnn.deterministic,cudnn_benchmark=torch.backends.cudnn.benchmark)
   a.write(folder/'pre_sampling_pairing.json',dict(status='PASS',at=a.now(),transitions=0,optimizer_updates=0,fields=observed,CPU_CUDA_RNG_pairing=True,empty_Adam=True,teacher_source_sha256=current,
    interpreter_hash_pairing=True,startup_pythonhashseed=launch['startup_pythonhashseed'],launcher_hash_probe=launch['launcher_hash_probe'],runtime_seed_environment=os.environ.get('PYTHONHASHSEED')))
  assert teacher_digest==current,'frozen teacher changed'
  infos=dict(infos or {},source_anchor=dict(beta=.1,teacher_state_sha256=current))
  audit=folder/'teacher_audit.json';history=a.read(audit) if audit.exists() else {};history[str(completed)]=current;a.write(audit,history)
  return native_save(runner,path,infos)
 def update(alg):
  result=native_update(alg)
  assert len(alg.last_source_kl)==20
  with (folder/'source_updates.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(dict(minibatches=alg.last_source_kl),allow_nan=False)+'\n')
  return result
 registry.load_rl_cfg=agent_cfg;MjlabOnPolicyRunner.save=save;SourceAnchoredPPO.update=update
 try:a.train(arm,'low',seed,smoke=smoke)
 finally:registry.load_rl_cfg=native_cfg;MjlabOnPolicyRunner.save=native_save;SourceAnchoredPPO.update=native_update
 m=a.read(folder/'manifest.json');m.update(intervention='source_KL_beta_.1',teacher_state_sha256=teacher_digest,
  normalized_common_agent_sha256=a.read(folder/'pre_sampling_pairing.json')['fields']['common_agent_sha256']);a.write(folder/'manifest.json',m)

def evaluate(arm,seed,point,smoke=False):install();a.evaluate(arm,'low',seed,point,smoke=smoke)

def cell(arm,seed,smoke=False):
 train(arm,seed,smoke)
 for point in ((2,) if smoke else a.CHECKPOINTS):
  gc.collect()
  import torch
  torch.cuda.empty_cache();evaluate(arm,seed,point,smoke)

def verify_smoke():
 from scripts.summarize_g1_best_phase_control import score
 bindings=[]
 for arm in ('clock','learned'):
  folder=OUT/'smoke'/f'{arm}_low_{a.SEEDS[0]}';m=a.read(folder/'manifest.json')
  assert m['status']=='COMPLETED' and m['updates']==2 and m['transitions']==3072
  assert m['protocol_sha256']==a.sha(OUT/'prospective_protocol.json') and set(m['checkpoints'])=={'0','1','2'}
  for entry in m['checkpoints'].values():assert a.sha(entry['path'])==entry['sha256'] and a.sha(entry['policy'])==entry['policy_sha256']
  pre=a.read(folder/'pre_sampling_pairing.json');assert pre['status']=='PASS' and pre['CPU_CUDA_RNG_pairing'] and pre['empty_Adam']
  for control in ('clock','learned'):b.check_initialization(pre['fields'],a.read(AOUT/'training'/f'{control}_low_{a.SEEDS[0]}'/'manifest.json'))
  assert len(set(a.read(folder/'teacher_audit.json').values()))==1
  traces=[json.loads(x) for x in (folder/'source_updates.jsonl').read_text().splitlines()];assert len(traces)==2 and all(len(x['minibatches'])==20 for x in traces)
  evaluation=OUT/'smoke_evaluation'/f'{arm}_low_{a.SEEDS[0]}_u2'
  em,rows,raw=score(evaluation);assert len(rows)==1 and len(next(iter(raw.values()))['time'])==10 and em['horizon']==.2
  assert em['status']=='COMPLETED' and em['protocol_sha256']==m['protocol_sha256'] and em['policy_sha256']==m['checkpoints']['2']['policy_sha256']
  for path in (folder/'manifest.json',folder/'pre_sampling_pairing.json',folder/'teacher_audit.json',folder/'source_updates.jsonl',evaluation/'manifest.json'):
   bindings.append(dict(path=str(path),sha256=a.sha(path)))
 a.write(OUT/'smoke_verification.json',dict(status='PASS',at=a.now(),protocol_sha256=a.sha(OUT/'prospective_protocol.json'),files=bindings,train_transitions=6144,evaluation_transitions=20,PPO_updates=4,Adam_steps=80))

def smoke_processes():
 lock();module='scripts.g1_source_anchor_experiment_20261027'
 for arm in ('clock','learned'):
  argv=[module,'--stage','cell','--arm',arm,'--seed',str(a.SEEDS[0]),'--smoke']
  command=PREFIX+'sys.argv='+repr(argv)+';runpy.run_module('+repr(module)+',run_name="__main__")'
  subprocess.run([sys.executable,'-u','-c',command],cwd=ROOT,check=True,
   env=dict(os.environ,PYTHONHASHSEED='20263300',PYTHONIOENCODING='utf-8'),creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
 verify_smoke()

def check_smoke_gate(gate,protocol_sha256):
 if gate['status']!='PASS' or gate['protocol_sha256']!=protocol_sha256:raise ValueError('smoke gate is not current PASS')
 if len(gate['files'])!=10:raise ValueError('incomplete smoke bindings')
 for row in gate['files']:
  if a.sha(row['path'])!=row['sha256']:raise ValueError('smoke artifact changed: '+row['path'])

def worker(index):
 completed=[]
 for job in [j for j in build_jobs() if j['seed'] in a.SEEDS[index::2]]:
  arm,seed=job['arm'],job['seed'];name=f'{arm}_low_{seed}';module='scripts.g1_source_anchor_experiment_20261027'
  command=PREFIX+'sys.argv='+repr([module,'--stage','cell','--arm',arm,'--seed',str(seed)])+';runpy.run_module('+repr(module)+',run_name="__main__")'
  log=OUT/'logs'/f'{name}.log';log.parent.mkdir(parents=True,exist_ok=True)
  with log.open('a',encoding='utf-8') as stream:
   child=subprocess.Popen([sys.executable,'-u','-c',command],cwd=ROOT,stdout=stream,stderr=stream,env=dict(os.environ,PYTHONHASHSEED='20263300',PYTHONIOENCODING='utf-8'),creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
   a.write(OUT/f'parallel_worker_{index}.json',dict(status='RUNNING',at=a.now(),current=name,child_pid=child.pid,completed=completed));rc=child.wait()
  if rc:
   a.write(OUT/f'parallel_worker_{index}.json',dict(status='FAILED',at=a.now(),current=name,completed=completed,exit_code=rc,log=str(log)))
   raise RuntimeError(name+' failed '+str(rc))
  completed.append(name)
 a.write(OUT/f'parallel_worker_{index}.json',dict(status='COMPLETED',at=a.now(),completed=completed));return completed

def queue():
 lock();check_smoke_gate(a.read(OUT/'smoke_verification.json'),a.sha(OUT/'prospective_protocol.json'))
 verify_smoke()
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  futures=[pool.submit(worker,i) for i in range(2)]
  while not all(f.done() for f in futures):
   for f in futures:
    if f.done() and f.exception():raise f.exception()
   a.write(OUT/'queue_status.json',dict(status='RUNNING',at=a.now(),pid=os.getpid(),maximum_concurrent_cells=2));time.sleep(10)
  completed=[x for f in futures for x in f.result()]
 a.write(OUT/'queue_status.json',dict(status='ANALYZING',at=a.now(),completed=completed));analyze()
 a.write(OUT/'queue_status.json',dict(status='STAGE_D_COMPLETED',at=a.now(),completed=completed,analysis_sha256=a.sha(OUT/'analysis.json'),candidate=a.read(OUT/'analysis.json')['candidate']))

def guards(treatment,control):
 import numpy as np
 by={(r['seed'],r['eval_seed']):r for r in control};assert set(by)=={(r['seed'],r['eval_seed']) for r in treatment}
 avg=lambda rows,k:float(np.mean([r[k] for r in rows]))
 return dict(falls=sum(r['fell'] and not by[(r['seed'],r['eval_seed'])]['fell'] for r in treatment),tilt=avg(treatment,'tilt_rms_rad')-avg(control,'tilt_rms_rad'),
  heading=avg(treatment,'abs_heading_change_rad')-avg(control,'abs_heading_change_rad'),displacement=avg(treatment,'initial_direction_displacement_m')-avg(control,'initial_direction_displacement_m'))

def bank_drift():
 import torch
 from tensordict import TensorDict
 from rsl_rl.models import MLPModel
 rows=[]
 for arm in ('clock','learned'):
  for seed in a.SEEDS:
   capture=CAPTURE/f'{arm}_{seed}'/'batch.pt';pack=torch.load(capture,map_location='cpu',weights_only=False)
   obs=TensorDict({'actor':pack['observations']['actor'].reshape(-1,98)},batch_size=[1536])
   make=lambda:MLPModel(obs,{'actor':['actor']},'actor',29,hidden_dims=(512,256,128),activation='elu',obs_normalization=True,distribution_cfg=dict(class_name='GaussianDistribution',init_std=1.,std_type='scalar')).eval()
   teacher=make();current=make();base=a.read(AOUT/'training'/f'{arm}_low_{seed}'/'manifest.json')
   teacher.load_state_dict(torch.load(base['checkpoints']['0']['path'],map_location='cpu',weights_only=False)['actor_state_dict'])
   with torch.inference_mode():
    teacher.distribution.update(teacher.mlp(teacher.get_latent(obs)));ref=teacher.output_distribution_params
    for variant,folder in [('base',AOUT),('anchor',OUT)]:
     m=a.read(folder/'training'/f'{arm}_low_{seed}'/'manifest.json')
     for point in a.CHECKPOINTS:
      current.load_state_dict(torch.load(m['checkpoints'][str(point)]['path'],map_location='cpu',weights_only=False)['actor_state_dict'])
      current.distribution.update(current.mlp(current.get_latent(obs)));kl=current.distribution.kl_divergence(ref,current.output_distribution_params)
      rows.append(dict(arm=arm,seed=seed,variant=variant,point=point,source_kl_mean=float(kl.mean()),source_kl_max=float(kl.max()),capture_sha256=a.sha(capture),n=1536))
 return rows

def analyze():
 lock()
 import numpy as np
 from statsmodels.stats.multitest import multipletests
 from scripts.analyze_g1_phase_lr_factorial_20261027 import extra,effect,METRICS
 from scripts.summarize_g1_best_phase_control import score
 episodes=[];training=[];pairing=[];aggregates=[]
 baseline_raw={r['raw_path']:r['raw_sha256'] for r in a.read(AOUT/'analysis.json')['episodes']}
 for seed in a.SEEDS:
  for arm in ('clock','learned'):
   for variant,base in [('base',AOUT),('anchor',OUT)]:
    m=a.read(base/'training'/f'{arm}_low_{seed}'/'manifest.json');assert m['status']=='COMPLETED' and m['updates']==501 and m['transitions']==769536 and m['encoder_frozen'];training.append(dict(m,variant=variant))
    assert m['protocol_sha256']==a.sha(base/'prospective_protocol.json')
    native_traces=[json.loads(x) for x in (base/'training'/f'{arm}_low_{seed}'/'updates.jsonl').read_text().splitlines()]
    assert len(native_traces)==501 and [t['completed_update'] for t in native_traces]==list(range(1,502)) and all(len(t['minibatches'])==20 for t in native_traces)
    if variant=='anchor':
     pre=a.read(OUT/'training'/f'{arm}_low_{seed}'/'pre_sampling_pairing.json');b.check_initialization(pre['fields'],a.read(AOUT/'training'/f'{arm}_low_{seed}'/'manifest.json'))
     b.check_initialization(dict(m,common_agent_sha256=m['normalized_common_agent_sha256']),pre['fields'])
     assert pre['CPU_CUDA_RNG_pairing'] and len(set(a.read(OUT/'training'/f'{arm}_low_{seed}'/'teacher_audit.json').values()))==1
     traces=[json.loads(x) for x in (OUT/'training'/f'{arm}_low_{seed}'/'source_updates.jsonl').read_text().splitlines()];assert len(traces)==501 and all(len(t['minibatches'])==20 for t in traces)
   for point in a.CHECKPOINTS:
    scored={}
    for variant,base in [('base',AOUT),('anchor',OUT)]:
     m=a.read(base/'training'/f'{arm}_low_{seed}'/'manifest.json');entry=m['checkpoints'][str(point)]
     assert a.sha(entry['path'])==entry['sha256'] and a.sha(entry['policy'])==entry['policy_sha256']
     em,rows,raws=score(base/'evaluation'/f'{arm}_low_{seed}_u{point}');scored[variant]=(em,raws)
     expected={(sc,e) for sc in (['steady','speed_change','push'] if point==501 else ['steady']) for e in (range(20263411,20263421) if point==501 else range(20263411,20263414))}
     assert len(rows)==len(expected) and {(r['scenario'],r['seed']) for r in rows}==expected
     assert em['horizon']==20. and em['terrain']=='flat' and em['encoder_sha256']==a.ENCODER_SHA
     assert em['policy_sha256']==entry['policy_sha256'] and em['protocol_sha256']==m['protocol_sha256']
     for r in rows:
      if variant=='base':assert baseline_raw[r['raw_path']]==r['raw_sha256']
      episodes.append(extra(raws[('flat',r['scenario'],r['seed'])],dict(r,seed=seed,eval_seed=r['seed'],completed_updates=point,variant=variant)))
    assert scored['base'][0]['startup_model_hashes']==scored['anchor'][0]['startup_model_hashes']
    for key,raw in scored['anchor'][1].items():
     control=scored['base'][1][key];errors={k:float(np.max(np.abs(raw[k][0].astype(float)-control[k][0].astype(float)))) for k in ('position','quaternion','joint_pos','velocity')};assert max(errors.values())<=1e-6
     np.testing.assert_array_equal(raw['command'][:min(len(raw['command']),len(control['command']))],control['command'][:min(len(raw['command']),len(control['command']))]);pairing.append(dict(seed=seed,arm=arm,point=point,key=list(key),errors=errors))
 for point in a.CHECKPOINTS:
  for scenario in (['steady','speed_change','push'] if point==501 else ['steady']):
   for arm in ('clock','learned'):
    for variant in ('base','anchor'):
     chosen=[r for r in episodes if r['completed_updates']==point and r['scenario']==scenario and r['arm']==arm and r['variant']==variant]
     aggregates.append(dict(point=point,scenario=scenario,arm=arm,variant=variant,n=len(chosen),falls=sum(r['fell'] for r in chosen),**{k:float(np.mean([r[k] for r in chosen if r[k] is not None])) if any(r[k] is not None for r in chosen) else None for k in METRICS}))
 final=[r for r in episodes if r['completed_updates']==501 and r['scenario']=='steady'];dif=paired_contrasts(final);contrasts={k:effect(v) for k,v in dif.items()}
 for value,q in zip(contrasts.values(),multipletests([r['p'] for r in contrasts.values()],method='holm')[1]):value['holm_p']=float(q)
 treatment=[r for r in final if r['arm']=='learned' and r['variant']=='anchor'];anchor_clock=[r for r in final if r['arm']=='clock' and r['variant']=='anchor'];base_clock=[r for r in final if r['arm']=='clock' and r['variant']=='base']
 group=lambda rows,s:statistics.mean(r['mae_forward_velocity_mps'] for r in rows if r['seed']==s)
 base_diff=[group(treatment,s)-group(base_clock,s) for s in a.SEEDS]
 candidate=dual_candidate(dif['E_anchor_minus_clock_anchor'],guards(treatment,anchor_clock),base_diff,guards(treatment,base_clock))
 assert len(episodes)==540 and len([r for r in episodes if r['variant']=='anchor'])==270
 drift=bank_drift()
 result=dict(status='COMPLETED',at=a.now(),protocol_sha256=a.sha(OUT/'prospective_protocol.json'),training=training,episodes=episodes,aggregates=aggregates,pairing=pairing,contrasts=contrasts,candidate=candidate,fixed_observation_bank_drift=drift,formal_new_transitions=4617216,new_episodes=270,scope='Exploratory fixed three training seeds; no independent stable-superiority conclusion')
 a.write(OUT/'analysis.json',result)
 lines=['# 源策略KL约束单因素修正','','|更新|场景|相位|干预|MAE|航向|位移|倾斜|跌倒/n|','|---:|---|---|---|---:|---:|---:|---:|---:|']
 for r in aggregates:lines.append(f"|{r['point']}|{r['scenario']}|{r['arm']}|{r['variant']}|{r['mae_forward_velocity_mps']:.5f}|{r['abs_heading_change_rad']:.3f}|{r['initial_direction_displacement_m']:.3f}|{r['tilt_rms_rad']:.4f}|{r['falls']}/{r['n']}|")
 lines+=['','四项配对seed级比较：'+json.dumps(contrasts,ensure_ascii=False),'','双clock候选门：'+json.dumps(candidate,ensure_ascii=False),'',result['scope']];(OUT/'diagnosis.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--stage',required=True,choices=['lock','smoke','cell','queue','analyze']);parser.add_argument('--arm');parser.add_argument('--seed',type=int);parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
 try:
  if args.stage=='lock':lock()
  elif args.stage=='smoke':smoke_processes()
  elif args.stage=='cell':cell(args.arm,args.seed,args.smoke)
  elif args.stage=='queue':queue()
  else:analyze()
 except BaseException as error:
  if args.stage=='queue':a.write(OUT/'queue_status.json',dict(status='FAILED',at=a.now(),error=repr(error),traceback=traceback.format_exc()))
  raise
