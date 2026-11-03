"""Fresh namespace adapter for immutable R rollout; never rebind R globals."""
from pathlib import Path
from types import SimpleNamespace,FunctionType
import numpy as np
from scripts import g1_rough_domain_contract_20261028 as c
from scripts import g1_rough_domain_experiment_20261028 as ex
from scripts import g1_rough_transfer_experiment_20261028 as r

a=ex.a
ROUTE_ACTIONS=c.r.route_actions


def policy_entry(method,source_seed,smoke=False,base=None):
    c.training_map(method,source_seed)
    base=ex.OUT if base is None else Path(base)
    folder=base/('smoke_training' if smoke else 'training')/f'{method}_{source_seed}'
    m=a.read(folder/'manifest.json');target=2 if smoke else 251
    if m.get('status')!='COMPLETED' or m.get('updates')!=target or m.get('transitions')!=target*1536:raise ValueError('incomplete S final policy')
    if m['protocol_sha256']!=a.sha(base/'prospective_protocol.json'):raise ValueError('S model protocol mismatch')
    entry=m['checkpoints'][str(target)]
    if a.sha(entry['path'])!=entry['sha256'] or a.sha(entry['policy'])!=entry['policy_sha256']:raise ValueError('S policy hash changed')
    return dict(method=method,train_seed=source_seed,phase=c.source_phase(method),completed_updates=target,
        checkpoint=entry['path'],checkpoint_sha256=entry['sha256'],policy=entry['policy'],policy_sha256=entry['policy_sha256'],
        training_manifest=str(folder/'manifest.json'),training_manifest_sha256=a.sha(folder/'manifest.json'))


def make_adapter(base,verify,entry):
    # Functions retain code/defaults but use an independent explicit namespace.
    ns=dict(vars(r));ns.update(OUT=Path(base),lock=verify,policy_entry=entry)
    contract=dict(vars(c.r));contract.update(METHODS=c.METHODS,TRAIN_SEEDS=c.TRAIN_SEEDS,TERRAINS=c.TERRAINS,
        SCENARIOS=c.SCENARIOS,world_slots=c.slots,build_jobs=c.evaluation_jobs,paired_analysis=c.paired_analysis)
    ns['c']=SimpleNamespace(**contract)
    for name,value in vars(r).items():
        if isinstance(value,FunctionType) and value.__module__==r.__name__:
            ns[name]=FunctionType(value.__code__,ns,name,value.__defaults__,value.__closure__)
    ns.update(lock=verify,policy_entry=entry)
    return SimpleNamespace(**ns)


def guarded_numpy(module,guard):
    class GuardedNumpy:
        def __getattr__(self,name):return getattr(module,name)
        def savez_compressed(self,*args,**kwargs):
            guard();return module.savez_compressed(*args,**kwargs)
    return GuardedNumpy()


def evaluate(method,terrain_seed,smoke=False):
    c.source_phase(method)
    if terrain_seed not in c.TERRAINS:raise ValueError('outside S evaluation maps')
    ex.check_disk()
    adapter=make_adapter(ex.OUT,ex.verify_protocol,lambda m,s:policy_entry(m,s,smoke))
    native_physics=adapter.physical_snapshot
    def physics(env):
        ex.check_disk()
        value=native_physics(env)
        ex.record_map(terrain_seed,value)
        return value
    ns=adapter.rollout.__globals__;ns['physical_snapshot']=physics
    native_contacts=ns['read_contacts'];calls=0
    def contacts(*args,**kwargs):
        nonlocal calls
        if calls%50==0:ex.check_disk()
        calls+=1
        return native_contacts(*args,**kwargs)
    ns['read_contacts']=contacts
    # Keep np module itself unchanged: only this namespace's large raw writes
    # receive the disk guard, including each terminal episode and physics file.
    ns['np']=guarded_numpy(np,ex.check_disk)
    adapter.rollout(method,terrain_seed,smoke)


def analyze():
    adapter=make_adapter(ex.OUT,ex.verify_protocol,policy_entry)
    # Validate all S training/model bindings before any COMPLETED analysis write.
    for j in c.training_jobs():ex.verify_training(j['method'],j['source_seed'])
    for j in c.evaluation_jobs():
        folder=ex.OUT/'rollouts'/f'{j["method"]}_terrain{j["terrain_seed"]}'
        m=a.read(folder/'manifest.json')
        for p in m['policies']:
            if p!=policy_entry(j['method'],p['train_seed']):raise ValueError('eval bound to wrong S final model')
    # R's raw SHA/contact/termination/initial/metric checks stay unchanged.
    adapter.analyze()
    value=a.read(ex.OUT/'analysis.json')
    value.update(formal_training_transitions=4626432,formal_PPO_updates=3012,formal_Adam_steps=60240,
        limitation='exploratory 2x2 domain adaptation of three reused D501 source lineages; no fresh-seed or feedback-causal confirmation')
    a.write(ex.OUT/'analysis.json',value)
    return value
