"""Fresh-map confirmation contract; no sampling on import."""
import math
import numpy as np
from scripts import g1_rough_transfer_contract_20261028 as r

METHODS=('RE_live','RE_frozen','RC_clock','FC_clock')
BASE_METHOD={'RE_live':'RE','RE_frozen':'RE','RC_clock':'RC','FC_clock':'FC'}
TRAIN_SEEDS=(20261901,20261902,20261903)
TERRAINS=(20264901,20264902,20264903)
EVAL_SEEDS=(20264911,20264912,20264913,20264914,20264915)
SCENARIOS=('steady','push')
GUARDS=dict(r.GUARDS)
FLAT_GUARDS=dict(r.GUARDS)

def base_method(method):
    if method not in METHODS: raise ValueError('outside confirmation methods')
    return BASE_METHOD[method]

def source_phase(method):
    return 'learned' if method.startswith('RE') else 'clock'

def world_slots():
    return [dict(train_seed=s,eval_seed=e,patch_col=i) for s in TRAIN_SEEDS for i,e in enumerate(EVAL_SEEDS)]

def build_jobs():
    return [dict(method=m,terrain_seed=t) for m in METHODS for t in TERRAINS]

def array_digest(values): return r.array_digest(values)
def contact_exposure(*args,**kwargs): return r.contact_exposure(*args,**kwargs)
def episode_metrics(*args,**kwargs): return r.episode_metrics(*args,**kwargs)
def termination_from_raw(*args,**kwargs): return r.termination_from_raw(*args,**kwargs)
def route_actions(*args,**kwargs): return r.route_actions(*args,**kwargs)

def seed_comparison(values): return r.seed_comparison(values)
def holm(values): return r.holm(values)

def paired_analysis(rows):
    """Recompute confirmation contrasts from episode rows only."""
    expected={(j['method'],j['terrain_seed'],s['train_seed'],s['eval_seed'],q)
              for j in build_jobs() for s in world_slots() for q in SCENARIOS}
    by={}
    metrics=('completed','fell',*GUARDS)
    for row in rows:
        key=tuple(row[k] for k in ('method','terrain_seed','train_seed','eval_seed','scenario'))
        if key in by or not all(math.isfinite(float(row[m])) for m in metrics):
            raise ValueError('duplicate or nonfinite confirmation episode')
        by[key]=row
    if set(by)!=expected: raise ValueError('missing or extra confirmation episodes')
    means={method:{metric:[float(np.mean([by[method,t,s,e,q][metric]
        for t in TERRAINS for e in EVAL_SEEDS for q in SCENARIOS]))
        for s in TRAIN_SEEDS] for metric in metrics} for method in METHODS}
    contrasts={
        'RE_live_minus_RC_clock':{'RE_live':1,'RC_clock':-1},
        'RE_live_minus_RE_frozen':{'RE_live':1,'RE_frozen':-1},
    }
    family={}
    for name,weights in contrasts.items():
        for metric in ('completed','fell'):
            values=sum(weight*np.asarray(means[method][metric]) for method,weight in weights.items())
            family[name+':'+metric]=seed_comparison(values)
    adjusted=holm([item['p_exact'] for item in family.values()])
    for item,p in zip(family.values(),adjusted): item['p_holm']=p
    candidates={}
    for other in ('RC_clock','RE_frozen'):
        diffs={metric:np.asarray(means['RE_live'][metric])-means[other][metric] for metric in metrics}
        gates={
            'completion_margin':bool(diffs['completed'].mean()>=.10-1e-12),
            'fall_margin':bool(diffs['fell'].mean()<=-.10+1e-12),
            'completion_seed_direction':bool(np.sum(diffs['completed']>0)>=2),
            'fall_seed_direction':bool(np.sum(diffs['fell']<0)>=2),
        }
        for metric,limit in GUARDS.items():
            gates[metric]=bool(diffs[metric].mean()>=limit-1e-12 if limit<0 else diffs[metric].mean()<=limit+1e-12)
        candidates[other]=dict(gates=gates,passed=all(gates.values()),
            differences={metric:seed_comparison(values) for metric,values in diffs.items()})
    return dict(independent_training_seeds=3,rough_seed_means=means,family=family,
        candidate=dict(comparators=candidates,passed=all(v['passed'] for v in candidates.values()),
            meaning='fresh-map/eval confirmation with matched live-vs-frozen feedback ablation'),
        inference='conditional on three frozen S source lineages and the pre-registered fresh rough maps/seeds')
