"""Exact action-rate algebra on the complete locked policy sampling cohort."""
import json
import numpy as np
from scripts.audit_g1_training_mechanism_20261026 import OUT,read,sha,longpath,write
from scripts import diagnose_g1_policy_sampling_20261026 as experiment


def decompose(mean,noise,mask):
    dm,dn=np.diff(mean.astype(float),axis=0),np.diff(noise.astype(float),axis=0)
    terms=dict(mean=-.05*np.sum(dm*dm,axis=1),noise=-.05*np.sum(dn*dn,axis=1),
               cross=-.1*np.sum(dm*dn,axis=1),total=-.05*np.sum((dm+dn)**2,axis=1))
    np.testing.assert_allclose(terms['mean']+terms['noise']+terms['cross'],terms['total'],atol=1e-12,rtol=1e-12)
    return {k:float(v[mask[1:]].mean()) for k,v in terms.items()}


def run():
    gold=decompose(np.array([[0],[1],[1]]),np.array([[0],[1],[-1]]),np.ones(3,dtype=bool))
    for k,v in dict(mean=-.025,noise=-.125,cross=-.05,total=-.2).items():
        np.testing.assert_allclose(gold[k],v,atol=1e-12)
    analysis=read(OUT/'sampling_analysis.json');assert analysis['verified_episodes']==36
    rows=[]
    for row in analysis['episodes']:
        directory=experiment.OUT/'rollouts'/f"{row['arm']}_{row['mode']}_{row['train_seed']}"
        m=read(directory/'manifest.json')
        path=experiment.ROOT/row['raw_path'];assert sha(path)==row['raw_sha256']
        with np.load(longpath(path)) as raw:
            mean=raw['mean_actions'].astype(float)
            noise=raw['actions'].astype(float)-mean
            # Actual injected noise includes the float32 rounding in stored actions.
            terms=decompose(mean,noise,raw['time']>=3.)
            actual=-.05*np.sum(np.diff(raw['actions'].astype(float),axis=0)**2,axis=1)
            np.testing.assert_allclose(terms['total'],actual[(raw['time']>=3.)[1:]].mean(),atol=1e-12)
            sigma=np.asarray(m['action_sigma'])
            rows.append(dict(train_seed=row['train_seed'],arm=row['arm'],mode=row['mode'],
                eval_seed=row['seed'],raw_sha256=row['raw_sha256'],after3_action_rate_reward=terms,
                independent_noise_only_expectation=-.1*float(np.square(sigma).sum())*m['noise_scale']**2))
    groups=[]
    for seed in (*experiment.SEEDS,'all'):
        for arm in ('clock','learned'):
            for mode in ('deterministic','sampled'):
                chosen=[r for r in rows if r['arm']==arm and r['mode']==mode and
                        (seed=='all' or r['train_seed']==seed)]
                groups.append(dict(train_seed=seed,arm=arm,mode=mode,episodes=len(chosen),
                    **{k:float(np.mean([r['after3_action_rate_reward'][k] for r in chosen])) for k in gold},
                    independent_noise_only_expectation=float(np.mean([r['independent_noise_only_expectation'] for r in chosen]))))
    report=dict(status='COMPLETED',script_sha256=sha(__file__),
        sampling_analysis_sha256=sha(OUT/'sampling_analysis.json'),gold_algebra_passed=True,
        episodes=rows,aggregates=groups,
        scope='Exact finite-difference decomposition of recorded action-rate penalty after3s, excluding first action. '
              'Mean actions depend on earlier noise; cross term is retained, not assumed independent or zero. '
              'This is a different cohort from training and does not recover the full training reward or its gradients. '
              'Noise dominance in this term does not establish E-specific harm or causal training dependence.')
    write('action_noise_analysis.json',report)
    print(json.dumps([r for r in groups if r['train_seed']=='all'],indent=2),flush=True)


if __name__=='__main__':
    run()
