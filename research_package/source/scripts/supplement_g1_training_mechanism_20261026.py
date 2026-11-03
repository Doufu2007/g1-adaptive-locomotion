"""Descriptive policy drift, inherited normalization and exploration diagnostics."""
import json
from pathlib import Path

import numpy as np
import torch

from scripts.audit_g1_training_mechanism_20261026 import (
    ROOT, OUT, FEEDBACK, TRAIN, SEEDS, forward, longpath, pool, read, sha, write,
)


def gaussian_kl(old_mean, old_sigma, mean, sigma):
    return (torch.log(sigma/old_sigma)+(old_sigma.square()+(old_mean-mean).square())
            /(2*sigma.square())-.5).sum(dim=-1)


def run():
    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    audit=read(OUT/'training_audit.json')
    obs,_=pool()
    source=ROOT/'logs/rsl_rl/g1_velocity/2026-08-10_09-35-45_baseline_full/model_9900.pt'
    source_state=torch.load(longpath(source),map_location='cpu',weights_only=False)['actor_state_dict']
    source_actions=forward(source_state,obs)
    source_std=source_state['distribution.std_param']
    records=[]
    for record in audit['checkpoints']:
        assert sha(record['checkpoint'])==record['checkpoint_sha256']
        state=torch.load(longpath(record['checkpoint']),map_location='cpu',weights_only=False)['actor_state_dict']
        actions=forward(state,obs)
        kl=gaussian_kl(source_actions,source_std,actions,state['distribution.std_param'])
        hybrid=dict(state)
        for key in ('obs_normalizer._mean','obs_normalizer._std','obs_normalizer._var','obs_normalizer.count'):
            hybrid[key]=source_state[key]
        norm_delta=actions-forward(hybrid,obs)
        records.append(dict(train_seed=record['train_seed'],arm=record['arm'],updates=record['updates'],
            source_to_checkpoint_conditional_KL_mean=float(kl.mean()),
            source_to_checkpoint_conditional_KL_median=float(kl.median()),
            normalizer_change_only_action_rms=float(torch.sqrt(norm_delta.square().mean())),
            total_action_change_rms=record['action_rms_change_from_source']))
    # Real deterministic trajectories: noise scale relative to across-time means.
    ratios=[]
    for row in read(FEEDBACK/'analysis.json')['episodes']:
        if row['scenario']!='steady' or row['condition'] not in ('clock_clock','learned_live'):
            continue
        state=torch.load(longpath(TRAIN/f"flat_{row['arm']}_seed{row['train_seed']}"/'model_500.pt'),map_location='cpu',weights_only=False)['actor_state_dict']
        with np.load(longpath(ROOT/row['raw_path'])) as raw:
            action=raw['actions'][raw['time']>=3].astype(float)
        sigma=state['distribution.std_param'].numpy()
        groups={}
        for name,indices in (('all',list(range(29))),('legs',list(range(12))),('hip_pitch',[0,6]),('knee',[3,9])):
            variation=float(np.sqrt(action[:,indices].var(axis=0).mean()))
            noise=float(np.sqrt(np.square(sigma[indices]).mean()))
            groups[name]=dict(mean_action_temporal_std_rms=variation,training_noise_std_rms=noise,
                              noise_to_mean_action_variation=noise/variation)
        ratios.append(dict(train_seed=row['train_seed'],arm=row['arm'],eval_seed=row['seed'],
                           raw_sha256=row['raw_sha256'],groups=groups))
    write('policy_drift_supplement.json',dict(status='COMPLETED',classification='POST_RESULTS_EXPLORATORY',
        script_sha256=sha(__file__),training_audit_sha256=sha(OUT/'training_audit.json'),
        observation_pool_shape=list(obs.shape),checkpoint_records=records,noise_ratios=ratios,
        scope='KL measured on the common audit state pool, not logged training minibatches. '
              'Normalizer replacement is an offline conditional diagnostic, not a deployed policy intervention. '
              'Action variation/noise ratios do not establish a benefit from stochastic deployment.'))
    print(json.dumps(dict(status='COMPLETED',first_update=[r for r in records if r['updates']==1],
                          final_normalizer=[r for r in records if r['updates']==501]),indent=2),flush=True)


if __name__=='__main__':
    run()
