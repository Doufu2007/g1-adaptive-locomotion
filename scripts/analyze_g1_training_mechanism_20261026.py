"""All-condition gait chronology and locked Gaussian deployment contrasts."""
import argparse
from collections import defaultdict
import json

import numpy as np
from scipy.stats import t
from statsmodels.stats.multitest import multipletests

from scripts.audit_g1_training_mechanism_20261026 import ROOT, OUT, FEEDBACK, SEEDS, read, sha, longpath, write
from scripts import diagnose_g1_policy_sampling_20261026 as experiment
from scripts.g1_stateful_phase import contact_labels
from scripts.summarize_g1_best_phase_control import score
from scripts.analyze_g1_phase_feedback_probe_20261026 import supplement


def chronology():
    source=ROOT/'results/g1_best_e_cause_audit_20261026/phase_intervention_analysis.json'
    analysis=read(source)
    conditions={'holdout_clock_clock':('clock',501),'holdout_learned_raw':('learned',501),
                'midpoint_clock':('clock',251),'midpoint_learned':('learned',251)}
    rows=[]
    for record in analysis['episodes']:
        if record['scenario']!='steady' or record['condition'] not in conditions:
            continue
        path=ROOT/record['raw_path']
        assert sha(path)==record['raw_sha256']
        with np.load(longpath(path)) as raw:
            mask=raw['time']>=3
            amplitude=np.percentile(raw['joint_pos'][mask],95,axis=0)-np.percentile(raw['joint_pos'][mask],5,axis=0)
            _,_,_,cycles=contact_labels(raw['contact'],debounce=3)
            travel=[float(raw['velocity'][s:e,0].sum())*.02 for s,e,_ in cycles
                    if raw['time'][s]>=3 and .15<=raw['time'][e]-raw['time'][s]<=2]
            arm,updates=conditions[record['condition']]
            rows.append(dict(train_seed=record['train_seed'],arm=arm,updates=updates,
                eval_seed=record['eval_seed'],raw_path=str(path),raw_sha256=record['raw_sha256'],
                full20_mae_mps=record['mae_forward_velocity_mps'],
                after3_mean_velocity_mps=float(raw['velocity'][mask,0].mean()),
                cycle_forward_travel_m=float(np.mean(travel)),
                hip_pitch_90pct_range_rad=float(amplitude[[0,6]].mean()),
                knee_90pct_range_rad=float(amplitude[[3,9]].mean())))
    assert len(rows)==120
    aggregates=[]
    for seed in SEEDS:
        for arm in ('clock','learned'):
            for updates in (251,501):
                selected=[r for r in rows if (r['train_seed'],r['arm'],r['updates'])==(seed,arm,updates)]
                assert len(selected)==10
                aggregates.append(dict(train_seed=seed,arm=arm,updates=updates,episodes=10,
                    **{k:float(np.mean([r[k] for r in selected])) for k in (
                        'full20_mae_mps','after3_mean_velocity_mps','cycle_forward_travel_m',
                        'hip_pitch_90pct_range_rad','knee_90pct_range_rad')}))
    write('gait_chronology.json',dict(status='COMPLETED',classification='POST_RESULTS_EXPLORATORY',
        source_sha256=sha(source),script_sha256=sha(__file__),episodes=rows,aggregates=aggregates,
        scope='All corresponding steady midpoint/final episodes; descriptive, not checkpoint selection or causal mediation.'))
    print(json.dumps(aggregates,indent=2),flush=True)


def effect(values):
    values=np.asarray(values,dtype=float)
    mean,sd=float(values.mean()),float(values.std(ddof=1))
    se=sd/np.sqrt(3)
    half=float(t.ppf(.975,2)*se)
    p=float(2*t.sf(abs(mean/se),2)) if se else (1. if mean==0 else 0.)
    return dict(per_training_seed=dict(zip(map(str,SEEDS),values.tolist())),mean_difference_mps=mean,
                paired_t_95pct_ci_mps=[mean-half,mean+half],two_sided_p=p,
                standardized_paired_effect_dz=mean/sd if sd else None)


def sampling():
    protocol=experiment.lock()
    assert read(experiment.OUT/'queue_status.json')['status']=='COMPLETED'
    rows,manifests,starts={}, {}, {}
    episodes,integrity=[],[]
    for seed in SEEDS:
        for arm in ('clock','learned'):
            for mode in ('deterministic','sampled'):
                name=f'{arm}_{mode}_{seed}'
                folder=experiment.OUT/'rollouts'/name
                manifest,scored,raws=score(folder)
                assert manifest['episode_count']==3 and manifest['experiment_script_sha256']==protocol['script_sha256']
                assert manifest['encoder_sha256']==protocol['E_sha256']
                manifests[(seed,arm,mode)]=manifest
                sigma=np.asarray(manifest['action_sigma'],dtype=np.float32)
                for row in scored:
                    raw=raws[('flat','steady',row['seed'])]
                    epsilon=np.random.default_rng(np.random.SeedSequence([20262910,seed,row['seed']])).standard_normal((1000,29)).astype(np.float32)[:len(raw['time'])]
                    np.testing.assert_array_equal(raw['standard_normal'],epsilon)
                    np.testing.assert_array_equal(raw['actions'],raw['mean_actions']+manifest['noise_scale']*sigma*epsilon)
                    np.testing.assert_array_equal(raw['observation'][:,9:11],raw['phase_vector'])
                    np.testing.assert_array_equal(raw['episode_step'],np.arange(len(raw['time'])))
                    padded=np.zeros(1000,dtype=float)
                    padded[:len(raw['time'])]=raw['velocity'][:,0]
                    assert abs(float(np.abs(padded-.5).mean())-row['mae_forward_velocity_mps'])<1e-10
                    row=supplement(raw,dict(row,train_seed=seed,mode=mode,condition=f'{arm}_{mode}'))
                    episodes.append(row)
                    key=(seed,arm,mode,row['seed'])
                    rows[key]=row
                    starts[key]={k:raw[k][0].copy() for k in ('observation','position','quaternion','joint_pos','velocity')}
    for seed in SEEDS:
        base=manifests[(seed,'clock','deterministic')]
        for arm in ('clock','learned'):
            for mode in ('deterministic','sampled'):
                m=manifests[(seed,arm,mode)]
                assert m['startup_model_hashes']==base['startup_model_hashes']
                assert m['common_configuration_sha256']==base['common_configuration_sha256']
            for eval_seed in experiment.EVAL:
                a,b=starts[(seed,arm,'deterministic',eval_seed)],starts[(seed,arm,'sampled',eval_seed)]
                errors={k:float(np.max(np.abs(a[k].astype(float)-b[k].astype(float)))) for k in a}
                assert max(errors.values())<=1e-6
                integrity.append(dict(train_seed=seed,arm=arm,eval_seed=eval_seed,initial_errors=errors))
    values={}
    for arm in ('clock','learned'):
        values[arm]=[float(np.mean([rows[(seed,arm,'sampled',es)]['mae_forward_velocity_mps']
                                   -rows[(seed,arm,'deterministic',es)]['mae_forward_velocity_mps']
                                   for es in experiment.EVAL])) for seed in SEEDS]
    contrasts=[dict(name='E_sampled_minus_deterministic',**effect(values['learned'])),
               dict(name='clock_sampled_minus_deterministic',**effect(values['clock'])),
               dict(name='phase_by_sampling_interaction',**effect(np.asarray(values['learned'])-values['clock']))]
    for c,q in zip(contrasts,multipletests([c['two_sided_p'] for c in contrasts],method='fdr_bh')[1]):
        c['q_bh_three_contrasts']=float(q)
    aggregates=[]
    metrics=('mae_forward_velocity_mps','mean_observed_forward_velocity_mps','tilt_rms_rad',
             'abs_heading_change_rad','initial_direction_displacement_m')
    for seed in (*SEEDS,'all'):
        for arm in ('clock','learned'):
            for mode in ('deterministic','sampled'):
                chosen=[r for r in episodes if r['arm']==arm and r['mode']==mode
                        and (seed=='all' or r['train_seed']==seed)]
                aggregates.append(dict(train_seed=seed,arm=arm,mode=mode,episodes=len(chosen),falls=sum(r['fell'] for r in chosen),
                    **{k:float(np.mean([r[k] for r in chosen])) for k in metrics}))
    old=read(FEEDBACK/'analysis.json')
    repeat=[]
    for row in episodes:
        if row['mode']!='deterministic':
            continue
        condition='learned_live' if row['arm']=='learned' else 'clock_clock'
        original=next(r for r in old['episodes'] if (r['train_seed'],r['seed'],r['scenario'],r['condition'])
                      ==(row['train_seed'],row['seed'],'steady',condition))
        repeat.append(dict(train_seed=row['train_seed'],arm=row['arm'],eval_seed=row['seed'],
            new_minus_previous_mae_mps=row['mae_forward_velocity_mps']-original['mae_forward_velocity_mps']))
    result=dict(status='COMPLETED',protocol_sha256=sha(experiment.OUT/'prospective_protocol.json'),
        script_sha256=sha(__file__),verified_episodes=len(episodes),pairing_checks=integrity,
        actual_noise_and_phase_injection_verified=True,all_raw_sha_verified=True,
        contrasts=contrasts,aggregates=aggregates,episodes=episodes,deterministic_repeat=repeat,
        inference='Exploratory effects for three fixed trained policies per arm and fixed startup cohort. '
                  'The conditional action distribution is restored, not the training observation/physics domain. '
                  'No exact physical replay identity is claimed; n=3 normality is untestable.')
    write('sampling_analysis.json',result)
    print(json.dumps(dict(contrasts=contrasts,aggregates=aggregates),indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=['chronology','sampling'],required=True)
    args=parser.parse_args()
    chronology() if args.stage=='chronology' else sampling()
