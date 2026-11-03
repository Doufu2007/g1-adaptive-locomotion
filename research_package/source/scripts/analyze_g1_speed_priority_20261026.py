"""Recompute every trajectory in the locked speed-weight diagnostic."""
import json
import numpy as np
from scipy.stats import t
from scripts import diagnose_g1_speed_priority_20261026 as exp
from scripts.summarize_g1_best_phase_control import score
from scripts.analyze_g1_phase_feedback_probe_20261026 import heading
from scripts.g1_stateful_phase import contact_labels
from scripts.audit_g1_training_mechanism_20261026 import longpath


def effect(values):
    x=np.asarray(values,dtype=float);assert x.shape==(3,) and np.isfinite(x).all()
    mean,sd=float(x.mean()),float(x.std(ddof=1));se=sd/np.sqrt(3)
    half=float(t.ppf(.975,2)*se)
    return dict(per_training_seed=dict(zip(map(str,exp.SEEDS),x.tolist())),
        mean_difference_mps=mean,paired_t_95pct_ci_mps=[mean-half,mean+half],
        two_sided_p=float(2*t.sf(abs(mean/se),2)) if se else (1. if mean==0 else 0.),
        standardized_paired_effect_dz=mean/sd if sd else None,
        training_seeds=3,comparison_family_size=1,
        interpretation='Single prospectively specified new comparison, after earlier exploratory experiments; '
                       'three fixed trained policies, not nine independent training runs.')


def run():
    protocol=exp.lock();assert exp.read(exp.OUT/'queue_status.json')['status']=='COMPLETED'
    training=[]
    for seed in exp.SEEDS:
        folder=exp.OUT/'training'/f'learned_original_reward_{seed}'
        training.append(exp.verify_training(folder,seed))
        m=exp.read(folder/'manifest.json')
        assert m['mode']=='speed_priority' and m['speed_weight']==2. and m['pose_weight']==1.
        assert m['additional_completed_updates']==100 and m['collected_transitions']==153600
        assert exp.sha(folder/'policy.onnx')==m['policy_sha256']
        assert m['intervention_script_sha256']==exp.sha(exp.__file__)
    rows,starts,manifests=[],{},{}
    for seed in exp.SEEDS:
        for arm in ('original_reward','speed_priority'):
            directory=(exp.CACHE/'rollouts'/f'training_holdout_learned_original_reward_{seed}'
                if arm=='original_reward' else exp.OUT/'rollouts'/f'speed_priority_{seed}')
            manifest,scored,raws=score(directory)
            assert manifest['status']=='COMPLETED' and manifest['episode_count']==3
            assert manifest['seeds']==list(exp.EVAL)
            assert exp.sha(manifest['policy'])==manifest['policy_sha256']
            manifests[(seed,arm)]=manifest
            for row in scored:
                raw=raws[('flat','steady',row['seed'])]
                assert np.isfinite(raw['actions']).all()
                assert np.isfinite(raw['joint_pos']).all()
                assert (raw['command']==.5).all()
                padded=np.zeros(1000);padded[:len(raw['time'])]=raw['velocity'][:,0]
                np.testing.assert_allclose(abs(padded-.5).mean(),row['mae_forward_velocity_mps'],atol=1e-12)
                yaw=heading(raw);direction=np.array([np.cos(yaw[0]),np.sin(yaw[0])])
                mask=raw['time']>=3.
                amplitude=np.percentile(raw['joint_pos'][mask],95,axis=0)-np.percentile(raw['joint_pos'][mask],5,axis=0)
                _,_,_,cycles=contact_labels(raw['contact'],debounce=3)
                travel=[float(raw['velocity'][s:e,0].sum())*.02 for s,e,_ in cycles
                    if raw['time'][s]>=3 and .15<=raw['time'][e]-raw['time'][s]<=2]
                row.update(train_seed=seed,condition=arm,
                    abs_heading_change_rad=abs(float(yaw[-1]-yaw[0])),
                    initial_direction_displacement_m=float((raw['position'][-1,:2]-raw['position'][0,:2])@direction),
                    after3_mean_velocity_mps=float(raw['velocity'][mask,0].mean()),
                    cycle_forward_travel_m=float(np.mean(travel)) if travel else None,
                    hip_pitch_90pct_range_rad=float(amplitude[[0,6]].mean()),
                    knee_90pct_range_rad=float(amplitude[[3,9]].mean()))
                rows.append(row)
                starts[(seed,arm,row['seed'])]={k:raw[k][0].copy() for k in ('position','quaternion','joint_pos','velocity')}
    pairing=[]
    for seed in exp.SEEDS:
        a,b=[manifests[(seed,k)] for k in ('original_reward','speed_priority')]
        assert a['startup_model_hashes']==b['startup_model_hashes']
        assert a['cause_audit']['packages']==b['cause_audit']['packages'],'Evaluation software identity changed'
        for es in exp.EVAL:
            errors={k:float(np.max(np.abs(starts[(seed,'original_reward',es)][k].astype(float)
                         -starts[(seed,'speed_priority',es)][k].astype(float)))) for k in starts[(seed,'original_reward',es)]}
            assert max(errors.values())<=1e-6
            pairing.append(dict(train_seed=seed,eval_seed=es,initial_state_errors=errors))
        # The policy itself is the only action source that changes in evaluation.
        ad=exp.CACHE/'rollouts'/f'training_holdout_learned_original_reward_{seed}'
        bd=exp.OUT/'rollouts'/f'speed_priority_{seed}'
        assert exp.sha(ad/'env.yaml')==exp.sha(bd/'env.yaml'),'Evaluation configuration drift'
    metrics=('mae_forward_velocity_mps','mean_observed_forward_velocity_mps','tilt_rms_rad',
             'abs_heading_change_rad','initial_direction_displacement_m','after3_mean_velocity_mps',
             'cycle_forward_travel_m','hip_pitch_90pct_range_rad','knee_90pct_range_rad')
    aggregates=[]
    for seed in (*exp.SEEDS,'all'):
        for arm in ('original_reward','speed_priority'):
            chosen=[r for r in rows if r['condition']==arm and (seed=='all' or r['train_seed']==seed)]
            aggregates.append(dict(train_seed=seed,condition=arm,episodes=len(chosen),falls=sum(r['fell'] for r in chosen),
                **{k:float(np.mean([r[k] for r in chosen])) if all(r[k] is not None for r in chosen) else None for k in metrics}))
    def group(seed,arm):
        return next(r for r in aggregates if (r['train_seed'],r['condition'])==(seed,arm))
    contrast=effect([group(s,'speed_priority')['mae_forward_velocity_mps']-group(s,'original_reward')['mae_forward_velocity_mps'] for s in exp.SEEDS])
    a,b=group('all','original_reward'),group('all','speed_priority')
    delta={k:b[k]-a[k] for k in metrics if b[k] is not None and a[k] is not None}
    guards=dict(no_new_falls=b['falls']<=a['falls'],tilt=delta['tilt_rms_rad']<=.02,
        heading=delta['abs_heading_change_rad']<=.2,direction=delta['initial_direction_displacement_m']>=-.5)
    all_negative=all(v<0 for v in contrast['per_training_seed'].values())
    statistical=contrast['paired_t_95pct_ci_mps'][1]<0
    branch=('SUPPORTED_CONDITIONAL_SENSITIVITY' if all_negative and statistical and all(guards.values()) else
        'SPEED_GUARDRAIL_TRADEOFF' if delta['mae_forward_velocity_mps']<0 and not all(guards.values()) else
        'INCONCLUSIVE_MIXED_OR_IMPRECISE' if delta['mae_forward_velocity_mps']<0 else 'NO_REPAIR_AT_THIS_WEIGHT')
    report=dict(status='COMPLETED',classification='POST_RESULTS_EXPLORATORY',
        protocol_sha256=exp.sha(exp.OUT/'prospective_protocol.json'),analysis_script_sha256=exp.sha(__file__),
        training_integrity=training,verified_new_episodes=9,verified_cached_episodes=9,
        pairing_checks=pairing,evaluation_configuration_identical=True,
        contrast=contrast,guardrails=dict(checks=guards,all_pass=all(guards.values()),aggregate_differences=delta),
        decision_branch=branch,aggregates=aggregates,episodes=rows,
        limits='Known cached controls, one new speed-weight level, n=3, same-machine software but no exact '
               'physical reproducibility claim. This tests sensitivity of late continuation; it does not '
               'identify the sole cause of original501 performance or prove E better than clock.')
    exp.write(exp.OUT/'analysis.json',report)
    print(json.dumps(dict(branch=branch,contrast=contrast,guardrails=report['guardrails'],aggregates=aggregates),indent=2),flush=True)


if __name__=='__main__':
    run()
