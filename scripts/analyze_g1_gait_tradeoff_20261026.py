"""Post-result descriptive gait decomposition of every prior reward condition."""
from collections import defaultdict
import json
import numpy as np

from scripts import g1_phase_feedback_probe_20261026 as exp
from scripts.g1_stateful_phase import contact_labels


def run():
    rows=[]
    for seed in exp.SEEDS:
        for arm in ('clock','learned'):
            for mode in ('baseline501','original_reward','pose_removed'):
                folder=exp.prior.OUT/'rollouts'/f'training_holdout_{arm}_{mode}_{seed}'
                for record in exp.read(folder/'episodes.json'):
                    path=folder/record['raw_path']
                    assert exp.prior.sha(path)==record['raw_sha256']
                    with np.load(path) as raw:
                        mask=raw['time']>=3.
                        theta=np.unwrap(raw['angle'].astype(float))
                        start=int(np.flatnonzero(mask)[0])
                        phase_frequency=(theta[-1]-theta[start])/(2*np.pi*(raw['time'][-1]-raw['time'][start]))
                        _,_,_,cycles=contact_labels(raw['contact'],debounce=3)
                        frequency,travel=[],[]
                        for s,e,_ in cycles:
                            dt=raw['time'][e]-raw['time'][s]
                            if raw['time'][s]>=3 and .15<=dt<=2:
                                frequency.append(float(1/dt))
                                travel.append(float(raw['velocity'][s:e,0].sum())*.02)
                        amplitude=np.percentile(raw['joint_pos'][mask],95,axis=0)-np.percentile(raw['joint_pos'][mask],5,axis=0)
                        rows.append(dict(train_seed=seed,arm=arm,mode=mode,eval_seed=record['seed'],
                            raw_path=str(path),raw_sha256=record['raw_sha256'],
                            after3_mean_velocity_mps=float(raw['velocity'][mask,0].mean()),
                            phase_frequency_hz=float(phase_frequency),
                            mean_contact_cycle_frequency_hz=float(np.mean(frequency)) if frequency else None,
                            mean_body_forward_travel_per_cycle_m=float(np.mean(travel)) if travel else None,
                            hip_pitch_90pct_range_rad=float(amplitude[[0,6]].mean()),
                            knee_90pct_range_rad=float(amplitude[[3,9]].mean()),
                            ankle_pitch_90pct_range_rad=float(amplitude[[4,10]].mean())))
    metrics=('after3_mean_velocity_mps','phase_frequency_hz','mean_contact_cycle_frequency_hz',
             'mean_body_forward_travel_per_cycle_m','hip_pitch_90pct_range_rad',
             'knee_90pct_range_rad','ankle_pitch_90pct_range_rad')
    aggregates=[]
    for seed in exp.SEEDS:
        for arm in ('clock','learned'):
            for mode in ('baseline501','original_reward','pose_removed'):
                current=[r for r in rows if r['train_seed']==seed and r['arm']==arm and r['mode']==mode]
                aggregates.append(dict(train_seed=seed,arm=arm,mode=mode,episodes=len(current),
                    **{k:float(np.mean([r[k] for r in current if r[k] is not None])) for k in metrics}))
    result=dict(status='COMPLETED',at=exp.prior.now(),verified_episodes=54,
        script_sha256=exp.prior.sha(__file__),classification='POST_RESULTS_SUPPLEMENTARY_DESCRIPTION',
        prior_analysis_sha256=exp.prior.sha(exp.prior.OUT/'training_intervention_analysis.json'),
        scope='All prior reward continuation outcomes. Descriptive cadence/travel/joint-range decomposition; '
              'no new hypothesis test, no causal mediation estimate and no outcome selection. '
              'Contact cycle is left touchdown to next left touchdown, not a single footstep.',
        aggregates=aggregates,episodes=rows)
    exp.write('reward_gait_supplement.json',result)
    print(json.dumps(aggregates,ensure_ascii=False,indent=2))


if __name__=='__main__':
    run()
