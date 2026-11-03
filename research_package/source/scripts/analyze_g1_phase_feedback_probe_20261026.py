"""All-condition analysis for the predeclared frozen-policy feedback probe."""
import argparse
from collections import defaultdict
import gc
import json
from pathlib import Path

import numpy as np
from scipy.stats import t
from statsmodels.stats.multitest import multipletests

from scripts import g1_phase_feedback_probe_20261026 as exp
from scripts.summarize_g1_best_phase_control import score


def longest_run(mask):
    indices = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    return float(max(indices[1::2]-indices[::2], default=0)*.02)


def heading(raw):
    q = raw['quaternion']
    return np.unwrap(np.arctan2(2*(q[:, 0]*q[:, 3]+q[:, 1]*q[:, 2]),
                              1-2*(q[:, 2]**2+q[:, 3]**2)))


def describe_phase(raw):
    result = {}
    for label, theta in [('input', np.unwrap(raw['angle'].astype(float))),
                         ('shadow_E', raw['shadow_phase'].astype(float))]:
        hz = np.diff(theta)/(2*np.pi*.02)
        mask = raw['time'][1:] >= 3.
        hz = hz[mask]
        if not len(hz):
            continue
        result[label] = dict(mean_hz=float(hz.mean()), std_hz=float(hz.std()),
            quantiles_hz=dict(zip(('p0', 'p1', 'p5', 'p50', 'p95', 'p99', 'p100'),
                                 np.percentile(hz, [0, 1, 5, 50, 95, 99, 100]).tolist())),
            fraction_below_0_2Hz=float(np.mean(hz < .2)),
            fraction_above_3Hz=float(np.mean(hz > 3.)), longest_below_0_2Hz_seconds=longest_run(hz < .2))
        velocity = raw['velocity'][1:, 0][mask]
        lagged = {}
        for lag in (-25, -10, 0, 10, 25):
            # Positive lag: phase rate precedes the observed velocity.
            a, b = (hz[:-lag], velocity[lag:]) if lag > 0 else ((hz[-lag:], velocity[:lag]) if lag < 0 else (hz, velocity))
            lagged[str(lag*.02)] = float(np.corrcoef(a,b)[0,1]) if a.std() > 1e-5 and b.std() > 1e-5 else None
        result[label]['phase_rate_to_future_velocity_correlation'] = lagged
    return result


def supplement(raw, row):
    from scripts.g1_stateful_phase import contact_labels
    yaw = heading(raw)
    delta = raw['position'][-1, :2]-raw['position'][0, :2]
    direction = np.array([np.cos(yaw[0]), np.sin(yaw[0])])
    row.update(abs_heading_change_rad=abs(float(yaw[-1]-yaw[0])),
        initial_direction_displacement_m=float(delta@direction),
        max_abs_heading_deviation_rad=float(np.max(np.abs(yaw-yaw[0]))),
        phase_diagnostics=describe_phase(raw), windows=[])
    _, _, _, cycles = contact_labels(raw['contact'], debounce=3)
    gait=[]
    for start,end,_ in cycles:
        duration=raw['time'][end]-raw['time'][start]
        if raw['time'][start] < 3. or not .15 <= duration <= 2.:
            continue
        gait.append(dict(period_seconds=float(duration),frequency_hz=float(1/duration),
                         integrated_body_forward_velocity_m=float(np.sum(raw['velocity'][start:end,0])*.02)))
    row['gait_diagnostics']=dict(debounce_frames=3, cycles=len(gait),
        mean_cycle_frequency_hz=float(np.mean([r['frequency_hz'] for r in gait])) if gait else None,
        mean_body_forward_travel_per_cycle_m=float(np.mean([r['integrated_body_forward_velocity_m'] for r in gait])) if gait else None)
    row['yaw_rate_mean_rps']=float((yaw[-1]-yaw[0])/(raw['time'][-1]-raw['time'][0]))
    row['yaw_rate_rms_rps']=float(np.sqrt(np.mean((np.diff(yaw)/.02)**2)))
    for start, end in ((0,5),(5,10),(10,15),(15,20)):
        mask = (raw['time'] >= start) & (raw['time'] < end)
        expected = int((end-start)/.02)
        vx = np.zeros(expected)
        vx[:int(mask.sum())] = raw['velocity'][mask, 0]
        row['windows'].append(dict(start=start, end=end, observed_steps=int(mask.sum()),
            mae_mps=float(np.abs(vx-.5).mean()), mean_fixed_window_velocity_mps=float(vx.mean()),
            observed_lateral_rmse_mps=float(np.sqrt(np.mean(raw['velocity'][mask,1]**2))) if mask.any() else None))
    return row


METRICS = ('mae_forward_velocity_mps','mean_observed_forward_velocity_mps',
           'tilt_rms_rad','lateral_velocity_rmse_mps','abs_heading_change_rad',
           'initial_direction_displacement_m','max_abs_heading_deviation_rad')


def effect(values):
    x = np.asarray(values, dtype=float)
    mean, sd = float(x.mean()), float(x.std(ddof=1))
    se = sd/np.sqrt(len(x))
    half = float(t.ppf(.975, 2)*se)
    p = float(2*t.sf(abs(mean/se), 2)) if se else (1. if mean == 0 else 0.)
    return dict(per_training_seed=dict(zip(map(str, exp.SEEDS),x.tolist())), mean_difference_mps=mean,
                paired_t_95pct_ci_mps=[mean-half, mean+half], two_sided_p=p,
                standardized_paired_effect_dz=mean/sd if sd else None,
                seeds_with_lower_mae=int((x < 0).sum()))


def collect(partial=False):
    protocol = exp.lock()
    rows, sources, groups, pairing = [], [], {}, []
    for seed in exp.SEEDS:
        for arm, modes in (('learned', exp.E_MODES), ('clock', exp.C_MODES)):
            for mode in modes:
                name = f'{arm}_{mode}_{seed}'
                directory = exp.OUT/'rollouts'/name
                if not (directory/'manifest.json').exists() or exp.read(directory/'manifest.json')['status'] != 'COMPLETED':
                    continue
                manifest, scored, raws = score(directory)
                assert manifest['experiment_script_sha256'] == protocol['source_hashes']['scripts/g1_phase_feedback_probe_20261026.py']
                assert manifest['encoder_sha256'] == exp.prior.ENCODER_SHA
                assert exp.prior.sha(manifest['policy']) == manifest['policy_sha256']
                assert len(scored) == 6 and manifest['seeds'] == list(exp.EVAL)
                by_key = {}
                for row in scored:
                    raw = raws[('flat',row['scenario'],row['seed'])]
                    assert raw['observation'].shape[1:] == (98,)
                    np.testing.assert_array_equal(raw['observation'][:,9:11], raw['phase_vector'])
                    np.testing.assert_array_equal(raw['episode_step'], np.arange(len(raw['time'])))
                    np.testing.assert_allclose(np.linalg.norm(raw['phase_vector'],axis=-1),1.,atol=2e-6)
                    item = supplement(raw, dict(row, train_seed=seed, mode=mode, condition=arm+'_'+mode))
                    rows.append(item)
                    by_key[(row['scenario'],row['seed'])] = dict(row=item,
                        initial={k:raw[k][0].copy() for k in ('position','quaternion','joint_pos','velocity')})
                groups[name] = dict(manifest=manifest, keyed=by_key)
                sources.append(dict(name=name, manifest_sha256=exp.prior.sha(directory/'manifest.json'),
                                    policy_sha256=manifest['policy_sha256'], source=manifest['source']))
                del raws
                gc.collect()
    for name, group in groups.items():
        base_name = f"clock_clock_{group['manifest']['train_seed']}"
        if base_name not in groups:
            continue
        base = groups[base_name]
        assert group['manifest']['startup_model_hashes'] == base['manifest']['startup_model_hashes']
        assert group['manifest']['common_configuration_sha256'] == base['manifest']['common_configuration_sha256']
        errors=[]
        for key, value in group['keyed'].items():
            errors.extend(float(np.max(np.abs(value['initial'][field].astype(float)-base['keyed'][key]['initial'][field].astype(float))))
                          for field in value['initial'])
        assert max(errors) <= 1e-6, (name, max(errors))
        pairing.append(dict(name=name, base=base_name, max_initial_state_error=max(errors), physics_equal=True, common_config_equal=True))
    aggregates=[]
    conditions = [(a,m) for a,ms in (('learned',exp.E_MODES),('clock',exp.C_MODES)) for m in ms]
    for arm, mode in conditions:
        for seed in (*exp.SEEDS, 'all'):
            for scenario in ('steady','push'):
                selected=[r for r in rows if r['arm']==arm and r['mode']==mode and r['scenario']==scenario
                          and (seed=='all' or r['train_seed']==seed)]
                if selected:
                    diagnostics={}
                    for source in ('input','shadow_E'):
                        diagnostics[source] = {key:float(np.mean([r['phase_diagnostics'][source][key] for r in selected]))
                                               for key in ('mean_hz','std_hz','fraction_below_0_2Hz','fraction_above_3Hz','longest_below_0_2Hz_seconds')}
                    aggregates.append(dict(condition=arm+'_'+mode, train_seed=seed, scenario=scenario,
                        episodes=len(selected), falls=sum(r['fell'] for r in selected),
                        **{k:float(np.mean([r[k] for r in selected])) for k in METRICS},
                        phase_diagnostics=diagnostics,
                        gait_diagnostics={k:float(np.mean([r['gait_diagnostics'][k] for r in selected if r['gait_diagnostics'][k] is not None]))
                                          if any(r['gait_diagnostics'][k] is not None for r in selected) else None
                                          for k in ('mean_cycle_frequency_hz','mean_body_forward_travel_per_cycle_m')},
                        window_mae_mps=[float(np.mean([r['windows'][i]['mae_mps'] for r in selected])) for i in range(4)]))
    contrasts=[]
    for name, treatment, control, scenario in exp.CONTRASTS:
        values=[]
        for seed in exp.SEEDS:
            a={r['seed']:r for r in rows if r['condition']==treatment and r['train_seed']==seed and r['scenario']==scenario}
            b={r['seed']:r for r in rows if r['condition']==control and r['train_seed']==seed and r['scenario']==scenario}
            if set(a)!=set(exp.EVAL) or set(b)!=set(exp.EVAL):
                break
            values.append(float(np.mean([a[k]['mae_forward_velocity_mps']-b[k]['mae_forward_velocity_mps'] for k in exp.EVAL])))
        if len(values)==3:
            contrasts.append(dict(name=name,treatment=treatment,control=control,scenario=scenario,**effect(values)))
    if len(contrasts)==len(exp.CONTRASTS):
        for record, q in zip(contrasts,multipletests([r['two_sided_p'] for r in contrasts],method='fdr_bh')[1]):
            record['q_bh_nine_contrasts']=float(q)
    identity=[]
    for seed in exp.SEEDS:
        donor=exp.OUT/'rollouts'/f'learned_live_{seed}'
        recipient=exp.OUT/'rollouts'/f'learned_self_replay_{seed}'
        if not (recipient/'manifest.json').exists() or exp.read(recipient/'manifest.json')['status']!='COMPLETED':
            continue
        for scenario in ('steady','push'):
            for eval_seed in exp.EVAL:
                with np.load(donor/f'steady_{eval_seed}.npz') as a, np.load(recipient/f'{scenario}_{eval_seed}.npz') as b:
                    end=min(len(a['time']),len(b['time']),1000 if scenario=='steady' else 250)
                    errors={k:float(np.max(np.abs(a[k][:end].astype(float)-b[k][:end].astype(float))))
                            for k in ('phase_vector','actions','position','quaternion','joint_pos','velocity')}
                    identity.append(dict(train_seed=seed, eval_seed=eval_seed, scenario=scenario,
                                         compared_steps=end,max_errors=errors,pass_tolerance_1e_5=max(errors.values())<=1e-5))
    complete=len(rows)==180 and len(contrasts)==9
    numerical_path=exp.OUT/'numerical_repeatability.json'
    numerical=exp.read(numerical_path) if numerical_path.exists() else None
    if not partial:
        assert complete
        assert exp.read(exp.OUT/'queue_status.json')['status']=='COMPLETED'
        assert len(identity)==18
        assert all(r['max_errors']['phase_vector']==0 for r in identity), 'actual replay phase differs from source'
        if not all(r['pass_tolerance_1e_5'] for r in identity):
            assert numerical and numerical['status']=='COMPLETED', 'Strict identity failed; complete preserved numerical controls before interpretation'
    all_controls_ready=complete and len(identity)==18 and (
        all(r['pass_tolerance_1e_5'] for r in identity) or
        bool(numerical and numerical['status']=='COMPLETED'))
    result=dict(status='COMPLETED' if all_controls_ready else 'PARTIAL', at=exp.prior.now(),
        analysis_script_sha256=exp.prior.sha(__file__), verified_episodes=len(rows), all_raw_sha_verified=True,
        protocol_sha256=exp.prior.sha(exp.OUT/'prospective_protocol.json'), pairing_checks=pairing,
        identity_checks=identity, strict_identity_passes=sum(r['pass_tolerance_1e_5'] for r in identity),
        exact_replay_phase_input_verified=all(r['max_errors']['phase_vector']==0 for r in identity),
        numerical_repeatability=numerical, aggregates=aggregates, contrasts=contrasts, episodes=rows, sources=sources,
        inference='Exploratory conditional effects on three pretrained policy seeds. Episodes are nested. Small-n t intervals assume normal seed-level effects; not a causal test of pretraining history.')
    exp.write('analysis'+('_partial' if partial else '')+'.json',result)
    print(json.dumps(dict(status=result['status'],episodes=len(rows),
        aggregates=[r for r in aggregates if r['train_seed']=='all'],contrasts=contrasts),ensure_ascii=False,indent=2))
    return result


def sensitivity():
    # State-conditioned action sensitivity is descriptive, not a control benefit.
    import onnxruntime as ort
    options=ort.SessionOptions(); options.intra_op_num_threads=options.inter_op_num_threads=1
    all_rows=[]
    for seed in exp.SEEDS:
        paths=[]
        for arm in ('clock','learned'):
            for step in (250,500):
                path=exp.prior.checkpoint_policy(arm,seed,step)
                paths.append((arm+f'_{step+1}',path))
            for reward in ('original_reward','pose_removed'):
                path=exp.prior.OUT/'training_interventions/training'/f'{arm}_{reward}_{seed}'/'policy.onnx'
                manifest=exp.read(path.parent/'manifest.json')
                assert exp.prior.sha(path)==manifest['policy_sha256']
                paths.append((arm+'_601_'+reward,path))
        datasets=[]
        for arm, mode in (('clock','clock'),('learned','live')):
            directory=exp.OUT/'rollouts'/f'{arm}_{mode}_{seed}'
            for row in exp.read(directory/'episodes.json'):
                if row['scenario']!='steady':
                    continue
                path=directory/row['raw_path']; assert exp.prior.sha(path)==row['raw_sha256']
                with np.load(path) as raw:
                    idx=np.flatnonzero(raw['time']>=3.)[::10]
                    datasets.append((arm,raw['observation'][idx].copy(),raw['time'][idx].copy(),raw['phase_vector'][0].copy()))
        for label,path in paths:
            policy=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
            def action(obs):
                return np.concatenate([policy.run(None,{policy.get_inputs()[0].name:x[None].astype(np.float32)})[0] for x in obs])
            for source in ('clock','learned'):
                ds=[d for d in datasets if d[0]==source]
                obs=np.concatenate([d[1] for d in ds]); time=np.concatenate([d[2] for d in ds])
                freeze=np.concatenate([np.tile(d[3],(len(d[1]),1)) for d in ds])
                base=action(obs)
                variants={}
                for name,phi in [('rotate_plus15',np.pi/12),('rotate_minus15',-np.pi/12),('rotate90',np.pi/2)]:
                    v=obs.copy(); s,c=obs[:,9],obs[:,10]
                    v[:,9]=s*np.cos(phi)+c*np.sin(phi); v[:,10]=c*np.cos(phi)-s*np.sin(phi)
                    variants[name]=v
                v=obs.copy(); v[:,9:11]=freeze; variants['frozen']=v
                v=obs.copy(); theta=2*np.pi*time/.6; v[:,9:11]=np.stack((np.sin(theta),np.cos(theta)),-1); variants['clock']=v
                changes={}
                dynamic_scale=float(np.sqrt(np.mean(np.var(base,axis=0))))
                for name,v in variants.items():
                    delta=action(v)-base
                    changes[name]=dict(action_rms_difference=float(np.sqrt(np.mean(delta**2))),
                        relative_to_action_temporal_rms=float(np.sqrt(np.mean(delta**2)))/dynamic_scale if dynamic_scale>1e-9 else None,
                        per_joint_rms_difference=np.sqrt(np.mean(delta**2,axis=0)).tolist())
                all_rows.append(dict(train_seed=seed,policy=label,policy_sha256=exp.prior.sha(path),
                    state_source=source,states=len(obs),action_temporal_rms=dynamic_scale,changes=changes))
                print(json.dumps(dict(sensitivity=label,seed=seed,state_source=source)),flush=True)
            del policy
            gc.collect()
    result=dict(status='COMPLETED',at=exp.prior.now(),analysis_script_sha256=exp.prior.sha(__file__),
        subsampling='All steady donor episodes; time>=3 seconds; every10th sample (0.2s)',
        scope='Same physical observations with phase changed. Not a causal test of training history, attention, or quality.',rows=all_rows)
    exp.write('policy_phase_sensitivity.json',result)
    return result


def objective_audit():
    """Read source and saved task configuration; do not infer a heading target."""
    import yaml
    def plain(node):
        if isinstance(node, yaml.MappingNode):
            return {k.value:plain(v) for k,v in node.value}
        if isinstance(node, yaml.SequenceNode):
            return [plain(v) for v in node.value]
        return node.value
    path=exp.OUT/'rollouts'/f'clock_clock_{exp.SEEDS[0]}'/'env.yaml'
    cfg=plain(yaml.compose(path.read_text(encoding='utf-8')))
    command=cfg['commands']['twist']
    terms=cfg['observations']['actor']['terms']
    assert command['heading_command']=='false'
    cohort=[]
    for seed in exp.SEEDS:
        for arm,mode in (('clock','clock'),('learned','live')):
            current=exp.OUT/'rollouts'/f'{arm}_{mode}_{seed}'
            earlier=exp.prior.OUT/'rollouts'/f'training_holdout_{arm}_baseline501_{seed}'
            if not (current/'episodes.json').exists():
                continue
            a,b=exp.read(current/'manifest.json'),exp.read(earlier/'manifest.json')
            for new_seed,old_seed in zip(exp.EVAL,(20262711,20262712,20262713)):
                x,y=current/f'steady_{new_seed}.npz',earlier/f'steady_{old_seed}.npz'
                if not x.exists():
                    continue
                with np.load(x) as xr,np.load(y) as yr:
                    errors={k:float(np.max(np.abs(xr[k][0].astype(float)-yr[k][0].astype(float))))
                            for k in ('joint_pos','quaternion','velocity')}
                    shift=(xr['position'][0]-yr['position'][0]).tolist()
                cohort.append(dict(train_seed=seed,arm=arm,new_seed=new_seed,previous_seed=old_seed,
                    startup_physics_equal=a['startup_model_hashes']==b['startup_model_hashes'],
                    initial_state_errors=errors,initial_position_shift=shift))
    result=dict(status='VERIFIED',at=exp.prior.now(),configuration_sha256=exp.prior.sha(path),
        heading_command=command['heading_command'],heading_range=command['ranges']['heading'],
        angular_velocity_command=command['ranges']['ang_vel_z'],actor_terms=list(terms),
        reward_source_sha256=exp.prior.sha(exp.ROOT/'src/tasks/velocity/mdp/rewards.py'),
        comparison_with_previous_three_world_cohort=cohort,
        interpretation='Heading command disabled; track_angular_velocity penalizes instantaneous body angular velocity error. '
                       'There is no absolute initial-yaw restoring objective in this task. World-direction displacement remains a useful supplementary outcome, not the direct x-velocity reward.')
    exp.write('objective_audit.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',required=True,choices=['collect','sensitivity','objective'])
    parser.add_argument('--partial',action='store_true')
    args=parser.parse_args()
    if args.stage=='sensitivity':
        sensitivity()
    elif args.stage=='objective':
        print(json.dumps(objective_audit(),ensure_ascii=False,indent=2))
    else:
        collect(args.partial)
