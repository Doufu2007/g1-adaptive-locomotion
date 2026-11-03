"""Preserve strict replay deviations and measure numerical repeatability."""
import contextlib
import gc
import json
from pathlib import Path
import time
import traceback

import numpy as np
from scripts import g1_phase_feedback_probe_20261026 as exp

MAIN = exp.OUT


def compare():
    records=[]
    for seed in exp.SEEDS:
        first=MAIN/'rollouts'/f'learned_live_{seed}'
        repeat=MAIN/'numerical_controls/rollouts'/f'learned_live_{seed}'
        a,b=exp.read(first/'manifest.json'),exp.read(repeat/'manifest.json')
        assert a['startup_model_hashes']==b['startup_model_hashes']
        assert a['common_configuration_sha256']==b['common_configuration_sha256']
        assert a['policy_sha256']==b['policy_sha256']
        original={(r['scenario'],r['seed']):r for r in exp.read(first/'episodes.json')}
        for row in exp.read(repeat/'episodes.json'):
            source=original[(row['scenario'],row['seed'])]
            p,q=first/source['raw_path'],repeat/row['raw_path']
            assert exp.prior.sha(p)==source['raw_sha256'] and exp.prior.sha(q)==row['raw_sha256']
            with np.load(p) as x,np.load(q) as y:
                end=min(len(x['time']),len(y['time']))
                errors={k:float(np.max(np.abs(x[k][:end].astype(float)-y[k][:end].astype(float))))
                        for k in ('phase_vector','actions','position','quaternion','joint_pos','velocity')}
                initial={k:float(np.max(np.abs(x[k][0].astype(float)-y[k][0].astype(float))))
                         for k in ('observation','position','quaternion','joint_pos','velocity')}
                assert max(initial.values())==0
                records.append(dict(train_seed=seed,scenario=row['scenario'],eval_seed=row['seed'],
                    original_raw_sha256=source['raw_sha256'],repeat_raw_sha256=row['raw_sha256'],
                    exact_initial_input_equal=True,max_errors=errors,
                    strict_1e_5_pass=max(errors.values())<=1e-5,
                    repeat_minus_original_mae_mps=row['mae_forward_velocity_mps']-source['mae_forward_velocity_mps'],
                    repeat_fell=row['fell'],original_fell=source['fell']))
    result=dict(status='COMPLETED',at=exp.prior.now(),episodes=18,all_raw_sha_verified=True,
        script_sha256=exp.prior.sha(__file__),records=records,
        max_abs_mae_repeat_difference_mps=max(abs(r['repeat_minus_original_mae_mps']) for r in records),
        strict_reproducibility_passes=sum(r['strict_1e_5_pass'] for r in records),
        scope='Post-result technical repeatability diagnostic. Original strict threshold is preserved, not relaxed. '
              'Repeated live controls quantify the local numerical floor; do not establish exact replay identity or a causal mechanism.')
    exp.write('numerical_repeatability.json',result)
    return result


def run():
    amendment=MAIN/'numerical_control_amendment.json'
    if not amendment.exists():
        exp.write('numerical_control_amendment.json',dict(at=exp.prior.now(),
            timing='POST_FIRST_SELF_REPLAY_STEADY_RESULTS_BEFORE_ALL_REPEAT_CONTROLS',
            observed_issue='Same phase vectors exactly; physical/actions diverge beyond predeclared1e-5 over20s. '
                           'Initial full98-dim actor observation and state match exactly.',
            preserved_threshold=1e-5, new_control='Repeat live E once for ALL3 PPO seeds, ALL3 eval slots, both scenarios',
            new_control_episodes=18, source_checkpoint_completed_updates=501,
            PPO_training=False,model_or_seed_selection=False,
            original_nine_contrast_family_unchanged=True,
            inference_rule='Keep strict failed identity flags and disclose. Use repeat controls only to contextualize numerical drift, '
                           'not to erase original check or claim clean exact counterfactual feedback isolation.',
            script_sha256=exp.prior.sha(__file__)))
    while exp.read(MAIN/'queue_status.json')['status']!='COMPLETED':
        if exp.read(MAIN/'queue_status.json')['status']=='FAILED':
            raise RuntimeError('Main simulation failed; preserve and investigate')
        time.sleep(10)
    completed=[]
    try:
        for seed in exp.SEEDS:
            directory=MAIN/'numerical_controls/rollouts'/f'learned_live_{seed}'
            if (directory/'manifest.json').exists():
                assert exp.read(directory/'manifest.json')['status']=='COMPLETED'
                completed.append(seed)
                continue
            exp.write('numerical_control_status.json',dict(status='RUNNING',at=exp.prior.now(),current=seed,completed=completed))
            log=MAIN/'logs'/f'numerical_repeat_live_{seed}.log'
            try:
                exp.OUT=MAIN/'numerical_controls'
                with log.open('w',encoding='utf-8') as stream:
                    with contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
                        exp.rollout('learned','live',seed)
            finally:
                exp.OUT=MAIN
            completed.append(seed)
            gc.collect()
            print(json.dumps(dict(numerical_repeat_completed=seed)),flush=True)
        compare()
        exp.write('numerical_control_status.json',dict(status='COMPLETED',at=exp.prior.now(),completed=completed,episodes=18))
        from scripts import analyze_g1_phase_feedback_probe_20261026 as analysis
        analysis.collect()
        print('COMPLETED: numerical controls and full analysis',flush=True)
    except BaseException as error:
        exp.OUT=MAIN
        exp.write('numerical_control_status.json',dict(status='FAILED',at=exp.prior.now(),completed=completed,
            error=repr(error),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':
    run()
