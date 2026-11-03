"""Frozen development evaluation grid; independent of policy outcomes."""
import hashlib
import json

SCHEMA_VERSION = 2
CORE_NAMES = ('nominal', 'left_knee_75pct_0p4s', 'right_ankle_75pct_0p4s')


def scenarios():
    result = {'nominal': {'joints': (), 'probability': 0.0, 'duration_s': 0.0}}
    for label, joint in (('left_knee', 'left_knee_joint'), ('right_ankle', 'right_ankle_pitch_joint')):
        for probability in (0.25, 0.75, 1.0):
            for duration in (0.1, 0.4, 0.8):
                name = f'{label}_{round(probability*100)}pct_{str(duration).replace(".","p")}s'
                result[name] = {'joints': (joint,), 'probability': probability, 'duration_s': duration}
    # Complete all three core scenarios before the robustness grid.
    return {name: result[name] for name in (*CORE_NAMES, *(n for n in result if n not in CORE_NAMES))}


def protocol():
    value = {'schema_version': SCHEMA_VERSION, 'groups': list('ABCD'),
             'training_seeds': list(range(42,47)), 'evaluation_seeds': [1042,1043,1044],
             'num_envs':128, 'duration_s':20.0, 'fault_start_s':6.0,
             'core_scenarios':list(CORE_NAMES), 'scenarios':scenarios(),
             'purpose':'DEVELOPMENT_ONLY_ONE_FROZEN_REFERENCE_NOT_HELD_OUT_VALIDATION',
             'primary_comparisons':['D_vs_B', 'D_vs_C', 'D_vs_A'],
             'primary_metrics':['survival_rate', 'fault_alive_xy_rmse_mps',
                                'fault_alive_unaffected_joint_rmse_scaled', 'recovery_without_fall_rate'],
             'analysis_unit':'Training seed; average paired evaluation seeds within each training seed. Environments are rollout replicates, not independent trained models.'}
    value['sha256'] = hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
    return value
