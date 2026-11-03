"""Verify and summarize every predeclared cause-audit condition, without selection."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import t
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import diagnose_g1_best_e_closed_loop_20261026 as audit
from scripts.summarize_g1_best_phase_control import score

SEEDS = (20261901, 20261902, 20261903)
METRICS = ('mae_forward_velocity_mps', 'tilt_rms_rad',
           'lateral_velocity_rmse_mps', 'mean_observed_forward_velocity_mps',
           'forward_displacement_m', 'abs_heading_change_rad',
           'after3_velocity_mae_mps', 'after3_mean_velocity_mps')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def fingerprint(values):
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def load_condition(name, seed):
    directory = audit.OUT/'rollouts'/name
    if not (directory/'manifest.json').exists():
        return None
    original = read(directory/'manifest.json')
    if original['status'] != 'COMPLETED':
        return None
    manifest, original_rows, raws = score(directory)
    rows, initial = [], {}
    for original_row in original_rows:
        key = (original_row['terrain'], original_row['scenario'], original_row['seed'])
        raw = raws[key]
        mask = raw['time'] >= 3.
        row = {k: original_row[k] for k in METRICS if k in original_row}
        row.update(train_seed=seed, scenario=key[1], eval_seed=key[2],
                   fell=original_row['fell'], raw_path=original_row['raw_path'],
                   raw_sha256=original_row['raw_sha256'],
                   abs_heading_change_rad=abs(original_row['heading_change_rad']),
                   after3_velocity_mae_mps=float(np.mean(np.abs(
                       raw['velocity'][mask, 0]-raw['command'][mask]))) if mask.any() else None,
                   after3_mean_velocity_mps=float(np.mean(raw['velocity'][mask, 0])) if mask.any() else None)
        rows.append(row)
        initial[key] = {k: raw[k][0].copy() for k in
                        ('position', 'quaternion', 'joint_pos', 'velocity')}
        initial[key]['command_sha256'] = fingerprint(raw['command'])
        initial[key]['steps'] = len(raw['time'])
    expected = {(scenario, s) for scenario in manifest['scenarios'] for s in manifest['seeds']}
    assert {(r['scenario'], r['eval_seed']) for r in rows} == expected
    assert len(rows) == len(expected) == manifest['episode_count']
    assert manifest['cause_audit']['train_seed'] == seed
    assert audit.sha(manifest['policy']) == manifest['policy_sha256']
    result = dict(name=name, train_seed=seed, manifest_sha256=audit.sha(directory/'manifest.json'),
                  policy_sha256=manifest['policy_sha256'],
                  startup_model_hashes=manifest['startup_model_hashes'],
                  technical_runtime_recovery=manifest.get('technical_runtime_recovery'),
                  provenance=manifest['cause_audit'], rows=rows, initial=initial)
    del raws
    gc.collect()
    return result


def paired_check(a, b, same_policy=False):
    assert a['startup_model_hashes'] == b['startup_model_hashes'], (a['name'], b['name'])
    if same_policy:
        assert a['policy_sha256'] == b['policy_sha256']
    assert a['initial'].keys() == b['initial'].keys()
    max_error = 0.
    for key, first in a['initial'].items():
        second = b['initial'][key]
        for field in ('position', 'quaternion', 'joint_pos', 'velocity'):
            error = float(np.max(np.abs(first[field].astype(float)-second[field].astype(float))))
            assert error <= 1e-6, (a['name'], b['name'], key, field, error)
            max_error = max(max_error, error)
        if first['steps'] == second['steps']:
            assert first['command_sha256'] == second['command_sha256']
    return dict(first=a['name'], second=b['name'], episodes=len(a['initial']),
                initial_state_max_abs_error=max_error, startup_physics_equal=True,
                policy_weights_equal=a['policy_sha256'] == b['policy_sha256'])


def aggregate(groups):
    output = []
    for label, conditions in groups.items():
        for seed in (*SEEDS, 'all'):
            rows = [r for s, group in conditions.items() if seed == 'all' or s == seed
                    for r in group['rows']]
            for scenario in ('all', 'steady', 'speed_change', 'push'):
                current = [r for r in rows if scenario == 'all' or r['scenario'] == scenario]
                if not current:
                    continue
                output.append(dict(condition=label, train_seed=seed, scenario=scenario,
                    episodes=len(current), falls=sum(r['fell'] for r in current),
                    **{k:float(np.mean([r[k] for r in current if r[k] is not None]))
                       if any(r[k] is not None for r in current) else None for k in METRICS}))
    return output


def effect(differences):
    x = np.asarray(differences, dtype=float)
    assert len(x) == 3
    mean = float(x.mean())
    sd = float(x.std(ddof=1))
    se = sd/np.sqrt(3)
    half = float(t.ppf(.975, 2)*se)
    p = float(2*t.sf(abs(mean/se), 2)) if se else (1. if mean == 0 else 0.)
    return dict(mean_difference_mps=mean, per_training_seed=dict(zip(map(str, SEEDS), x.tolist())),
                training_seed_count=3, paired_t_95pct_ci_mps=[mean-half, mean+half],
                two_sided_p=p, standardized_paired_effect_dz=mean/sd if sd else None,
                direction_count_lower=int((x < 0).sum()))


def contrast(groups, name, positive, negative, metric='mae_forward_velocity_mps'):
    terms = [(label, weight) for label, weight in
             [(label, 1.) for label in positive]+[(label, -1.) for label in negative]]
    if not all(set(groups.get(label, {})) == set(SEEDS) for label, _ in terms):
        return None
    differences = []
    for seed in SEEDS:
        keyed = [{(r['scenario'], r['eval_seed']):r for r in groups[label][seed]['rows']}
                 for label, _ in terms]
        assert all(set(k) == set(keyed[0]) for k in keyed)
        differences.append(float(np.mean([sum(weight*rows[key][metric]
            for (_, weight), rows in zip(terms, keyed)) for key in sorted(keyed[0])])))
    return dict(name=name, metric=metric, positive_terms=positive, negative_terms=negative,
                **effect(differences))


def training_integrity(partial):
    import yaml
    base = audit.OUT/'training_interventions/training'
    records, checks = [], []
    for seed in SEEDS:
        by_arm = {}
        for arm in ('clock', 'learned'):
            pair = {}
            configs = {}
            for mode in ('original_reward', 'pose_removed'):
                directory = base/f'{arm}_{mode}_{seed}'
                if not (directory/'manifest.json').exists():
                    continue
                record = read(directory/'manifest.json')
                if record['status'] != 'COMPLETED':
                    continue
                assert record['additional_completed_updates'] == 100
                assert record['collected_transitions'] == 100*64*24
                assert record['encoder_frozen_verified']
                assert record['initial_optimizer_state_entries'] > 0
                assert audit.sha(directory/'policy.onnx') == record['policy_sha256']
                assert audit.sha(record['source_checkpoint']) == record['source_checkpoint_sha256']
                record = dict(record, final_checkpoint_sha256=audit.sha(directory/'model_600.pt'))
                records.append(record)
                pair[mode] = record
                # Compose nodes preserves Python-specific tags as data; never constructs objects.
                node = yaml.compose((directory/'env.yaml').read_text(encoding='utf-8'))
                configs[mode] = node
            if len(pair) != 2:
                continue
            a, b = pair['original_reward'], pair['pose_removed']
            fields = ('source_checkpoint_sha256', 'initial_actor_sha256', 'initial_critic_sha256',
                      'initial_learning_rate', 'initial_optimizer_state_entries', 'continuation_rng_seed',
                      'startup_model_hashes', 'collected_transitions')
            for field in fields:
                assert a[field] == b[field], (arm, seed, field)
            def plain(node, path=()):
                if isinstance(node, yaml.MappingNode):
                    return {k.value:plain(v, path+(k.value,)) for k,v in node.value}
                if isinstance(node, yaml.SequenceNode):
                    return [plain(v, path+(str(i),)) for i,v in enumerate(node.value)]
                if path == ('rewards','pose','weight'):
                    return 'CONTROLLED_POSE_WEIGHT'
                return [node.tag, node.value]
            assert plain(configs['original_reward']) == plain(configs['pose_removed'])
            d = base/f'{arm}_original_reward_{seed}'
            e = base/f'{arm}_pose_removed_{seed}'
            assert audit.sha(d/'agent.yaml') == audit.sha(e/'agent.yaml')
            checks.append(dict(arm=arm, train_seed=seed, restored_state_equal=True,
                               only_config_change='pose reward weight', budget_equal=True,
                               initial_actor_sha256=a['initial_actor_sha256'],
                               initial_critic_sha256=a['initial_critic_sha256']))
            by_arm[arm] = a
        if len(by_arm) == 2:
            assert by_arm['clock']['startup_model_hashes'] == by_arm['learned']['startup_model_hashes']
    if not partial:
        assert len(records) == 12 and len(checks) == 6
    probe_path = audit.OUT/'training_interventions/smoke_sampling_phase_consistency.json'
    probe = read(probe_path) if probe_path.exists() else dict(status='PENDING')
    if not partial:
        assert probe['status'] == 'PASS'
        assert probe['steps'] == 48 and probe['compared_environment_frames'] == 3072
    return dict(completed_training_runs=len(records), within_arm_reward_pairs=checks,
                sampling_phase_check=probe, manifests=records)


def collect(stage, partial=False):
    if stage == 'phase':
        labels = ['holdout_clock_clock', 'holdout_clock_raw', 'holdout_clock_aligned',
                  'holdout_learned_raw', 'holdout_learned_aligned',
                  'midpoint_clock', 'midpoint_learned']
        specifications = [
            ('clock_policy_raw_E_minus_clock', ['holdout_clock_raw'], ['holdout_clock_clock']),
            ('clock_policy_alignment_effect', ['holdout_clock_aligned'], ['holdout_clock_raw']),
            ('E_policy_alignment_effect', ['holdout_learned_aligned'], ['holdout_learned_raw']),
            ('E501_minus_clock501', ['holdout_learned_raw'], ['holdout_clock_clock']),
            ('clock501_minus_clock251', ['holdout_clock_clock'], ['midpoint_clock']),
            ('E501_minus_E251', ['holdout_learned_raw'], ['midpoint_learned']),
        ]
        expected = 630
        status_path = audit.OUT/'queue_status.json'
    else:
        labels = [f'training_holdout_{arm}_{mode}' for arm in ('clock', 'learned')
                  for mode in ('baseline501', 'original_reward', 'pose_removed')]
        specifications = []
        for arm in ('clock', 'learned'):
            label = 'training_holdout_'+arm+'_'
            specifications.extend([
                (arm+'_extra100_original_reward', [label+'original_reward'], [label+'baseline501']),
                (arm+'_pose_removal_effect', [label+'pose_removed'], [label+'original_reward'])])
        specifications.append(('phase_by_pose_interaction',
            ['training_holdout_learned_pose_removed', 'training_holdout_clock_original_reward'],
            ['training_holdout_learned_original_reward', 'training_holdout_clock_pose_removed']))
        expected = 54
        status_path = audit.OUT/'training_interventions/queue_status.json'
    groups = {label:{} for label in labels}
    for label in labels:
        for seed in SEEDS:
            condition = load_condition(label+'_'+str(seed), seed)
            if condition:
                groups[label][seed] = condition
    count = sum(len(g['rows']) for s in groups.values() for g in s.values())
    if not partial:
        assert count == expected, (count, expected)
        assert read(status_path)['status'] == 'COMPLETED'
    checks = []
    if stage == 'phase':
        pairs = [('holdout_clock_clock', l, l.startswith('holdout_clock')) for l in labels[1:]]
        pairs.append(('holdout_learned_raw', 'holdout_learned_aligned', True))
    else:
        pairs = [(f'training_holdout_{arm}_baseline501', f'training_holdout_{arm}_{mode}', False)
                 for arm in ('clock', 'learned') for mode in ('original_reward', 'pose_removed')]
        pairs.extend([('training_holdout_clock_'+mode, 'training_holdout_learned_'+mode, False)
                      for mode in ('baseline501', 'original_reward', 'pose_removed')])
    for a, b, same in pairs:
        for seed in SEEDS:
            if seed in groups[a] and seed in groups[b]:
                checks.append(paired_check(groups[a][seed], groups[b][seed], same))
    effects = [result for spec in specifications if (result:=contrast(groups, *spec)) is not None]
    complete_family = len(effects) == len(specifications)
    if complete_family:
        for row, q in zip(effects, multipletests([r['two_sided_p'] for r in effects], method='fdr_bh')[1]):
            row['q_bh_full_family'] = float(q)
            row['direction_established_at_q05'] = bool(q < .05 and
                not row['paired_t_95pct_ci_mps'][0] <= 0 <= row['paired_t_95pct_ci_mps'][1])
    result = dict(status='COMPLETED' if count == expected and complete_family else 'PARTIAL',
        stage=stage, computed_at=audit.now(), script_sha256=audit.sha(__file__),
        verified_raw_episodes=count, expected_raw_episodes=expected,
        all_files_verified=True, pairing_checks=checks, aggregates=aggregate(groups),
        speed_contrasts=effects, full_comparison_family_complete=complete_family,
        comparison_family_timing=(
            'Six reporting contrasts specified after the first seed partial holdout; exploratory. '
            'All simulation conditions and the phase correction were locked before holdout.'
            if stage == 'phase' else
            'Five contrasts specified in prospective_protocol.json before new training.'),
        inference='Exploratory paired differences across three original PPO training seeds; '
                  'evaluation episodes nested within policies. Small-n t intervals assume approximately '
                  'normal policy-level effects; no normality validation or broad generalization is possible.',
        selection='Every predeclared condition and final checkpoint included; no outcome exclusions',
        condition_sources=[{k:v for k,v in g.items() if k not in ('initial','rows')}
                           for seed_groups in groups.values() for g in seed_groups.values()],
        episodes=[dict(condition=label, **r) for label, seed_groups in groups.items()
                  for g in seed_groups.values() for r in g['rows']])
    if stage == 'training':
        result['training_integrity'] = training_integrity(partial)
    audit.write(stage+'_intervention_analysis'+('_partial' if partial else '')+'.json', result)
    print(json.dumps(dict(status=result['status'], stage=stage, episodes=count,
        aggregates=[r for r in result['aggregates'] if r['scenario']=='all' and r['train_seed']=='all'],
        effects=effects, pairing_checks=len(checks)), ensure_ascii=False, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['phase','training'], required=True)
    parser.add_argument('--partial', action='store_true')
    args = parser.parse_args()
    collect(args.stage, args.partial)
