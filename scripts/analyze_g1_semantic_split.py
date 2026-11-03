"""Audit coverage before producing a training-repeat-level screening report."""
from pathlib import Path
import numpy as np

from scripts import g1_semantic_split_study as study
from scripts.run_g1_semantic_split import prepare,read,seal_formal
from scripts.evaluate_g1_semantic_split import validate_coverage, verified_evaluation_rows


def analyze(output):
    output=Path(output);protocol=prepare(output);seal=seal_formal(output)
    rows=[];coverage=[]
    for policy in seal['policies']:
        directory=output/'evaluation/holdout'/f"{policy['arm']}_seed{policy['seed']}"
        manifest=read(directory/'manifest.json')
        if (manifest['status']!='COMPLETED' or manifest['identity']['policy_sha256']!=policy['policy_sha256']
                or manifest['identity']['protocol_sha256']!=study.protocol_digest(protocol)
                or manifest['episodes_sha256']!=study.sha(directory/'episodes.json')):
            raise ValueError('evaluation identity, completion or content failed')
        identity=dict(arm=policy['arm'], training_seed=policy['seed'], split='holdout',
                      policy_sha256=policy['policy_sha256'], protocol_sha256=study.protocol_digest(protocol),
                      source_sha256=seal['source_sha256'])
        episodes=verified_evaluation_rows(directory, manifest, protocol, identity)
        for entry in manifest['raw_trajectories']:
            if study.sha(directory/entry['file'])!=entry['sha256']:raise ValueError('raw trajectory changed')
        for row in episodes:
            if row['arm']!=policy['arm'] or row['training_seed']!=policy['seed']:
                raise ValueError('episode identity does not match policy')
        coverage.append(dict(arm=policy['arm'],seed=policy['seed'],episodes=validate_coverage(episodes,protocol,'holdout')))
        rows.extend(episodes)
    report=study.summarize_effects(rows)
    per_scenario=[]
    for arm in study.ARMS:
        for seed in protocol['training']['seeds']:
            for scenario in protocol['evaluation']['scenarios']:
                subset=[r for r in rows if r['arm']==arm and r['training_seed']==seed and r['scenario']==scenario]
                per_scenario.append(dict(arm=arm,training_seed=seed,scenario=scenario,episodes=len(subset),
                    planar_rmse_mps=float(np.mean([r['planar_rmse_mps'] for r in subset])),
                    yaw_rmse_radps=float(np.mean([r['yaw_rmse_radps'] for r in subset])),
                    failure_rate=float(np.mean([r['fell'] for r in subset])),
                    completion_rate=float(np.mean([r['completion'] for r in subset]))))
    config=protocol['analysis']
    promising=all(v['paired_n']==3 and v['all_seeds_improve']
                  and v['mean_relative_improvement']>=config['screening_relative_improvement']
                  and v['mean_failure_increase']<=config['max_failure_increase']
                  for v in report['comparisons'].values())
    # Check every robustness scenario; lower failure and error in clean conditions
    # cannot compensate for a large degradation hidden in an aggregate.
    guards=[]
    for control in ('matched','random'):
        for scenario in protocol['evaluation']['scenarios']:
            a=[r for r in per_scenario if r['arm']=='semantic' and r['scenario']==scenario]
            b=[r for r in per_scenario if r['arm']==control and r['scenario']==scenario]
            difference=float(np.mean([r['failure_rate'] for r in a])-np.mean([r['failure_rate'] for r in b]))
            guards.append(dict(control=control,scenario=scenario,failure_rate_increase=difference,
                               passed=difference<=config['max_failure_increase']))
    promising=promising and all(g['passed'] for g in guards)
    report.update(status='COMPLETED',analyzed_at=study.now(),protocol_sha256=study.protocol_digest(protocol),
        decision='PROMISING_FOR_EXPANSION' if promising else 'INCONCLUSIVE_OR_NO_SUPPORT',
        coverage=coverage,per_scenario=per_scenario,guardrails=guards,
        interpretation='Exploratory screening; no claim of novelty, generalization or guaranteed publication.')
    study.write(output/'analysis.json',report)
    study.write(output/'raw_evaluation_rows.json',rows)
    text=['# G1 身体状态与任务条件分流：首轮结果','',
          '本报告仅在 12 项完整训练和配对 holdout 评估全部通过来源/覆盖核对后生成。','',
          '**筛选结论：'+report['decision']+'**','',
          '| 对照 | 独立训练重复 | 平面 RMSE 差（分流−对照，m/s） | 平均相对改善 | 跌倒率变化 |',
          '|---|---:|---:|---:|---:|']
    for name,row in report['comparisons'].items():
        text.append(f"| {name} | {row['paired_n']} | {row['mean_difference_mps']:.5f} | {row['mean_relative_improvement']:.2%} | {row['mean_failure_increase']:.2%} |")
    text+=['','3 个训练种子用于筛选，回合数不能替代训练重复。请结合原始轨迹、每场景结果、置信区间及目标硬件延迟决定是否扩展。','']
    (output/'REPORT.md').write_text('\n'.join(text),encoding='utf-8')
    return report
