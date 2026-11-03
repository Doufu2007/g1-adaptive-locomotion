"""Recompute external-trial scores from saved predictions, without PyTorch/data loads."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/g1_external_methods_trials_20261022/run_v2'
SEEDS = (20261021, 20261022, 20261023)
ARMS = ('tcn10', 'modern10', 'tcn100_rate', 'tctst_scratch100', 'tctst_masked100', 'fld51')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20), b''):
            h.update(block)
    return h.hexdigest()


def write_report(runs, audit):
    def fmt(values, percent=False):
        scale = 100 if percent else 1
        mean = float(np.mean(values))*scale
        sd = float(np.std(values, ddof=1))*scale if len(values)>1 else None
        return f'{mean:.1f} ± {sd:.1f}' if percent and sd is not None else f'{mean:.3f} ± {sd:.3f}' if sd is not None else f'{mean:.3f}（n=1）'
    def get(run,mode,path):
        r=run['metrics'][mode]
        for part in path:r=r[part]
        return r
    lines=['# G1 外部方法顺序实测', '',
        f"状态：{'完整矩阵完成' if audit['complete'] else '进行中，仅已完成结果'}；{len(runs)}/19 次完整训练运行。", '',
        '固定已有 train/validation 开发划分；55820 个一致性训练端点，45236 个全验证端点。三个配对种子：20261021、20261022、20261023。监督均为最终第32轮；TCTST遮蔽预训练、FLD动力学预训练另各32轮。不按验证分数挑轮次。', '',
        '504周期指检测器一致性子集，99周期指事件计数不一致子集；两者均不代表独立触地真值。表中点误差先逐序列平均，再取四序列均值；门禁同样按四序列macro。±为种子间样本标准差。', '',
        '| 模型 / 输出 | 帧 | 种子数 | 一致性MAE rad | 全窗口MAE rad | 504门禁 % | 99门禁 % | 603门禁 % | 正式完整门禁通过 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    groups=[]
    for arm in ARMS:
        rr=sorted([r for r in runs if r['arm']==arm and r['seed_role']=='randomness_estimation'],key=lambda r:r['seed'])
        if not rr:continue
        for mode in rr[0]['metrics']:
            groups.append((arm,mode,rr))
            paths=[('point','consistency','macro_mae_rad'),('point','all','macro_mae_rad'),
                   ('cycles','consistency','macro_completion'),('cycles','mismatch','macro_completion'),('cycles','all','macro_completion')]
            values=[fmt([get(r,mode,path) for r in rr],i>=2) for i,path in enumerate(paths)]
            passed=sum(r['metrics'][mode]['official_full_freeze_eligible'] for r in rr)
            lines.append(f"| {arm} / {mode} | {rr[0]['metrics'][mode]['window']} | {len(rr)} | "+' | '.join(values)+f' | {passed}/{len(rr)} |')
    if audit['complete']:
        best_gate=max(groups,key=lambda item:np.mean([r['metrics'][item[1]]['cycles']['all']['macro_completion'] for r in item[2]]))
        best_mae=min(groups,key=lambda item:np.mean([r['metrics'][item[1]]['point']['consistency']['macro_mae_rad'] for r in item[2]]))
        arm,mode,rr=best_gate
        gate_values=[r['metrics'][mode]['cycles']['all']['macro_completion'] for r in rr]
        consistency_values=[r['metrics'][mode]['cycles']['consistency']['macro_completion'] for r in rr]
        passed=sum(r['metrics'][m]['official_full_freeze_eligible'] for a,m,runs in groups for r in runs)
        conclusion=f"完整实测已完成。全603周期macro均值最高为{arm}/{mode}：{fmt(gate_values,True)}%；其504周期macro为{fmt(consistency_values,True)}%。最低一致性MAE配置为{best_mae[0]}/{best_mae[1]}。正式完整门禁通过{passed}/21个输出评估（三种子×七种输出；FLD直接/跟踪共享检查点）。这是开发集配置比较，不能视作最终独立验证。"
        lines[3:3]=['',conclusion]
    lines+=['', '正式门禁：全603周期macro≥90%，各序列和各长度层≥80%，并满足每条件/每长度层至少10周期等覆盖要求。按半开周期排除终端重置帧，净推进2π容差π/6，最大相邻步进严格小于π/2。正式可冻结状态直接使用v4.2评分器输出。', '',
            '## 三种子逐次结果', '', '| 模型 / 输出 | seed | 一致性MAE rad | 全窗口MAE rad | 504门禁 % | 99门禁 % | 603门禁 % |', '|---|---:|---:|---:|---:|---:|---:|']
    for arm,mode,rr in groups:
        for r in rr:
            m=r['metrics'][mode]
            lines.append(f"| {arm}/{mode} | {r['seed']} | {m['point']['consistency']['macro_mae_rad']:.4f} | {m['point']['all']['macro_mae_rad']:.4f} | {100*m['cycles']['consistency']['macro_completion']:.2f} | {100*m['cycles']['mismatch']['macro_completion']:.2f} | {100*m['cycles']['all']['macro_completion']:.2f} |")
    lines+=['','## 配对差', '', '全部为候选减对照；MAE负值为改善，门禁正值为改善。仅三个种子的描述性结果，不作跨受试者显著性推断。', '', '| 候选 vs 对照 | MAE差 rad（逐种子） | 504门禁差百分点（逐种子） | 603门禁差百分点（逐种子） |', '|---|---|---|---|']
    for pair in audit['paired_differences']:
        cells=[]
        for label,scale in [('mae_consistency',1),('gate504',100),('gate603',100)]:
            cells.append(', '.join(f'{v*scale:+.3f}' for v in pair[label]['candidate_minus_baseline']))
        lines.append(f"| {pair['candidate']} vs {pair['baseline']} | "+' | '.join(cells)+' |')
    lines+=['','## 训练收敛记录', '', '仅用于判断是否完成有效拟合；不同目标的损失数值不可直接比较。所有验证预测均来自最后固定轮次。', '',
            '| 模型 | 阶段 | 首轮损失（种子均值） | 末轮损失（种子均值） |', '|---|---|---:|---:|']
    for arm in ARMS:
        entries=[r for r in audit['audited_runs'] if r['run_id'].startswith(arm+'_seed_') and not r['run_id'].endswith('_repro')]
        if not entries:continue
        for phase in entries[0]['training_curves']:
            first=np.mean([r['training_curves'][phase]['first_loss'] for r in entries])
            final=np.mean([r['training_curves'][phase]['final_loss'] for r in entries])
            lines.append(f'| {arm} | {phase} | {first:.6f} | {final:.6f} |')
    lines+=['','## 分层覆盖与504周期表现', '', '| 模型 / 输出 | 分组 | 周期数 | 完成率 % |', '|---|---|---:|---:|']
    for arm,mode,rr in groups:
        first=rr[0]['metrics'][mode]['cycles']['consistency']
        for key,subkey,countkey,valuekey in [('by_sequence',None,'cycles','macro_component'),('length_strata',None,'cycles','completion')]:
            for label,record in first[key].items():
                assert all(r['metrics'][mode]['cycles']['consistency'][key][label][countkey]==record[countkey] for r in rr)
                values=[r['metrics'][mode]['cycles']['consistency'][key][label][valuekey] for r in rr]
                lines.append(f'| {arm}/{mode} | {label} | {record[countkey]} | {fmt(values,True)} |')
    lines+=['','## 范围与复核', '',
        '- 本次是G1的29关节输入适配实测。ModernTCN使用固定10帧新相位头；TCTST按论文实现，100帧窗口左补8点以右对齐下采样；FLD使用官方编码器并增加零谱/零幅保护和末端相位换算。',
        '- TCTST原论文使用100Hz的200点、IMU输入及100轮训练；本次50Hz的100点、关节输入、32轮。v2恢复前20轮学习率预热及原始sin/cos平方误差，推理时归一化，速度损失仍为0.02权重；不使用验证驱动学习率、早停或选checkpoint。不能称原论文完整复现，外部2.729%周期RMSE不能直接与本次MAE或门禁对比。',
        '- FLD先32轮五步重建推进，再冻结编码器训练读出；tracked输出在同一检查点上，以固定alpha=0.1校正预测速度积分。重置仅依赖序列/真实帧缺口，不读取参考触地。',
        '- 新32轮TCN对照与此前8轮旧结果预算不同；二者不能当成同预算配对。',
        '- 未读取test数组和已查看新跑步序列；未重拟合标准化。验证仍是单一留出受试者的开发集，不是新的独立确认。未修改部署输入契约，未启动PPO。',
        '- 每次运行保留最终/部分检查点、原始预测、逐周期记录、训练日志、配置及哈希。partial checkpoint没有自动续训实现；失败重试从头训练并保留失败attempt。',
        f"- 预测重算和产物哈希已核验{len(audit['audited_runs'])}次完整运行；详见 independent_prediction_audit.json。",'']
    lines+=['- 训练配方修订：v1的TCTST首个种子25轮损失约1.00，在生成任何TCTST验证分数前中断。run_v2/protocol.json冻结学习率预热和原始状态回归修订；v1中断日志/第24轮检查点保留，不进入种子均值。详见相邻amendment_v2_training_recipe.md。',
            '- 继承结果保持原协议身份：已完成ModernTCN/TCN对照引用run_v1原始manifest和产物，run_v2记录其hash；新TCTST及后续运行使用v2协议，不把继承结果改写成新协议训练。','']
    repro=[r for r in runs if r['seed_role']=='fixed_repro']
    for r in repro:
        original=next(v for v in runs if v['arm']==r['arm'] and v['seed']==r['seed'] and v['seed_role']=='randomness_estimation')
        lines.append(f"- ModernTCN同种子完整复训：预测hash一致={r['prediction_sha256']==original['prediction_sha256']}；权重内容hash一致={r['weight_content_sha256']==original['weight_content_sha256']}。")
    lines+=['','来源：[TCTST](https://arxiv.org/html/2506.15150v1) · [ModernTCN官方仓库](https://github.com/luodhhh/ModernTCN) · [FLD](https://arxiv.org/html/2402.13820v1)。全文/源码来源快照见相邻g1_external_methods_search_20261022目录。','']
    (OUT/'external_methods_results_20261022.md').write_text('\n'.join(lines),encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--partial', action='store_true')
    args = parser.parse_args()
    protocol_hash = sha(OUT / 'protocol.json')
    protocol = read(OUT / 'protocol.json')
    for path, expected in protocol['code_hashes'].items():
        assert sha(ROOT / path) == expected, ('code changed', path)
    for path, expected in protocol['source_hashes'].items():
        assert sha(ROOT / path) == expected, ('data changed', path)
    if 'parent_protocol_path' in protocol:
        assert sha(ROOT/protocol['parent_protocol_path'])==protocol['parent_protocol_sha256']
    spec = importlib.util.spec_from_file_location('gate', ROOT / 'src/tasks/velocity/phasor/phase_gate_v4_2.py')
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    with np.load(OUT / 'validation_index.npz', allow_pickle=False) as z:
        index = {k: z[k] for k in z.files}
    names = sorted(set(index['sequence'].tolist()))
    expected_ids = {f'{arm}_seed_{seed}' for arm in ARMS for seed in SEEDS}
    expected_ids.add(f'modern10_seed_{SEEDS[0]}_repro')
    runs = [read(p) for p in sorted(OUT.glob('*/completed.json'))]
    actual = {r['run_id'] for r in runs}
    assert actual <= expected_ids
    if not args.partial:
        assert actual == expected_ids, ('incomplete matrix', sorted(expected_ids-actual))
    audited = []
    metrics_by_run = {}
    diagnostics = []
    for run in runs:
        assert run['status'] == 'COMPLETED'
        if run['run_id'] in protocol.get('carried_runs',{}):
            entry=protocol['carried_runs'][run['run_id']]
            assert run['protocol_sha256']==entry['original_protocol_sha256']
            assert sha(OUT/run['run_id']/'completed.json')==entry['manifest_sha256']
        else:
            assert run['protocol_sha256'] == protocol_hash
        directory = ROOT / run['attempt']
        epochs = [json.loads(line) for line in (directory / 'epochs.jsonl').read_text(encoding='utf-8').splitlines()]
        phases = {'supervised'}
        if run['arm'] == 'tctst_masked100': phases.add('masked_pretrain')
        if run['arm'] == 'fld51': phases.add('fld_pretrain')
        assert {e['phase'] for e in epochs} == phases
        for phase in phases:
            selected = [e for e in epochs if e['phase'] == phase]
            assert [e['epoch'] for e in selected] == list(range(1, 33))
            if run['arm'].startswith('tctst'):
                import math
                for e in selected:
                    expected_lr=6e-4*(.2+.8*(1-math.cos(math.pi*min((e['epoch']-1)/19.,1.)))/2)
                    np.testing.assert_allclose(e['learning_rate'],expected_lr,rtol=0,atol=1e-15)
        assert all(e['samples'] == 55820 for e in epochs if e['phase'] == 'supervised')
        for artifact, expected in run['artifacts'].items():
            assert sha(directory / artifact) == expected, ('artifact changed', directory, artifact)
        for mode in run['metrics']:
            prediction = np.load(directory / f'{mode}_predictions.npy', allow_pickle=False)
            assert prediction.shape[0] == 45236 and np.isfinite(prediction).all()
            assert prediction.ndim==2 and prediction.shape[1] in (2,3)
            np.testing.assert_allclose(np.linalg.norm(prediction[:,:2],axis=1),1.,rtol=0,atol=1e-5)
            if prediction.shape[1]==3:assert np.all(prediction[:,2]>=0)
            scored = gate.score_phase_cycles_v4_2(
                prediction[:, :2], index['cycles'].astype(np.float64)*2*np.pi,
                index['sequence'], index['frame'], index['left_touchdown'],
                period_cuts_frames=(50.66666666666666, 75.), required_conditions=names)
            saved = read(directory / f'{mode}_metrics.json')
            recorded = read(directory / f'{mode}_cycle_records.json')
            assert scored['observed_cycle_count'] == scored['evaluable_cycle_count'] == 603
            assert scored['encoder_freeze_eligible'] == saved['official_full_freeze_eligible']
            assert scored['encoder_freeze_eligible'] == recorded['encoder_freeze_eligible']
            np.testing.assert_allclose(scored['macro_cycle_completion_fraction'], saved['cycles']['all']['macro_completion'], rtol=0, atol=1e-12)
            labels = []
            for row, previous in zip(scored['cycle_records'], recorded['cycle_records']):
                mask = ((index['sequence'] == row['sequence_id']) & (index['frame'] >= row['start_frame']) & (index['frame'] < row['end_frame']))
                labels.append(int(index['right_touchdown'][mask].sum()) == 1)
                assert row['encoder_completed_one_turn'] == previous['encoder_completed_one_turn']
                np.testing.assert_allclose(row['encoder_net_progression_rad'], previous['encoder_net_progression_rad'], rtol=0, atol=1e-12)
            assert sum(labels) == 504
            consistent_rows = [r for r, keep in zip(scored['cycle_records'], labels) if keep]
            failed = [r for r in consistent_rows if not r['encoder_completed_one_turn']]
            diagnostics.append({'run_id':run['run_id'],'mode':mode,'cohort':'504 consistency cycles',
                'all_window_phase_vector_sd':prediction[:,:2].astype(np.float64).std(axis=0).tolist(),
                'all_window_circular_resultant_length':float(np.linalg.norm(prediction[:,:2].astype(np.float64).mean(axis=0))),
                'failed_cycles':len(failed),
                'nonpositive_net':sum(r['encoder_net_progression_rad']<=0 for r in failed),
                'insufficient_net':sum(0<r['encoder_net_progression_rad']<2*np.pi-np.pi/6 for r in failed),
                'excessive_net':sum(r['encoder_net_progression_rad']>2*np.pi+np.pi/6 for r in failed),
                'step_bound_failed':sum(r['encoder_max_abs_step_rad']>=np.pi/2 for r in failed),
                'median_abs_net_error_rad':float(np.median([abs(r['encoder_net_progression_rad']-2*np.pi) for r in failed])) if failed else None})
            for label, yes in [('consistency', True), ('mismatch', False)]:
                subset = [r for r, keep in zip(scored['cycle_records'], labels) if keep == yes]
                macro = np.mean([np.mean([r['encoder_completed_one_turn'] for r in subset if r['sequence_id'] == name]) for name in names])
                np.testing.assert_allclose(macro, saved['cycles'][label]['macro_completion'], rtol=0, atol=1e-12)
            angles = np.arctan2(prediction[:, 0], prediction[:, 1])
            error = np.abs(np.arctan2(np.sin(angles-index['cycles'].astype(np.float64)*2*np.pi), np.cos(angles-index['cycles'].astype(np.float64)*2*np.pi)))
            for label, mask in [('all', np.ones(len(error), bool)), ('consistency', index['consistency'])]:
                macro = np.mean([error[mask & (index['sequence'] == name)].mean() for name in names])
                np.testing.assert_allclose(macro, saved['point'][label]['macro_mae_rad'], rtol=0, atol=1e-12)
            metrics_by_run[run['run_id'], mode] = saved
        curves={phase:{'first_loss':next(e['loss'] for e in epochs if e['phase']==phase),
                       'final_loss':[e['loss'] for e in epochs if e['phase']==phase][-1],
                       'epochs':sum(e['phase']==phase for e in epochs)} for phase in phases}
        audited.append({'run_id': run['run_id'], 'phases': sorted(phases), 'epochs': len(epochs),'training_curves':curves,
                        'training_seconds': sum(e['seconds'] for e in epochs), 'artifact_hashes_verified': len(run['artifacts'])})
    paired = []
    pairs = [('modern10','direct','tcn10','direct'), ('tctst_scratch100','direct','tcn100_rate','direct'),
             ('tctst_masked100','direct','tctst_scratch100','direct'), ('fld51','tracked','fld51','direct')]
    for arm, mode, baseline, baseline_mode in pairs:
        if not all((f'{a}_seed_{seed}', m) in metrics_by_run for a,m in [(arm,mode),(baseline,baseline_mode)] for seed in SEEDS): continue
        row = {'candidate': f'{arm}/{mode}', 'baseline': f'{baseline}/{baseline_mode}', 'seeds': list(SEEDS)}
        for name,path in [('mae_consistency',('point','consistency','macro_mae_rad')), ('gate504',('cycles','consistency','macro_completion')), ('gate603',('cycles','all','macro_completion'))]:
            deltas=[]
            for seed in SEEDS:
                def value(a,m):
                    r=metrics_by_run[f'{a}_seed_{seed}',m]
                    return r[path[0]][path[1]][path[2]]
                deltas.append(value(arm,mode)-value(baseline,baseline_mode))
            row[name]={'candidate_minus_baseline':deltas,'mean':float(np.mean(deltas)),'sd':float(np.std(deltas,ddof=1))}
        paired.append(row)
    result = {'complete': actual == expected_ids, 'completed_runs': len(runs), 'expected_runs':19,
              'protocol_sha256':protocol_hash,'audited_runs':audited,'paired_differences':paired,
              'failure_diagnostics':diagnostics,
              'missing_run_ids':sorted(expected_ids-actual)}
    (OUT/'independent_prediction_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    write_report(runs,result)
    print(json.dumps({'complete':result['complete'],'completed_runs':len(runs),'scores_and_artifact_hashes_verified':True}))


if __name__ == '__main__':
    main()
