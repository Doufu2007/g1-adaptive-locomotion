"""Read-only aggregation of frozen analyses; no simulation or statistical refit."""
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
STAGES = {
    '确认': 'g1_rough_confirmation_20261028',
    'S': 'g1_rough_domain_adaptation_20261028',
    'R': 'g1_rough_transfer_20261028',
    'D': 'g1_source_anchor_20261027',
    'C': 'g1_phase_policy_cross_20261027',
    'B': 'g1_contact_phase_reference_20261027',
    'A': 'g1_phase_lr_factorial_20261027',
}
NAMES = {
    'RE_live': '粗糙训练＋实时学习相位',
    'RE_frozen': '相同学习策略＋相位恒定在回合初值',
    'RC_clock': '粗糙训练＋固定节奏',
    'FC_clock': '平地训练＋固定节奏',
    'RE': '粗糙训练＋学习相位', 'RC': '粗糙训练＋固定节奏',
    'FE': '平地训练＋学习相位', 'FC': '平地训练＋固定节奏',
    'AC': '原固定节奏', 'AE': '原学习相位',
    'DC': '源策略约束修正＋固定节奏', 'DE': '源策略约束修正＋学习相位',
}
METRICS = ['mae_forward_velocity_mps', 'forward_displacement_m',
           'tilt_rms_rad', 'abs_heading_change_rad', 'survival_seconds', 'moving_fraction']
METRIC_LABELS = ['速度平均绝对误差 m/s↓', '前进位移 m↑', '倾斜均方根 rad↓',
                 '绝对航向变化 rad↓', '存活时间 s↑', '持续前进时间占比↑']
data, sources = {}, {}
for stage, folder in STAGES.items():
    path = ROOT / folder / 'analysis.json'
    raw = path.read_bytes()
    data[stage] = json.loads(raw.decode('utf-8'))
    source = {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest(),
              'analysis_at': data[stage]['at'], 'scored_episode_count': len(data[stage]['episodes'])}
    audit_path = path.with_name('independent_audit.json')
    if audit_path.exists():
        audit_raw = audit_path.read_bytes()
        audit = json.loads(audit_raw.decode('utf-8'))
        assert audit['analysis_sha256'] == source['sha256'], (stage, 'audit binding')
        source.update(audit_status=audit['status'], audit_path=str(audit_path),
                      audit_sha256=hashlib.sha256(audit_raw).hexdigest())
    sources[stage] = source


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(map(str, headers)) + ' |',
                      '|' + '|'.join(['---'] * len(headers)) + '|'] +
                     ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def aggregate(es):
    result = {'n': len(es), 'completed': sum(e.get('completed', False) for e in es),
              'falls': sum(e['fell'] for e in es)}
    for key in METRICS:
        lookup = 'initial_direction_displacement_m' if key == 'forward_displacement_m' and key not in es[0] else key
        if lookup in es[0]:
            result[key] = mean(float(e[lookup]) for e in es)
    result['completion_rate'] = result['completed'] / len(es)
    result['fall_rate'] = result['falls'] / len(es)
    return result


def rough_row(method, es):
    m = aggregate(es)
    return [NAMES.get(method, method), f"{m['completed']}/{m['n']}（{100*m['completion_rate']:.2f}%）",
            f"{m['falls']}/{m['n']}（{100*m['fall_rate']:.2f}%）"] + [f'{m[k]:.4f}' for k in METRICS]


parts = [
    '**截至 2026 年 10 月 28 日的近期全部对比结果**',
    '范围：A/B/C/D 原因排查、R 粗糙迁移、S 等预算地形适配、新地图确认及相位冻结消融。'
    '本文件从封存 analysis.json 汇总已评分回合；置信区间和检验值直接引用原分析，未重新选择指标或拟合统计模型。',
    '最新确认于北京时间 2026-10-28 13:02 完成分析，13:07 独立 CPU 复算通过。'
    '同粗糙训练的公平固定节奏对照为 RC_clock；FC_clock 也在粗糙地形评测，但训练域为平地。',
    '**粗糙地形指标口径**',
    '每回合最长 20 秒；完成要求：全程未倒、前进位移至少 5 米、前进速度超过 0.1 m/s 的时间占比至少 80%、'
    '指定地形内部有效足地接触比例至少 95%。因此完成数与跌倒数之和可以小于回合数。'
    '速度误差按完整 20 秒计算，早倒后的速度按零补齐；位移、倾斜及航向沿用原评分口径。'
    '下表各项均为描述性均值。保护门通过表示未超出预设容许恶化范围，不表示该项数值必然更优。',
]
flat_rows, grouped_rows, episode_rows = [], [], []
for stage in ['确认', 'S', 'R']:
    groups = defaultdict(list)
    for e in data[stage]['episodes']:
        domain = '平地' if e['terrain_seed'] == 0 else '粗糙'
        groups[(domain, e['method'])].append(e)
    rough_methods = sorted(m for domain, m in groups if domain == '粗糙')
    assert all(len(groups[('粗糙', m)]) == 90 for m in rough_methods)
    label = {'确认': '最新：全新粗糙地图，固定 S 最终权重',
             'S': 'S：等预算地形适配后的旧评测地图',
             'R': 'R：平地训练策略直接迁移粗糙地图'}[stage]
    parts += [f'**{label}**', table(['方法', '完成', '跌倒'] + METRIC_LABELS,
                                                [rough_row(m, groups[('粗糙', m)]) for m in rough_methods])]
    for (domain, method), es in sorted(groups.items()):
        grouped_rows.append({'stage': stage, 'domain': domain, 'method': method,
                             'method_label': NAMES[method], **aggregate(es)})
        if domain == '平地':
            flat_rows.append([stage, NAMES[method], len(es), sum(e['fell'] for e in es)] +
                             [f'{aggregate(es)[k]:.4f}' for k in METRICS[:4]])
    parts += ['恒速／受推分层（各方法每场景 45 回合）：',
              table(['方法', '场景', '完成/45', '跌倒/45', '速度误差 m/s', '位移 m', '存活 s'], [
                  [NAMES[m], '恒速' if scenario == 'steady' else '受推',
                   sum(e['completed'] for e in es), sum(e['fell'] for e in es),
                   f"{aggregate(es)['mae_forward_velocity_mps']:.4f}",
                   f"{aggregate(es)['forward_displacement_m']:.4f}",
                   f"{aggregate(es)['survival_seconds']:.4f}"]
                  for m in rough_methods for scenario in ['steady', 'push']
                  if (es := [e for e in groups[('粗糙', m)] if e['scenario'] == scenario])])]
    family = data[stage]['statistics']['family']
    parts += ['原锁定的全部主要统计对比（差值均为前者减后者；完成/跌倒单位为百分点）：',
              table(['原比较标识', '均差', '95%区间', '三个训练来源各自差值', '精确检验', 'Holm校正'], [
                  [key, f"{100*v['mean']:+.2f}",
                   f"[{100*v['ci95'][0]:+.2f}, {100*v['ci95'][1]:+.2f}]",
                   '、'.join(f'{100*x:+.2f}' for x in v['per_seed']), v['p_exact'], v['p_holm']]
                  for key, v in family.items()])]
    cand = data[stage]['statistics']['candidate']
    parts += ['原预设门判决：' + '；'.join(f"相对 {NAMES.get(k, k)}：{'通过' if v['passed'] else '未通过'}"
                     for k, v in cand['comparators'].items()) + f"；整体：{'通过探索候选门' if cand['passed'] else '未通过'}。"]
    for method in rough_methods:
        es = groups[('粗糙', method)]
        seed_groups = [[e for e in es if e['train_seed'] == seed] for seed in [20261901, 20261902, 20261903]]
        assert all(len(x) == 30 for x in seed_groups)
        assert abs(mean(aggregate(x)['completion_rate'] for x in seed_groups) - aggregate(es)['completion_rate']) < 1e-12
        assert abs(mean(aggregate(x)['fall_rate'] for x in seed_groups) - aggregate(es)['fall_rate']) < 1e-12

parts += [
    '**最新确认与旧地图探索的直接对照**',
    table(['指标', 'S：学习相位减同粗糙训练固定节奏', '新地图确认：同一比较'], [
        ['完成率', '+14.44 个百分点', '+3.33 个百分点'],
        ['跌倒率', '−14.44 个百分点', '−2.22 个百分点'],
        ['速度误差 m/s', '−0.0420', '−0.0038'],
        ['前进位移 m', '+1.5723', '+0.6177'],
        ['倾斜 rad', '−0.0214', '+0.0031'],
        ['绝对航向变化 rad', '−0.1883', '−0.0262'],
        ['存活时间 s', '+1.4044', '−0.8149'],
    ]),
    '新地图确认相对平地训练固定节奏，完成率描述性增加 23.33 个百分点、跌倒率下降 32.22 个百分点；'
    '该比较同时包含训练地形差异，不能单独识别学习相位的贡献。',
    '相位冻结消融：实时学习相位完成 31/90、跌倒 34/90；相位固定在回合初值时完成 1/90、跌倒 84/90。'
    '完成率差 +33.33、跌倒率差 −55.56 个百分点。此干预也停止了相位正常推进；'
    '它检验实时相位与恒定相位的差别，尚未用保持节奏推进的回放或其他匹配干预单独分离状态反馈贡献。',
    '三个训练来源在 S 与确认阶段相同；新地图和新评测初态不是新的训练模型。'
    '上述精确检验的 Holm 校正值均为 1，未获得校正后统计显著优势。'
    '预设点效应门通过、描述性数值更好与统计显著属于不同判断。',
    '**补充：近期平地评测完整表**',
    'R、S 各方法平地均 30/30 完成、0/30 跌倒；包含恒速与受推。平地结果用于完整记录，与粗糙地形分开解释。',
    table(['阶段', '方法', '回合', '跌倒'] + METRIC_LABELS[:4], flat_rows),
]

for stage in ['A', 'B', 'D', 'C']:
    groups = defaultdict(list)
    for e in data[stage]['episodes']:
        if stage != 'C' and (e['completed_updates'] != 501 or e['scenario'] != 'steady'):
            continue
        if stage == 'A':
            key = (e['arm'], e['lr_mode'])
        elif stage == 'D':
            key = (e['arm'], e['variant'])
        elif stage == 'C':
            key = (e['controller'], e['phase'])
        else:
            key = (e['arm'],)
        groups[key].append(e)
    titles = {'A': 'A：学习率与相位的对照，最终501更新恒速',
              'B': 'B：接触参考相位对照，最终501更新恒速',
              'D': 'D：源策略约束修正，最终501更新恒速',
              'C': 'C：策略权重与相位输入交叉，固定权重零训练'}
    rows = []
    for key, es in sorted(groups.items()):
        agg = aggregate(es)
        if stage == 'C':
            label = '权重=' + key[0] + '；相位输入=' + key[1]
        else:
            label = ' / '.join(key)
        rows.append([label, len(es), agg['falls']] + [f'{agg[k]:.6f}' for k in METRICS[:4]])
    parts += [f'**{titles[stage]}**', table(['方法', '回合', '跌倒'] + METRIC_LABELS[:4], rows)]
    contrast_rows = []
    for name, v in data[stage]['contrasts'].items():
        ci = v.get('ci95')
        contrast_rows.append([name, f"{v['mean']:+.6f}",
                              f'[{ci[0]:+.6f}, {ci[1]:+.6f}]' if ci else '原分析仅描述性',
                              v.get('holm_p', '不适用')])
    parts += [table(['原比较标识（速度误差差值）', '均差 m/s', '95%区间', 'Holm校正'], contrast_rows)]

parts += [
    'A/B/D 使用各自固定501终点及原恒速初态，不与 C 的新九世界评测队列直接混算。'
    'B 及 D 包含复用的 A 基线回合；不同阶段回合数不能直接累加为独立样本量。'
    'C 中 source 权重在训练来源槽位间共享，属于输入兼容性描述诊断。',
    '**完成状态与成本（正式有效预算）**',
    table(['阶段', '已评分回合', '正式新增训练步', '正式新增PPO更新', '有效评测步', '审计'], [
        [stage, len(data[stage]['episodes']), data[stage].get('formal_training_transitions', 0),
         data[stage].get('formal_PPO_updates', 0), data[stage]['effective_eval_transitions'],
         sources[stage].get('audit_status', '见原阶段验证记录')]
        for stage in ['R', 'S', '确认']]),
    'S 正式新增 Adam 步 60,240；R 与确认阶段评测均无 PPO/Adam 更新。'
    '成功 smoke 与技术失败成本分别保存在各阶段 technical_cost_ledger、smoke/audit 和失败目录中，'
    '上表仅列正式有效预算，不代表全部历史消耗。',
    '**来源及复核**',
    '本次核对了封存分析文件 SHA-256、已有独立审计对分析文件的绑定、全部方法90回合覆盖、'
    '每训练来源30回合覆盖及汇总与来源均值口径一致性。独立 CPU 审计为同一操作者的分离计算实现；'
    '审计通过表示记录/实现/复算一致，不等于研究假设成立。',
]
for stage, source in sources.items():
    parts += [f"{stage}：[原分析]({Path(source['path']).as_posix()})；SHA-256 `{source['sha256']}`。"]
parts += ['历史编码器与外部对照索引：',
          '- `../g1_stateful_phase_20261025/results.md`：五类离线相位模型、三训练种子结果。',
          '- `../g1_phase_controller_adaptation_20261024/claim_evidence_table_20261025.md`：旧编码器控制对照。',
          '- `../g1_external_methods_trials_20261022/run_v2/external_methods_results_20261022.md`：历史外部方法试验。',
          '这些历史试验使用不同模型或协议，完整原记录继续保留。']

for stage, a in data.items():
    for e in a['episodes']:
        row = {'stage': stage}
        row.update({k: v for k, v in e.items() if not isinstance(v, (dict, list))})
        episode_rows.append(row)


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


(OUT / '最新全部对比结果.md').write_text('\n\n'.join(parts) + '\n', encoding='utf-8')
write_csv(OUT / 'rough_and_flat_method_summary.csv', grouped_rows)
write_csv(OUT / 'all_scored_episodes.csv', episode_rows)
manifest = {'generated_at': datetime.now(timezone.utc).isoformat(),
            'operation': 'read-only aggregation of existing sealed analyses; no sampling or statistical refit',
            'sources': sources, 'rough_and_flat_groups': grouped_rows,
            'locked_statistics': {s: data[s].get('statistics', data[s].get('contrasts')) for s in STAGES},
            'validation': 'PASS: audit-to-analysis hashes and balanced core rough groups/seed means',
            'episode_csv_rows': len(episode_rows),
            'episode_rows_are_not_unique_independent_trials': True}
(OUT / 'summary_provenance.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({'status': 'PASS', 'groups': len(grouped_rows), 'csv_episode_rows': len(episode_rows),
                  'report': str(OUT / '最新全部对比结果.md')}, ensure_ascii=True))
