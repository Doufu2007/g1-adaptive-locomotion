"""Descriptive analysis of the immutable 240-case G robot experiment.

No training, simulation, record replacement, significance selection, or outcome
exclusion. Each plotted dot is one training seed after averaging its three
matched evaluation repeats. Raw records and sample denominators are retained.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import shutil
import statistics as st
from datetime import datetime, timezone
from pathlib import Path
import sys

PROTOCOL = "3fc04aa37954e40615936cd0b53a225816156eb756067c60bb96475f2da10c92"
SCENARIOS = ("nominal", "left_knee_75pct_0p4s", "right_ankle_75pct_0p4s", "right_ankle_100pct_0p8s")
NAMES = ("正常", "左膝75%/0.4秒", "右踝75%/0.4秒", "右踝100%/0.8秒")
MODES = {"A": "none", "B": "fixed", "C": "global", "D": "selective"}
METRICS = {
    "survival_rate": ("20秒最终存活率", True),
    "first_life_xy_rmse_mps": ("首生命速度RMSE(m/s)", False),
    "fault_alive_xy_rmse_mps": ("故障期存活速度RMSE(m/s)", False),
    "first_life_reference_rmse_scaled": ("首生命参考误差(scaled)", False),
    "fault_alive_unaffected_joint_rmse_scaled": ("故障期未故障关节参考误差(scaled)", False),
    "recovery_without_fall_rate": ("跌倒前首次达到恢复窗比例", True),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def csv_rows(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def flatten(value, prefix=""):
    out = {}
    for key, item in value.items():
        k = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            out.update(flatten(item, k))
        elif isinstance(item, list):
            out[k] = json.dumps(item, ensure_ascii=False)
        else:
            out[k] = item
    return out


def describe(values):
    return {"n": len(values), "mean": st.mean(values), "sd": st.stdev(values) if len(values) > 1 else 0.,
            "min": min(values), "max": max(values), "median": st.median(values)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--alignment-helper", type=Path)
    args = ap.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    envelope = json.loads(args.input.read_text(encoding="utf-8"))
    records = envelope["evaluation"]["records"]
    expected = {(g, t, e, s) for g in MODES for t in range(42, 47) for e in (1042, 1043, 1044) for s in SCENARIOS}
    keyed = {(r["group"], r["train_seed"], r["evaluation_seed"], r["scenario"]): r for r in records}
    assert len(records) == len(keyed) == 240 and set(keyed) == expected
    assert envelope["evaluation"]["status"] == "completed"
    assert envelope["protocol"]["sha256"] == PROTOCOL
    for r in records:
        assert r["protocol_sha256"] == PROTOCOL and r["training_completed_updates"] == 3000
        assert r["num_envs"] == 128 and r["duration_s"] == 20 and r["mode"] == MODES[r["group"]]
        for sample in r["valid_samples"].values():
            assert sample["frame_environment_samples"] == sample["finite_frame_environment_samples"]
        for m in METRICS:
            assert r.get(m) is None or math.isfinite(r[m])
    write_json(out / "raw_final_evaluation.json", envelope["evaluation"])
    write_json(out / "frozen_protocol.json", envelope["protocol"])
    write_json(out / "final_checkpoints.json", envelope["checkpoints"])
    csv_rows(out / "raw_cases_240.csv", [flatten(r) for r in records])
    seed_rows, stats, paired = [], [], []
    by_seed = {}
    for scenario in SCENARIOS:
        for group in MODES:
            for seed in range(42, 47):
                rows = [keyed[group, seed, e, scenario] for e in (1042, 1043, 1044)]
                row = {"scenario": scenario, "group": group, "train_seed": seed, "evaluation_repeats": 3,
                       "descriptive_trajectory_count": 384, "falls_total": sum(r["falls"] for r in rows)}
                for m in METRICS:
                    v = [r.get(m) for r in rows]
                    assert all(x is None for x in v) or all(x is not None for x in v)
                    row[m] = None if v[0] is None else st.mean(v)
                by_seed[group, seed, scenario] = row
                seed_rows.append(row)
            for m in METRICS:
                v = [by_seed[group, t, scenario][m] for t in range(42, 47)]
                stats.append({"scenario": scenario, "group": group, "metric": m,
                              "training_seeds": 5, "evaluation_repeats_per_seed": 3,
                              **({"applicable": False} if v[0] is None else {"applicable": True, **describe(v)})})
        for base in "ABC":
            for m, (_, higher) in METRICS.items():
                pairs = [(t, by_seed["D", t, scenario][m], by_seed[base, t, scenario][m]) for t in range(42, 47)]
                if pairs[0][1] is None:
                    continue
                diff = [d - b for _, d, b in pairs]
                favorable = sum((z > 1e-12 if higher else z < -1e-12) for z in diff)
                tied = sum(abs(z) <= 1e-12 for z in diff)
                paired.append({"scenario": scenario, "comparison": f"D-{base}", "metric": m,
                               "higher_is_better": higher, **describe(diff), "favorable_seeds": favorable,
                               "tied_seeds": tied, "unfavorable_seeds": 5 - favorable - tied,
                               **{f"seed{t}_difference": d - b for t, d, b in pairs}})
    csv_rows(out / "training_seed_means.csv", seed_rows)
    csv_rows(out / "group_descriptives.csv", stats)
    csv_rows(out / "paired_seed_differences.csv", paired)
    result = {"input_sha256": sha(args.input), "analysis": "complete descriptive five-training-seed paired analysis",
              "aggregation": "mean of the three matched evaluation RMSE/rate records within each training seed, then mean and sample SD over five training seeds; no pooled-frame RMSE",
              "inference": "descriptive only; no new significance/noninferiority/success threshold and no post-hoc filtering",
              "raw_cases": 240, "training_seeds": 5, "group_statistics": stats, "paired_differences": paired,
              "created_at_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version()}
    write_json(out / "analysis_summary.json", result)
    write_json(out / "valid_sample_denominators.json", [{k: r[k] for k in ("group", "train_seed", "evaluation_seed", "scenario", "survival_rate", "falls", "fault_entry_survivors", "fault_exit_survivors", "recovery_eligible_envs", "valid_samples")} for r in records])
    figs = plot(out, by_seed, paired, args.alignment_helper)
    report(out, envelope, records, stats, paired, figs)
    write_json(out / "analysis_manifest.json", {"at_utc": datetime.now(timezone.utc).isoformat(),
               "input_sha256": sha(args.input), "files": [{"name": str(p.relative_to(out)), "bytes": p.stat().st_size, "sha256": sha(p)} for p in sorted(out.rglob("*")) if p.is_file() and p.name != "analysis_manifest.json"]})
    print(json.dumps({"status": "completed", "cases": 240, "seed_mean_rows": len(seed_rows), "paired_rows": len(paired), "output": str(out)}, ensure_ascii=False))


def plot(out, by_seed, paired, helper):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    if helper:
        shutil.copyfile(helper, out / "audit_panel_alignment.py")
    sys.path.insert(0, str(out))
    from audit_panel_alignment import require_matplotlib_panel_alignment
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "svg.fonttype": "none", "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
    scenario = SCENARIOS[-1]
    metrics = [("survival_rate", "Final survival (%)", 100),
               ("first_life_xy_rmse_mps", "First-life speed RMSE (m/s)", 1),
               ("fault_alive_unaffected_joint_rmse_scaled", "Unaffected-joint error (scaled)", 1),
               ("recovery_without_fall_rate", "First recovery attainment (%)", 100)]
    colors = ["#7D8994", "#638CA5", "#A69581", "#508B80"]
    exported = []
    for kind in ("severe_absolute", "severe_paired"):
        fig, axes = plt.subplots(2, 2, figsize=(183/25.4, 142/25.4))
        fig.subplots_adjust(left=.12, right=.98, top=.91, bottom=.13, wspace=.48, hspace=.63)
        for letter, ax, (metric, label, scale) in zip("abcd", axes.flat, metrics):
            ax.text(-.19, 1.13, letter, transform=ax.transAxes, fontweight="bold", fontsize=10)
            ax.set_ylabel(label)
            ax.grid(axis="y", color="#E7E9EB", linewidth=.5)
            ax.set_axisbelow(True)
            if kind == "severe_absolute":
                for j, group in enumerate(MODES):
                    values = [by_seed[group, t, scenario][metric] * scale for t in range(42, 47)]
                    ax.scatter(j + np.linspace(-.13, .13, 5), values, s=17, color=colors[j], zorder=3)
                    ax.errorbar(j, st.mean(values), yerr=st.stdev(values), fmt="_", color="#202A31", capsize=3, markersize=12, linewidth=1, zorder=4)
                ax.set_xticks(range(4), list(MODES))
                ax.set_xlim(-.5, 3.5)
                if scale == 100:
                    ax.set_ylim(88, 104)
                    ax.set_yticks([90, 95, 100])
                elif metric == "first_life_xy_rmse_mps":
                    ax.set_ylim(0, .22); ax.set_yticks([0, .1, .2])
                else:
                    ax.set_ylim(0, .8); ax.set_yticks([0, .2, .4, .6, .8])
            else:
                ax.axhline(0, color="#969DA3", linewidth=.8)
                for j, base in enumerate("ABC"):
                    row = next(r for r in paired if r["scenario"] == scenario and r["metric"] == metric and r["comparison"] == f"D-{base}")
                    values = [row[f"seed{t}_difference"] * scale for t in range(42, 47)]
                    ax.scatter(j + np.linspace(-.13, .13, 5), values, s=18, color=colors[j], zorder=3)
                    ax.errorbar(j, st.mean(values), yerr=st.stdev(values), fmt="_", color="#202A31", capsize=3, markersize=12, linewidth=1, zorder=4)
                ax.set_xticks(range(3), ["D - A", "D - B", "D - C"])
                ax.set_xlim(-.5, 2.5)
                ax.set_ylabel("D - control\n" + label.replace("(%)", "(pp)"))
        fig.text(.12, .97, "Right ankle: 100% target loss for 0.8 s", fontsize=10)
        fig.text(.12, .025, "Dots: 5 training seeds, each mean of 3 evaluation repeats. Bars: mean +/- seed SD.", fontsize=7)
        fig.canvas.draw()
        require_matplotlib_panel_alignment(fig, json_out=out/f"{kind}.alignment.json", overlay_svg=out/f"{kind}.alignment.svg", strict=True)
        fig.savefig(out/f"{kind}.pdf")
        fig.savefig(out/f"{kind}.svg")
        fig.savefig(out/f"{kind}.png", dpi=300)
        plt.close(fig)
        exported.append(kind)
    # The normal-condition check is a separate boundary panel, with full raw seeds.
    fig, ax = plt.subplots(figsize=(100/25.4, 83/25.4))
    fig.subplots_adjust(left=.23, bottom=.2, top=.86, right=.97)
    for j, group in enumerate(MODES):
        vals = [by_seed[group, t, "nominal"]["first_life_xy_rmse_mps"] for t in range(42, 47)]
        ax.scatter(j + np.linspace(-.13, .13, 5), vals, color=colors[j], s=20)
        ax.errorbar(j, st.mean(vals), yerr=st.stdev(vals), fmt="_", color="#202A31", capsize=3, markersize=12)
    ax.set_xticks(range(4), list(MODES)); ax.set_ylabel("First-life speed RMSE (m/s)")
    ax.set_ylim(0, .18); ax.set_yticks([0, .06, .12, .18]); ax.set_title("Normal condition")
    fig.text(.22,.03,"5 training seeds; 3 repeats each; mean +/- SD",fontsize=7)
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,json_out=out/"normal_speed.alignment.json",strict=True)
    fig.savefig(out/"normal_speed.pdf")
    fig.savefig(out/"normal_speed.svg")
    fig.savefig(out/"normal_speed.png", dpi=300)
    plt.close(fig)
    return exported + ["normal_speed"]


def report(out, envelope, records, stats, paired, figs):
    def cell(g, s, m, scale=1):
        row=next(r for r in stats if r["group"]==g and r["scenario"]==s and r["metric"]==m)
        return "不适用" if not row["applicable"] else f"{row['mean']*scale:.5f} ± {row['sd']*scale:.5f}"
    lines=["# G1间歇目标丢失实验：2026年11月3日阶段成果", "",
           "**实验训练与正式评价已经完成；选择性放松相对固定参考约束的稳定优势仍未得到充分支持，不能宣布原研究主张已经证成。**", "",
           "20个模型各从对应v1父2000更新追加1000到3000；四组×五训练种子×三匹配评价种子×四场景=240案例全部完成。每例128环境、20秒。正式评价完成UTC："+envelope["evaluation"]["at_utc"]+"。", "",
           "A无模仿约束，B固定模仿约束，C整体放松，D局部选择性放松与平滑恢复；C/D按相同trace匹配总权重质量。故障是新关节目标间歇丢失、保持上一已应用目标，并非物理关节角度锁死。训练仍是平地、单参考、固定速度，同时受影响1–2个下肢关节。", "",
           "## 统计口径", "",
           "先对每个训练种子的三个匹配评价记录等权取均值，再对五个训练种子计算均值与样本SD（ddof=1）。下表误差是每案例RMSE的均值，不是汇总原始帧后重算的RMSE。128环境和三评价种子均不是额外独立训练重复。未删除失败种子或真实跌倒，全部5配对差保存在CSV。CSV/JSON中的比例及比例差保留0–1 fraction单位，本文与图中分别乘100显示百分比及百分点；其余误差保留原单位。", "",
           "本报告是开发性、描述性分析；没有事后新增达标阈值、显著性检验或非劣性界限，不将未检出优势解释为等效或无效。样本仅五训练种子，范围/SD不是置信区间。", "",
           "## 四场景完整结果", "",
           "|场景|组|最终存活%（均值±种子SD）|首生命速度RMSE m/s|故障期未故障关节参考误差 scaled|首次恢复达标%|描述性跌倒次数|", "|---|---|---:|---:|---:|---:|---:|"]
    for s, name in zip(SCENARIOS,NAMES):
        for g in MODES:
            falls=sum(r["falls"] for r in records if r["group"]==g and r["scenario"]==s)
            lines.append(f"|{name}|{g}|{cell(g,s,'survival_rate',100)}|{cell(g,s,'first_life_xy_rmse_mps')}|{cell(g,s,'fault_alive_unaffected_joint_rmse_scaled')}|{cell(g,s,'recovery_without_fall_rate',100)}|{falls}|")
    lines += ["", "每组每场景为5模型×3评价×128=1920条模拟轨迹，仅作描述性分母。故障从6秒起，轻故障0.4秒、严重0.8秒。全场景指标、每案例分母、首生命参考误差及故障窗口速度误差见原始CSV和完整统计JSON。", "",
              "## 主张的证据与边界", "",
              "严重右踝中D与B的平均最终存活率相同（均约99.5833%），D的首生命速度及未故障关节参考误差均值略低；这些小差异和五种子间方向必须一起评估，不能仅用均值宣布选择性机制胜出。相对C，D平均存活、速度、参考指标较好，但C的单个较差种子影响较大；全部配对差而非挑选一个种子是解释依据。轻右踝中D有4次、B有2次跌倒，不能用严重场景的较好均值掩盖轻故障取舍。", "",
              "下表列严重故障的全部配对差摘要；差值均为D减对照，存活/恢复正值较好，误差负值较好。种子完整差值见paired_seed_differences.csv。", "",
              "|比较|指标|平均差|种子差最小–最大|有利/持平/不利种子|", "|---|---|---:|---:|---:|"]
    for p in paired:
        if p["scenario"]!=SCENARIOS[-1] or p["metric"] not in ("survival_rate","first_life_xy_rmse_mps","fault_alive_unaffected_joint_rmse_scaled","recovery_without_fall_rate"):continue
        scale=100 if p["metric"] in ("survival_rate","recovery_without_fall_rate") else 1
        suffix="百分点" if scale==100 else ""
        lines.append(f"|{p['comparison']}|{METRICS[p['metric']][0]}|{p['mean']*scale:.6f}{suffix}|{p['min']*scale:.6f}–{p['max']*scale:.6f}|{p['favorable_seeds']}/{p['tied_seeds']}/{p['unfavorable_seeds']}|")
    lines += ["", "首次恢复比例的代码含义：在首次跌倒之前曾达到连续0.1秒的相对速度恢复阈值窗；达到后再跌倒不会撤销该记录。因此它可能高于20秒最终存活率，不能写成全20秒无跌倒保证。恢复相对基线、阈值至少0.25m/s或基线加0.15m/s，也不代表绝对跟踪完美。", "",
              "故障窗口和故障后参考误差按存活样本计算，原则上存在筛选可能。本次全部故障案例的故障窗入口、出口均为128，故障窗样本完整，未发生窗内死亡选择；实际选择主要出现在首生命和故障后窗口。首生命误差虽排除重置后的新生命，较早跌倒仍会缩短分母。valid_sample_denominators.json保留每案例实际帧数、有限样本和恢复eligible数，最终存活与条件误差必须并列。参考误差为scaled代理，不能证明视觉自然性。", "",
              "软件机制、父检查点和精确恢复证据合格，证明实现按设定执行，不等于研究效果优于对照。G与v1同时改变追加预算和故障分布，仅作描述；E8不纳入公平主比较。单冻结参考、平地、固定速度、模拟器开发实验不能冒称独立参考验证、复杂地形、实机或普遍有效性。", "",
              "## 图表", "", "每点一个训练种子，先平均三个评价重复；黑横线和误差棒为五种子均值±样本SD。严重场景存活与恢复纵轴为88–104%，明确截取高存活区以便显示种子差，SD误差棒可超过100%但不代表实际概率超过100%；全部实际百分比另在表格给出。配对图单位为百分点或原误差单位，不使用显著性星号。"]
    for f in figs:lines += ["",f"![{f}]({f}.png)"]
    lines += ["", "## 交付状态", "", "训练20/20、正式评价240/240、描述性分析与图表已生成。离机完整成果包需另外核验全部ZIP、SHA256、CRC和必要成员覆盖；本报告本身不等于完整模型离机备份，不代表可释放云端实例。后续强化、换粗糙地形或新参考均延后，须另立协议。", "",
              "复现：Python运行analyze_g_stage.py，提供完整最终采集JSON与输出目录；绘图仅依赖matplotlib/numpy，随包提供audit_panel_alignment.py。原始记录、协议、最终模型清单、全部训练种子均值、组统计、配对差、分母和输入SHA均已保留。", "",
              "流程软件参考：Kassis T, Agarwal V, He Y, Patel D, Brueckner AM. Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents. 2026. arXiv:2609.00065, DOI:10.48550/arXiv.2609.00065（元数据2026-11-03核验；仅统计流程来源，不作为G1效果证据）。", ""]
    (out/"阶段成果_结论与限制.md").write_text("\n".join(lines),encoding="utf-8")


if __name__=="__main__":
    main()
