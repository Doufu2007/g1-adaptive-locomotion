"""Score the frozen 360 episodes, respecting ten-seed batch shards (v3 repair)."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

from scripts import summarize_g1_selected_phase_control_v2 as old
from scripts.summarize_g1_best_phase_control import aggregate


def validate_batch_manifests(manifests):
    if len(manifests)!=3:
        raise ValueError("expected three ten-seed batches")
    for index,m in enumerate(manifests):
        expected_seeds=list(old.SEEDS[index*10:(index+1)*10])
        if (m["seeds"]!=expected_seeds or tuple(m["scenarios"])!=old.SCENARIOS
                or m["horizon"]!=20. or m["parallel_worlds"]!=10):
            raise ValueError("batch protocol mismatch")


def claim_rule_passed(inference):
    return bool(inference["rough"]["fall_rate"]["bootstrap_95pct_ci"][1]<0
        and inference["rough"]["speed_mae"]["bootstrap_95pct_ci"][1]<0
        and inference["flat"]["speed_mae"]["bootstrap_95pct_ci"][1]<=0)


def main():
    # Preserve the frozen evaluator and raw telemetry; repair aggregation only.
    from importlib.util import module_from_spec, spec_from_file_location
    path = old.OUT/"continue_missing_rough_batches_20261024.py"
    spec = spec_from_file_location("_verify_selected_study",path)
    verifier = module_from_spec(spec)
    spec.loader.exec_module(verifier)
    verifier.verify_frozen_assets()
    results,checks,aggregates,inference = [],[],[],{}
    for terrain in ("flat","rough"):
        arms = {arm:old.collect(terrain,arm) for arm in ("clock","learned")}
        cm,cr,ct = arms["clock"]
        lm,lr,lt = arms["learned"]
        for manifests in (cm,lm):
            validate_batch_manifests(manifests)
        if len({m["policy_sha256"] for m in cm+lm}) != 1:
            raise ValueError("actor weights differ in direct-swap study")
        if {m["encoder_sha256"] for m in cm+lm} != {old.sha(old.RUNTIME)}:
            raise ValueError("selected encoder mismatch")
        if len({json.dumps(m["startup_model_hashes"],sort_keys=True) for m in cm+lm}) != 1:
            raise ValueError("startup physics mismatch")
        cb = {(r["scenario"],r["seed"]):r for r in cr}
        lb = {(r["scenario"],r["seed"]):r for r in lr}
        for key in sorted(ct):
            a,b = ct[key],lt[key]
            errors = {k:float(np.max(np.abs(a[k][0].astype(float)-b[k][0].astype(float))))
                      for k in ("position","quaternion","joint_pos","velocity")}
            if max(errors.values()) > 1e-6:
                raise ValueError(f"unmatched initial states: {key}")
            n = min(len(a["command"]),len(b["command"]))
            np.testing.assert_array_equal(a["command"][:n],b["command"][:n])
            np.testing.assert_array_equal(cb[key[1:]]["initial_env_origin"],lb[key[1:]]["initial_env_origin"])
            checks.append(dict(terrain=terrain,scenario=key[1],seed=key[2],initial_state_max_errors=errors))
        inference[terrain] = dict(
            fall_rate=old.cluster_bootstrap_differences(cr,lr,"fall_rate"),
            speed_mae=old.cluster_bootstrap_differences(cr,lr,"mae_forward_velocity_mps"))
        for arm,rows in (("clock",cr),("learned",lr)):
            results.extend(rows)
            for scenario in ("all",*old.SCENARIOS):
                subset=[r for r in rows if scenario=="all" or r["scenario"]==scenario]
                aggregates.append(dict(terrain=terrain,arm=arm,scenario=scenario,**aggregate(subset)))
    passed=claim_rule_passed(inference)
    result=dict(schema="g1.selected_phase_control_comparison.v3",computed_at=datetime.now(timezone.utc).isoformat(),
        aggregation_repairs=["ten-seed manifest shards compared with their own cohort","terrain-keyed inference accessed consistently"],
        scoring_script_sha256=old.sha(Path(__file__)),runtime_sha256=old.sha(old.RUNTIME),
        primary_inference=inference,predeclared_claim_rule_passed=bool(passed),
        aggregates=aggregates,pairing_checks=checks,episodes=results)
    (old.OUT/"comparison.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    lines=["# 选中 TCTST 与固定 clock 的完整直接替换对照","",
        "每臂每地形 30 seed × 3 场景，20 秒；同 actor、同初态、同物理参数。跌倒后剩余时间按零速评分。","",
        "| 地形 | 输入相位 | 回合 | 跌倒 | 存活(s) | 速度 MAE(m/s) |",
        "|---|---|---:|---:|---:|---:|"]
    for r in aggregates:
        if r["scenario"]=="all":
            lines.append(f"| {r['terrain']} | {r['arm']} | {r['episodes']} | {r['falls']} | {r['survival_seconds']:.3f} | {r['mae_forward_velocity_mps']:.5f} |")
    lines += ["",f"预先冻结的改善判据：{'通过' if passed else '未通过'}。",
        "", "这是原 clock actor 的直接替换研究；不能据此判断等预算策略适配后的效果。",
        "", "汇总 v3 修复了每批 10 seed 与总计 30 seed 的核验混淆、推断字典索引错误；未修改原始轨迹、逐回合评分和判据。",""]
    (old.OUT/"comparison.md").write_text("\n".join(lines),encoding="utf-8")
    print(json.dumps(dict(primary_inference=inference,predeclared_claim_rule_passed=bool(passed)),indent=2))


if __name__=="__main__":
    main()
