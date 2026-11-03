# 配对 McNemar 置信区间的修正（2026-10-31）

## 缺陷

`mcnemar()` 返回的 `ci95_pp` **只有应有宽度的一半**。三个分析脚本里是同一份代码：

- `scripts/g1_period_sweep_analysis_20261028.py`
- `scripts/g1_period_sweep_analysis_cpu_20261028.py`
- `scripts/g1_period_sweep_analysis_cpu_m_20261029.py`

出错的四行：

```python
share = a_only / nd
se = math.sqrt(share * (1 - share) / nd)      # 少了一个因子 2
point = (a_only - b_only) / n
ci95_pp = [100 * (point - 1.96 * se * nd / n),
           100 * (point + 1.96 * se * nd / n)]
```

## 推导

记不一致对 `nd = a_only + b_only`，其中学习独有 `a_only`、固定周期独有 `b_only`，
`q = a_only / nd`。比例差

    diff = (a_only - b_only) / n = (2q - 1) · (nd / n)

所以

    SE(diff) = 2 · (nd / n) · sqrt(q(1-q) / nd)

原代码漏掉的就是这个前导 2：方差小了 4 倍，**区间恰好窄一半**。

## 不受影响的部分

- 完成数 / 跌倒数（`a`、`b`）
- 精确 McNemar p 值（只用到不一致对计数）
- 逐图分解与地图层 CI（`map_breakdown()` 用的是另一套 t 分布公式，本身正确）
- 所有点估计 `diff_pp`
- E1 / E2 判决（判据是 p 值与完成数，不含 CI）

因此 **结论不变**，只有第一类表里的配对区间需要加宽。

## 已修正并重新生成

| 文件 | 状态 |
|---|---|
| `analysis_M_sweep_20261030.json` | 已用修正后代码重新生成 |
| `analysis_M_sweep_paired_20261030.json` | 已用修正后代码重新生成 |
| `analysis_M_sweep_20261030_ci_narrow_SUPERSEDED.json` | 修正前的原样归档（SHA 前缀 `06e80e3ae640de63`） |
| `analysis_M_sweep_paired_20261030_ci_narrow_SUPERSEDED.json` | 修正前的原样归档（SHA 前缀 `c3772f4822196895`） |

## 尚未改写的旧结果文件（冻结基线，不覆盖）

下列文件里存的仍是窄区间。**不要直接引用其中的 `ci95_pp`**，用右列修正值：

| 文件 / 字段 | 点估计 | 存储值（窄） | 修正值 |
|---|---|---|---|
| `analysis_M_cpu.json` /primary/completed | +19.33 pp | [+15.92, +22.74] | **[+12.52, +26.15]** |
| `analysis_M_cpu.json` /fell | −14.00 pp | [−17.59, −10.41] | **[−21.18, −6.82]** |
| `analysis.json` /A2_learned_vs_best_fixed/completed | −4.44 pp | [−11.12, +2.23] | [−17.79, +8.91] |
| `analysis.json` /A2_learned_vs_best_fixed/fell | −1.11 pp | [−7.73, +5.51] | [−14.35, +12.13] |
| `analysis.json` /A2_learned_vs_nominal/completed | +3.33 pp | [−3.27, +9.93] | [−9.87, +16.54] |
| `analysis_round1_20261028_backup.json` /A2_* | 同上 | 同上 | 同上 |

`g1_period_sweep_v1_gate_failed_20261028/analysis.json` 的四处区间都是 `nd = 0`（全部一致，区间本就为 [0,0]），不受影响。

## 备用口径

若改用多项式 Wald（不条件于 `nd`）：

    SE(diff) = sqrt((a_only + b_only) - (a_only - b_only)² / n) / n

结果与上面的条件口径只差约 0.3–0.5 pp（如 0.55 为 [+10.15, +24.52] vs [+10.51, +24.16]），
两种口径不改变任何显著性的判定。
