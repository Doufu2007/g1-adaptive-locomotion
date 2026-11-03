# 粗糙地形候选的独立评测与反馈消融设计（草案）

状态：DESIGN_DRAFT_NOT_LOCKED；本草案不启动采样，不改变冻结S/D/R。

## 目的

验证S阶段通过探索门的候选：RE（rough域适配+学习相位）在新的粗糙地图和新的评测seed上是否仍优于RC（rough域适配+固定clock），并用同一RE checkpoint做live/frozen阶段反馈消融。

## 固定因素

- S最终checkpoint、encoder、critic、normalizer、reward、beta、solver和actor输入维度全部固定，按S三条source lineage逐一评测；不做PPO。
- 新rough地图：20264901、20264902、20264903；不得回流S/R训练或评测。
- 新评测seed：20264911–20264915；不得与S/R重合。每个source lineage×地图×seed×steady/push均配对。
- 方法：RE_live、RE_frozen（同RE权重，将learned phase冻结为episode初值）、RC_clock、FC_clock。每方法3 source×3地图×5 seed×2场景，共360回合，最多360000有效评测步，0 PPO/Adam。

## 预先锁定的主门

1. RE_live相对RC_clock：粗糙完成率差至少+0.10、跌倒率差至多−0.10；按地图聚合的3个family中至少2个方向有利；rough MAE、位移、倾斜、航向保护门全部通过。
2. RE_live相对RE_frozen：同样的完成/跌倒门和地图方向门；它是“近期运动反馈是否提供适应”的直接消融。
3. FC_clock只作为跨域锚点，不把跨域差异解释为phase因果。

统计单位是3条source lineage与3个地图family，不把360回合当360个独立训练；精确双侧p最小为0.25，family比较统一Holm。任何CI跨零照实报告。

## 原始记录与审计

每回合保存完整20秒轨迹或第一次native终止后的零尾、phase输入/learned-phase内部值/clock值、接触点与指定patch内标记、初态、物理/hfield数组、policy和checkpoint SHA。独立CPU审计在结果后复算raw SHA、模型绑定、初态配对、phase干预正确性、物理与统计门。

## 失败树与停止

smoke、初态/物理/phase输入或接触边界任一失败即保留目录并停止；不得改地图、seed、阈值、beta、模型或追加预算追赢。若任一主门失败，结论为候选未获独立确认；不再追加S预算。

反馈消融实现必须先通过源码绑定、CPU零采样测试、fresh review和定点smoke；实现审查通过前不封存协议。
