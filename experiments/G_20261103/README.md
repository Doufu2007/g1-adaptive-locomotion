# G: intermittent target-loss reference-constraint experiment (2026-11-03)

本批平地实验已完成：A/B/C/D × 5 个训练种子共 20 个模型，各由对应 v1 父模型 2000 更新追加至 3000；正式评价 240/240 案例（4 场景 × 20 模型 × 3 匹配评价种子），每例 128 环境、20 秒。

**实验完整，但“选择性放松 D 稳定优于固定约束 B”的研究强主张尚未证成。** 严重右踝场景 D/B 平均存活率均为 99.58333%，各有 8/1920 条描述性轨迹跌倒；速度与参考误差的优势不在五个训练种子中一致。低关节参考误差不能证明视觉自然性。

这是故障感知参考约束对照，和本仓库既有语义健康双流策略主线独立。G 为平地、固定速度、单开发参考；不代表粗糙地形、独立人体参考、实机或普遍有效性的证据。

## 阅读与复算

- `analysis/阶段成果_结论与限制.md`：完整中文结果、配对差、恢复指标定义和限制。
- `analysis/raw_cases_240.csv`、`training_seed_means.csv`、`paired_seed_differences.csv`：全案例与五训练种子配对结果。
- `evidence/protocol.json`、`checkpoints.json`：冻结协议、20 个最终模型身份；`evaluation.json`：完整评价记录。
- `analysis/analyze_g_stage.py`：从 `evidence/final_evaluation_envelope.json` 复算；依赖 numpy/matplotlib，绘图字体需另行配置。release 内也保留该封包和未修改的训练/评价源码、回执和日志。
- `public_manifest.json`：本次 Git 交付的独立文件清单；`PUBLICATION_EXCLUSIONS.json`：公开包排除项。

历史分析正文中的“离机待核验/粗糙延后”是生成当时的状态。完整 G 私有成果包已于北京时间 20:37:19 实际下载并通过 SHA256、ZIP CRC、803 成员哈希、20 模型与 240 案例覆盖核验。公开包是另一个有意删去受限数据的制品，不能替代完整私有备份。v1 全历史包不在这次核验范围。

## 数据与来源

**不分发原始或重定向人体动作参考。** `walking_reference.npz` 已排除；`provenance/walking_reference.json` 保留来源、版本、SHA256 与许可声明。其元数据标明原 LAFAN1 动作为 CC BY-NC-ND 4.0，重定向仓库另为 BSD-3-Clause，并明确不得分发适配后的参考。代码许可不能覆盖该动作数据许可。

源码以冻结实验中的字节保留，既有版权与上游许可继续适用；本次上传不重新授权第三方代码或数据。模型训练使用该受限开发参考；不得从模型可下载推断上游数据具有额外许可，使用者须遵守上游条款。模型已仅用静态 ZIP/pickletools 检查，未见嵌入参考动作的键或字符串；这不是权重许可的法律结论。

本包是实验源码快照和证据包，**不是完整可启动训练环境**。原云端冻结包缺少 `scripts/system_check.py`、`scripts/cloud_training_monitor.py`、完整依赖锁和机器人 XML/mesh；参考 NPZ 也有意排除。`source/UPSTREAM_AND_DEPENDENCIES.md` 逐项说明。源码及历史回执中的 `/root/...`、容器名等是历史基础设施路径，不是可用连接信息。原云端地址和凭据未包含在公开包中。无参考与环境补齐时，可复算已有记录，不能宣称完整重跑训练。

## 下一阶段

用户已决定本批上传后结束平地训练，后续采用独立版本的粗糙地形实验。不会在冻结 G 上改变地形、观测、奖励或预算；粗糙阶段尚未启动。
