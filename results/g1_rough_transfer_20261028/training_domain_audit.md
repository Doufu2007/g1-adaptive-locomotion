# 训练域只读审计

12项固定501策略的env.yaml均为terrain_type=plane、terrain_generator=null；manifest均COMPLETED/501，模型与环境文件SHA均已登记。R的rough heightfield、物理数组和真实足地暴露在independent_audit.json中复核。没有新增环境采样或PPO。

这确定了平地续训→粗糙评测的地形分布差异；尚未确定它导致失败，更未解释E相对clock的额外差距。R缺少粗糙域训练对照，不能从R本身作因果结论。

当前磁盘可用约33.23GB；此为磁盘快照，不证明CUDA长期稳定。

下一项可证伪问题：在一次事前固定、两方法等预算粗糙适配后，持续行走和跌倒是否改善，E是否仍提供额外收益。须另行锁训练/评测地图隔离、相同预算和强clock；不改R阈值、beta、评测地图或选择checkpoint。当前只完成域审计，没有新训练、没有新采样。
