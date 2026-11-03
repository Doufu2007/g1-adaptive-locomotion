# 阶段C执行计划
Spec: docs/superpowers/specs/2026-10-27-g1-phase-policy-cross-design.md

Task1 冻结可审计矩阵与路由：先测试固定12组合、9槽按seed路由、重复/缺失结果拒绝、同phase配对六项比较，再实现最小函数。接口不改变A/B。
Task2 实现冻结9world分组rollout：加载每slot原policy，actor/critic phase替换为指定原输入，score/原始SHA/初始物理配对和seed聚合。先unit/语法/source审查通过，再锁协议，0新增训练。
Task3 顺序固定12group执行，保存进度/每组日志，全部完成自动analysis+diagnosis。复核108回合最大108000transition、全部12cells覆盖、source不冒充独立训练。按预定六项机制对照决定单因素修正，尚无候选不启动独立验证。
