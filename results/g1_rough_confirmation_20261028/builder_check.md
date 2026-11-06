# Fresh builder check — rough confirmation

状态：PASS（零采样静态检查）。

- 矩阵为 `RE_live`、`RE_frozen`、`RC_clock`、`FC_clock` × 3 个新粗糙地图；评测 seed 为 5 个新值，全部与 S/R 分离。
- 只读取 S 的 checkpoint 251、策略文件和 manifest；不训练、不更新 PPO/Adam。
- `RE_live` 使用原始 learned-phase 路径；`RE_frozen` 在每回合第一次相位输入时缓存该值，后续 actor 只接收缓存值，同时保存 live 与 used 两份相位轨迹。
- 粗糙地形由真实 hfield 构造，接触点按物理 hfield 几何和预绑定 patch 复核。
- 四个方法共用相同环境构造、命令、求解器、接触容量和 episode 预算；方法差异只在 phase policy 与 checkpoint 绑定。
- 新确认目录没有 queue sentinel、rollout、smoke 或 analysis；当前仍禁止启动正式队列。

已运行：Python 编译检查通过；5 个纯 CPU/契约测试通过；CPU preflight 通过（0 environment、0 eval transitions、0 PPO、0 Adam）。
