# 粗糙地形 GPU 资源诊断

2026-11-03 在 RTX 4080 SUPER、两个 vLLM 服务保持运行时完成。

- 原生 rough generator（1024 环境、CCD 500、接触上限 128）在 Warp CUDA graph 首次 forward 时 OOM；构造阶段成功，wrapper reset 失败。
- 500 次 CCD 的 EPA 临时数组约 8.1 GiB，仅该项就超过约 6 GiB 可用显存。
- 同一 1024 环境、1×5 `random_rough`、接触 128、约束 1500，CCD 50 完成 wrapper reset 和 48 步零动作仿真；峰值约 27,058 MiB，仍有约 5,173 MiB 空闲。
- 正式修订统一使用 CCD 50，并逐控制步检查 contact/constraint 容量；触顶即技术失败。

这只是资源与数值配置证据，不是策略性能结果。500 次 OOM 的失败 manifest、日志和旧协议目录在远端 `rough_smoke_500_oom_attempt/` 保留。
