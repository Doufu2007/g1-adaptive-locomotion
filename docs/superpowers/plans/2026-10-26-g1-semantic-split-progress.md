# 执行账本：2026-10-26-g1-semantic-split.md

- 初始 commit：1425b15；分支 codex/g1-semantic-split；已复制当前用户源码依赖，原工作区保留。
- Task 1: completed — 冻结协议、真实字段索引、四 Actor、指标和统计；完整 21 项检查通过。
- Task 2: completed — 接入 RSL-RL Gaussian/归一化与 ONNX；四组最终版本均已通过真实工程训练和导出。
- Task 3: implementation completed / scientific execution blocked — 配置、软件环境、来源 SHA、checkpoint、RNG 恢复与预算校验已实现。每组 1,000 更新 pilot 及 12 次正式训练未启动。
- Task 4: implementation completed — 七场景、配对种子、固定时域、失败补值、原始轨迹、CPU 延迟及训练级分析。最终四组共 28 段工程评估通过、四组 CPU 计时完成；尚无正式方法效果结论。
- Task 5: implementation and packaging completed — 串行队列、GPU 准入、日志、停止状态、运行说明与可迁移包已交付；485 文件 ZIP 的 CRC 和封存源码逐项 SHA 校验通过。正式实验待执行，保留全部失败尝试。
- Task 6: completed — 独立只读审查发现三处问题，均修复并补回归检查；二次复核确认关闭，无新明确问题。
- 算力核对：本机 RTX 4050 6GB，其他研究进程活跃；三台历史云端 SSH 端口拒绝连接；已异步询问当前可用实例。
- 查新核对：Firecrawl 未登录，公开 API 通过单请求直连恢复；8 次查询 61 条未去重记录，核对 7 篇全文。已发现本体历史／指令编码近邻，具体组合新颖性仍为 UNKNOWN，见 literature/REVIEW.md。
- 审查修复：队列恢复区分新任务／已有任务；身体横向推扰旋转到世界坐标；恢复和分析时从绑定身份的 NPZ 重算指标，并核验 JSON/NPZ hash 与覆盖。
- 下一步：获得可用 GPU 后执行 README 的 queue 命令；以 GPU pilot 实测替换历史耗时外推。此处未声明实验研究已完成。
- 本机可行性更正：已发现真实历史 1,024 环境／51 更新／280.7 秒记录。本机可以运行，22GB 只是原方案保守准入值；用户正在选择本机缩小采样预算或保持总采样量，尚未更改冻结协议。
- 2026-11-03 授权更新：用户选定 1,024 环境 × 10,001 更新、四 Actor × 三种子的首轮正式训练；总采样量为原方案 1/4，见云端目录 resource_amendment.md。其他科学常量沿用原协议。
- 2026-11-03 实例切换：当前端点 connect.weste.seetacloud.com:16963；远端目录与环境保留，旧训练/vLLM进程均已不存在，本任务没有停止服务。已重跑原 21 项检查并恢复 engineering pilot；original/matched 已完成，semantic 从401更新恢复，random随后。
- 2026-11-03 新正式适配器：formal_1024.py 独立输出、显式协议修订、等待四组pilot验证、12次正式从零训练、全部封存后holdout、延迟/分析/打包。新增预算/子进程/准入/禁止pilot复用等检查先失败后通过。
- 独立增量审查：发现1项P2打包恢复问题；先写测试复现既有归档导致恢复误报失败，再实现封存文件清单、逐文件/归档内容hash验证和一致复用。该回归转绿，新增适配器5项测试通过。实际启动状态另见云端 queue_status 快照；不以调度完成冒充研究完成。
