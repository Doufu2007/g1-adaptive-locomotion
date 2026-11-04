# G1 extero terrain-scan pilot exploratory results

Scope: one training seed (20261105), 300 PPO updates per arm, 512 environments, and a 285-D actor input (98 base terms plus the native 187-D terrain scan). Each arm has 42 exploratory episodes: 6 velocity commands, 7 scenarios, episode seed 20262501, and a 5 s horizon. These are development rollouts, not a formal holdout.

| arm | completion | planar RMSE (all) | yaw RMSE (all) | survival (all) | clean planar RMSE | clean survival |
|---|---:|---:|---:|---:|---:|---:|
| original | 0.000 | 0.4606 | 0.3444 | 0.222 | 0.4454 | 0.231 |
| matched | 0.000 | 0.4515 | 0.3647 | 0.236 | 0.4430 | 0.239 |
| semantic | 0.000 | 0.4552 | 0.4396 | 0.105 | 0.4555 | 0.097 |
| random | 0.000 | 0.4936 | 0.3989 | 0.242 | 0.4775 | 0.256 |

All four arms had zero completion in this 5 s rough-terrain exploratory rollout. The semantic split also had the lowest survival fraction in this pass. With one training seed and universal early termination, these numbers diagnose an undertrained or mismatched pilot and do not support a semantic-split superiority claim.

## CPU ONNX latency

| arm | p50 (µs) | p95 (µs) | p99 (µs) | samples |
|---|---:|---:|---:|---:|
| original | 37.891 | 112.366 | 133.503 | 2000 |
| matched | 28.429 | 89.974 | 91.012 | 2000 |
| semantic | 28.740 | 85.179 | 116.811 | 2000 |
| random | 35.963 | 106.812 | 118.504 | 2000 |

Raw trajectories, per-arm summaries, policy hashes, protocol, and final checkpoints are stored beside this report. A second exploratory pass was run concurrently during debugging; it remains separate and is not pooled with this report.
