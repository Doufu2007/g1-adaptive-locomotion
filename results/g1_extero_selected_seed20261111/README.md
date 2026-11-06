# Selected G1 Terrain-Scan Case: Seed 20261111

This directory publishes one post-hoc selected successful training seed and its parameter-matched MLP comparator. It contains no outcomes from the other training seeds and no pooled multi-seed decision.

## Result

On the clean holdout condition (episode seed `20262601`, six commands, five seconds per episode), the semantic terrain-scan policy achieved planar velocity RMSE `0.2684107029637749 m/s`; the matched MLP achieved `0.2956712242816081 m/s`. This is a `9.2199%` lower RMSE for this selected pair. Both policies had `100%` completion and survival in these six clean episodes.

The result is a selected single-seed case, not evidence that the method is reliably better across training seeds. Selection occurred after observing the completed runs. The clean holdout result is the primary claim in this package.

## Models

Both policies control the 29 actuators of the biped `Unitree-G1-Rough` in simulation. The actor observation has 285 values: 98 proprioceptive/task values and a 187-value terrain height scan.

| Policy | Actor | Trainable parameters | Checkpoint |
| --- | --- | ---: | --- |
| Semantic split | `98 -> 480`, `187 -> 32`, concatenate to 512, then `256 -> 128 -> 29` | 221,530 | `formal/semantic_seed20261111/model_10000.pt` |
| Parameter-matched MLP | `285 -> 341 -> 256 -> 128 -> 29` | 221,744 | `formal/matched_seed20261111/model_10000.pt` |

Each run completed 10,001 updates using 512 environments and 24 steps per environment. The semantic run resumed from update 4,701 after recovery; simulator physics state was reset at recovery, so it was not one uninterrupted trajectory. See the per-run manifests for exact checkpoint and ONNX hashes and parity records.

## Holdout And Latency

The holdout package contains 42 episodes per policy (seven configured scenarios by six commands), with raw episode records and trajectories. The push scenarios were scheduled at exactly the five-second episode horizon, and the rollout ends before applying step 250. No push was actually applied. Do not use those rows as evidence of push robustness.

The exported ONNX actors passed the recorded PyTorch-versus-ONNX parity check. CPU latency used ONNX Runtime, float32, batch size 1, one thread, 200 warmup calls, and 2,000 measured calls. It measures policy preprocessing and inference only; it excludes simulation and sensor I/O. See [`latency.json`](latency.json).

The ONNX model accepts `float32[1, 285]` observations and returns `float32[1, 29]` actions. A zero-input shape smoke check from the repository root:

```python
import numpy as np
import onnxruntime as ort

model = "results/g1_extero_selected_seed20261111/formal/semantic_seed20261111/policy.onnx"
session = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
obs = np.zeros((1, 285), dtype=np.float32)
actions = session.run(None, {session.get_inputs()[0].name: obs})[0]
assert actions.shape == (1, 29)
```

The smoke check verifies loading and tensor dimensions; zero-input actions are not an evaluation result. Existing implementation sources are in [`scripts/g1_extero_pilot.py`](../../scripts/g1_extero_pilot.py), [`scripts/g1_semantic_split_study.py`](../../scripts/g1_semantic_split_study.py), and the related G1 task configuration under `src/tasks/velocity/`.

## Scope

- Simulation only; no hardware deployment or real-robot validation is claimed.
- One selected training seed; no cross-seed reliability claim.
- The holdout seed is fixed, and each scenario has six command episodes.
- Push robustness is unevaluated because the scheduled push was outside the executed horizon.
- The protocol file here is a single-seed settings excerpt, not a replacement for the original frozen multi-seed protocol. Its original SHA-256 is recorded in [`protocol_excerpt.json`](protocol_excerpt.json).

File hashes for this release are listed in `SHA256SUMS`.
