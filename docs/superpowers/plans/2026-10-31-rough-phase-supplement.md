# Rough phase supplement implementation

Goal: implement and execute the three user-authorized additions on rough terrain.
Spec: ../specs/2026-10-31-rough-phase-supplement-design.md
Architecture: new contract, isolated rollout/training runner, raw audit and single-worker queue; reuse frozen helpers read-only.
Tech Stack: Python 3.13, NumPy/PyTorch/mjlab/MuJoCo/ONNXRuntime, pytest.
Global constraints: no commits/cleanup/server/site-packages/old experiment changes; finite prelocked budget; E-drive outputs; stop on technical failure.
Review focus: state-independent open-loop phase, paired physical starts, new seed RNG, source actor and teacher, no test-driven selection, raw-derived analysis.

- [ ] 1. Fresh builder check of written spec, resolve routine choices; record rulings.
- [ ] 2. Contract tests RED then GREEN: schedules, phase interventions, split/budget, statistics.
- [ ] 3. Implement isolated raw rollout, calibrated bank, physical pairing and bounded perturbations.
- [ ] 4. Implement paired new rough training, fixed-period validation selection, fixed checkpoint testing.
- [ ] 5. CPU preflight, real GPU training smoke and CPU rollout/repro checks; separate accounting.
- [ ] 6. Fresh complete implementation review, fix findings, seal source/input hashes.
- [ ] 7. Start one bounded queue, verify actual sampling and persist handoff/monitor.
- [ ] 8. Complete independent raw/budget audit and report evidence when queue completes.

Ruling: user authorized the concrete experiment families and requested execution; routine finite design parameters are operator choices, documented before results. Do not re-request generic permission.
Ruling: no worktree transfer: only new uniquely named files are written, existing dirty workspace and absolute model imports remain intact, and no commit or cleanup is performed.
