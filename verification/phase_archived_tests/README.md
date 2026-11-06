# Archived PHASOR tests

These files preserve historical checks for implementation review. They include checks that require original models, sealed protocols, recorded trajectories, fixtures, or runtime layouts, so they are not the directly runnable test set for this public package. The `archive_` prefix excludes them from default pytest collection. Test values and logic are unchanged.

| Archived file | Original relative path |
| --- | --- |
| `archive_g1_rough_confirmation_20261028.py` | `tests/test_g1_rough_confirmation_20261028.py` |
| `archive_g1_rough_domain_20261028.py` | `tests/test_g1_rough_domain_20261028.py` |
| `archive_g1_source_anchor_20261027.py` | `tests/test_g1_source_anchor_20261027.py` |
| `archive_g1_stateful_flat_ppo.py` | `tests/test_g1_stateful_flat_ppo.py` |
| `archive_g1_stateful_flat_ppo_remote.py` | `tests/test_g1_stateful_flat_ppo_remote.py` |
| `archive_g1_stateful_phase.py` | `tests/test_g1_stateful_phase.py` |
| `archive_g1_stateful_remote_20261025.py` | `tests/test_g1_stateful_remote_20261025.py` |

To run an archived test, copy it back to `tests/` at the original relative path shown above and restore the `test_` filename. First restore its original model and fixture dependencies: baseline checkpoint, selected encoder, sealed training/rollout data, protocols, independent audit, directory layout, and runtime packages. Restore redacted local paths as needed. Original and public SHA256 hashes are mapped in `../phase_source_provenance.json`. Restore and inspect these dependencies in an isolated checkout before selecting tests.

Seven pure contract checks in the historical `test_g1_rough_domain_20261028.py::RoughDomainContractTests` passed, excluding `test_real_training_cfg_inserts_terrain_before_reset`. That result covers the finite design, routing, statistical contracts, and candidate gates only; it does not validate model restoration or real environment configuration.

The top-level `tests/` keeps the fully passing `test_g1_phase_supplement_20261031.py`, `test_g1_rough_transfer_20261028.py`, and `test_g1_phase_lr_factorial_20261027.py`: 27 checks passed. These archived files are historical implementation review material. Some call PPO updates, backpropagation, or runtime bootstrap and must be selected explicitly after their fixtures are restored.
