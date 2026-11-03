"""Linux entry point for the unchanged frozen-E flat PPO comparison."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODULE = 'scripts.run_g1_stateful_flat_ppo_remote_20261025'


def install():
    from scripts import g1_stateful_flat_ppo as study
    from scripts import run_g1_stateful_flat_ppo_20261025 as runner
    from scripts.run_g1_stateful_remote_20261025 import bootstrap
    study.base.bootstrap = bootstrap
    study.previous.bootstrap = bootstrap
    runner.MODULE = MODULE
    if not getattr(study.asset_paths, '_linux_entry_point', False):
        original = study.asset_paths
        def assets():
            return original() + [Path(__file__).resolve(),
                ROOT/'scripts/run_g1_stateful_remote_20261025.py',
                ROOT/'tests/test_g1_stateful_flat_ppo_remote.py']
        assets._linux_entry_point = True
        study.asset_paths = assets
    return runner


if __name__ == '__main__':
    install().main()
