"""The Linux entry point must propagate to every queued GPU subprocess."""
import importlib
import subprocess


def test_remote_queue_child_preserves_linux_bootstrap_and_asset_identity(monkeypatch, tmp_path):
    remote = importlib.import_module('scripts.run_g1_stateful_flat_ppo_remote_20261025')
    from scripts import g1_stateful_flat_ppo as study
    from scripts import run_g1_stateful_flat_ppo_20261025 as runner
    from scripts import run_g1_stateful_remote_20261025 as linux
    monkeypatch.setattr(study.base, 'bootstrap', study.base.bootstrap)
    monkeypatch.setattr(study.previous, 'bootstrap', study.previous.bootstrap)
    monkeypatch.setattr(study, 'asset_paths', study.asset_paths)
    monkeypatch.setattr(runner, 'MODULE', runner.MODULE)
    original_assets = study.asset_paths()
    remote.install()
    assert study.base.bootstrap is linux.bootstrap
    assert study.previous.bootstrap is linux.bootstrap
    assets = study.asset_paths()
    assert all(path in assets for path in original_assets)
    assert study.ROOT/'scripts/run_g1_stateful_flat_ppo_remote_20261025.py' in assets
    assert study.ROOT/'scripts/run_g1_stateful_remote_20261025.py' in assets
    remote.install()
    assert study.asset_paths() == assets
    monkeypatch.setattr(study, 'OUT', tmp_path)
    monkeypatch.setattr(runner, 'wait_gpu', lambda: None)
    calls = []
    class Process:
        pid = 12345
        def __init__(self, command, **kwargs):
            calls.append((command, kwargs))
        def wait(self):
            return 0
    monkeypatch.setattr(subprocess, 'Popen', Process)
    runner.child('train', 'learned_test', arm='learned', seed=20261901)
    command, kwargs = calls[0]
    assert command[command.index('-m')+1] == remote.MODULE
    assert kwargs['env']['PYTHONHASHSEED'] == '20261901'
