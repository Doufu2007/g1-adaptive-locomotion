"""A rerun must bind the selected E in every process and preserve the old study."""
import importlib
from pathlib import Path
import subprocess

import pytest
import torch


@pytest.fixture
def installed(monkeypatch):
    from scripts import g1_stateful_flat_ppo as study
    from scripts import run_g1_stateful_flat_ppo_20261025 as runner
    for obj, names in (
        (study, ('OUT', 'ENCODER_SEED', 'RUNTIME', 'RUNTIME_SHA', 'asset_paths', 'freeze_assets')),
        (study.base, ('bootstrap',)), (study.previous, ('bootstrap',)),
        (runner, ('MODULE', 'worker', 'summarize')),
    ):
        for name in names:
            monkeypatch.setattr(obj, name, getattr(obj, name))
    entry = importlib.import_module('scripts.run_g1_best_e_ppo_20261026')
    monkeypatch.setattr(entry, '_installed', False)
    entry.install()
    return entry, study, runner


def test_best_e_loads_exact_weights_and_keeps_matched_budget(installed):
    entry, study, runner = installed
    from scripts.g1_stateful_online import load_online
    assert study.ENCODER_SEED == 20261026
    assert study.RUNTIME_SHA == '8e6b15dcf409f51948f49c18fc04b686ba502e53586476db9c78d8c35a0fc346'
    assert study.sha(study.RUNTIME) == study.RUNTIME_SHA
    assert study.OUT.name == 'g1_stateful_flat_ppo_best_e_20261026'
    assert study.TRAIN_SEEDS == (20261901, 20261902, 20261903)
    assert study.TRANSITIONS == 769536
    torch.manual_seed(794)
    state = torch.get_rng_state().clone()
    runtime = load_online(study.RUNTIME, num_envs=1, device='cpu')
    assert torch.equal(state, torch.get_rng_state())
    assert not any(p.requires_grad for p in runtime.encoder.parameters())
    before = study.asset_paths()
    assert entry.ROOT/'scripts/run_g1_best_e_ppo_20261026.py' in before
    assert entry.install() is runner
    assert study.asset_paths() == before


def test_every_child_reinstalls_best_e_binding(installed, monkeypatch, tmp_path):
    entry, study, runner = installed
    monkeypatch.setattr(study, 'OUT', tmp_path)
    monkeypatch.setattr(runner, 'wait_gpu', lambda: None)
    calls = []
    class Process:
        pid = 4812
        def __init__(self, command, **kwargs):
            calls.append((command, kwargs))
        def wait(self):
            return 0
    monkeypatch.setattr(subprocess, 'Popen', Process)
    runner.child('train', 'best_e', arm='learned', seed=20261901)
    command, kwargs = calls[0]
    assert command[command.index('-m')+1] == entry.MODULE
    assert kwargs['env']['PYTHONHASHSEED'] == '20261901'


def test_best_e_seal_records_selection_and_amendment(installed, monkeypatch, tmp_path):
    entry, study, runner = installed
    monkeypatch.setattr(study, 'OUT', tmp_path)
    record = study.freeze_assets()
    assert record['encoder_selection'] == entry.SELECTION
    assert record['authorization'] == entry.AUTHORIZATION
    assert record['encoder_seed'] == 20261026
    assert record['encoder_sha256'] == study.RUNTIME_SHA
    assets = {r['path'] for r in record['assets']}
    assert entry.PROTOCOL.relative_to(entry.ROOT).as_posix() in {Path(p).as_posix() for p in assets}
    assert study.verify_assets() == record
