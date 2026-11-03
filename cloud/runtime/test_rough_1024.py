import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


def adapter():
    path = Path(__file__).with_name('rough_1024.py')
    assert path.is_file(), 'rough execution adapter is missing'
    spec = importlib.util.spec_from_file_location('rough_under_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rough_amendment_preserves_approved_budget_and_records_separate_cohort():
    m = adapter()
    p = m.amended_protocol()
    assert p['task'] == 'Unitree-G1-Rough'
    assert p['terrain_mode'] == 'blind_rough'
    assert p['training']['num_envs'] == 1024
    assert p['training']['formal_iterations'] == 10001
    assert len(m.execution.study.training_jobs(p, 'formal')) == 12
    assert p['terrain_amendment']['reuse_flat_weights'] is False
    assert p['terrain_amendment']['learned_phase_factor_in_this_cohort'] is False
    assert p['terrain_amendment']['adapter_sha256'] == m.execution.study.sha(m.HERE)


def test_rough_child_prepares_same_protocol_in_independent_directory(tmp_path):
    m = adapter()
    result = subprocess.run(m.worker_command(tmp_path, 'prepare'),
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    p = json.loads((tmp_path/'formal_1024/protocol.json').read_text())
    assert p == m.amended_protocol()


def test_flat_pilots_cannot_satisfy_rough_engineering_gate(tmp_path):
    m = adapter()
    root = tmp_path/'rough_smoke_1024'
    root.mkdir()
    (root/'smoke_gate.json').write_text(json.dumps({'status': 'COMPLETED',
        'source_sha256': 'flat-source', 'protocol_sha256': 'flat-protocol'}))
    with pytest.raises(ValueError, match='identity'):
        m.verify_rough_smokes(tmp_path)
