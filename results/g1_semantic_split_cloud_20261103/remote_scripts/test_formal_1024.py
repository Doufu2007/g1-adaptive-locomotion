"""Checks that the resource amendment cannot silently change the study."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts import g1_semantic_split_study as study


def adapter():
    path = Path(__file__).with_name('formal_1024.py')
    assert path.exists(), 'the approved 1024-environment formal adapter is missing'
    spec = importlib.util.spec_from_file_location('formal_adapter_under_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_amendment_changes_only_resource_budget_and_records_authorization():
    module = adapter()
    before = copy.deepcopy(module.ORIGINAL_PROTOCOL)
    after = module.amended_protocol()
    amendment = after.pop('resource_amendment')
    assert after['training']['num_envs'] == 1024
    assert after['training']['formal_iterations'] == 10001
    after['training']['num_envs'] = before['training']['num_envs']
    assert after == before
    assert amendment['formal_transitions_per_run'] == 245784576
    assert amendment['total_formal_transitions'] == 2949414912
    assert amendment['user_authorization'] == 'APPROVED'
    assert amendment['pilot_weights_used_for_formal'] is False
    assert amendment['adapter_sha256'] == study.sha(module.__file__)


def test_child_process_prepares_the_amended_protocol_not_original(tmp_path):
    module = adapter()
    result = subprocess.run(module.worker_command(tmp_path, 'prepare'),
                            capture_output=True, text=True, cwd=ROOT)
    assert result.returncode == 0, result.stderr
    protocol = json.loads((tmp_path/'formal_1024/protocol.json').read_text())
    assert protocol == module.amended_protocol()
    assert len(json.loads((tmp_path/'formal_1024/training_matrix.json').read_text())['formal']) == 12


def test_unknown_gpu_job_never_gets_admitted_even_with_free_memory():
    module = adapter()
    allowed = {123: 10.0}
    known = [{'pid': 123, 'created_at': 10.0}]
    assert module.capacity_decision(known, allowed, 2000, 1387) == 'READY'
    assert module.capacity_decision(known, allowed, 1000, 1387) == 'WAITING_FOR_GPU_CAPACITY'
    reused_pid = [{'pid': 123, 'created_at': 11.0}]
    assert module.capacity_decision(reused_pid, allowed, 32000, 1387) == 'WAITING_FOR_GPU_CAPACITY'


def test_legacy_pilot_cannot_be_resumed_as_formal(tmp_path):
    module = adapter()
    protocol = module.amended_protocol()
    from scripts import run_g1_semantic_split as runner
    pilot = {'arm': 'semantic', 'seed': 20262100, 'stage': 'pilot',
             'iterations': 1000, 'num_envs': 1024,
             'protocol_sha256': 'engineering-only', 'source_sha256': 'source'}
    study.write(tmp_path/'manifest.json', {'status': 'COMPLETED', 'identity': pilot})
    formal = dict(pilot, stage='formal', seed=20262101, iterations=10001,
                  protocol_sha256=study.protocol_digest(protocol))
    with pytest.raises(ValueError, match='identity differs'):
        runner.recovery_for_run(tmp_path, formal, requested=True)


def test_delivery_recovers_after_archive_exists_without_receipt(tmp_path):
    module = adapter()
    output = tmp_path/'formal_1024'
    output.mkdir()
    study.write(output/'protocol.json', module.amended_protocol())
    study.write(output/'source_manifest.json', {'sha256': 'fixture-source'})
    study.write(output/'completion.json', {'status': 'COMPLETED'})
    (output/'policy.onnx').write_bytes(b'final-policy-fixture')
    original = module.package_results(tmp_path, output)
    # The archive exists, but the supervisor died before writing its receipt.
    recovered = module.package_results(tmp_path, output)
    assert recovered['sha256'] == original['sha256']
    assert recovered['bytes'] == original['bytes']
    (output/'policy.onnx').write_bytes(b'changed-policy')
    with pytest.raises(ValueError, match='changed'):
        module.package_results(tmp_path, output)
