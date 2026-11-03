"""Backup and shutdown gates must fail closed on incomplete or corrupt results."""
import hashlib
import io
import json
import sys
import tarfile

import pytest

from scripts import g1_best_e_delivery_20261026 as delivery


def test_completion_gate_rejects_missing_runs_wrong_encoder_or_pending_queue():
    audit = dict(status='PASS', training_runs=6, evaluations=9, episodes=270,
                 updates_per_run=501, transitions_per_run=769536,
                 training_pairs_verified=3, evaluation_initial_state_pairs=180,
                 encoder_frozen=True)
    delivery.require_complete({'status': 'COMPLETED'}, audit, delivery.ENCODER_SHA)
    for bad in (dict(audit, training_runs=5), dict(audit, episodes=269),
                dict(audit, encoder_frozen=False)):
        with pytest.raises(ValueError):
            delivery.require_complete({'status': 'COMPLETED'}, bad, delivery.ENCODER_SHA)
    with pytest.raises(ValueError):
        delivery.require_complete({'status': 'RUNNING'}, audit, delivery.ENCODER_SHA)
    with pytest.raises(ValueError):
        delivery.require_complete({'status': 'COMPLETED'}, audit, '0'*64)


def test_download_verifies_embedded_manifest_and_each_file(tmp_path):
    data = b'checkpoint and raw trajectory'
    manifest = {'files': [{'path': 'results/example.bin', 'size': len(data),
                          'sha256': hashlib.sha256(data).hexdigest()}]}
    archive = tmp_path/'export.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        for name, content in [('results/example.bin', data),
                              ('export_manifest.json', json.dumps(manifest).encode())]:
            info = tarfile.TarInfo(name); info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
    digest = delivery.sha(archive)
    assert delivery.extract_verified(archive, manifest, digest, tmp_path/'good') == 1
    with pytest.raises(ValueError):
        delivery.extract_verified(archive, manifest, '0'*64, tmp_path/'bad')
    bad = {'files': [dict(manifest['files'][0], sha256='0'*64)]}
    with pytest.raises(ValueError):
        delivery.extract_verified(archive, bad, digest, tmp_path/'badfile')


@pytest.mark.skipif(sys.platform != 'win32', reason='Windows MAX_PATH regression')
def test_download_extracts_long_tensorboard_filename_on_windows(tmp_path):
    name = 'results/training/' + 'a' * 90 + '/events.out.tfevents.' + 'b' * 90
    data = b'complete event log'
    manifest = {'files': [{'path': name, 'size': len(data),
                          'sha256': hashlib.sha256(data).hexdigest()}]}
    archive = tmp_path/'long_export.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        for path, content in [(name, data),
                              ('export_manifest.json', json.dumps(manifest).encode())]:
            info = tarfile.TarInfo(path); info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
    destination = tmp_path/'snapshot'
    assert len(str(destination/name)) > 260
    assert delivery.extract_verified(archive, manifest, delivery.sha(archive), destination) == 1
