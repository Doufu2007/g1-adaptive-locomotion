"""A server may shut down only after a complete, verified local delivery."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile

import pytest


@pytest.fixture
def remote_delivery():
    path = Path(__file__).resolve().parents[1] / "results/g1_stateful_phase_20261025/remote_migration/prepare_delivery.py"
    spec = importlib.util.spec_from_file_location("remote_delivery_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def delivery():
    path = Path(__file__).resolve().parents[1] / "results/g1_stateful_phase_20261025/remote_migration/download_then_shutdown.py"
    assert path.exists(), "verified download/shutdown workflow is missing"
    spec = importlib.util.spec_from_file_location("stateful_delivery_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def complete_audit():
    return dict(status="PASS", training_runs=15, updates_per_run=2000,
                evaluations=15, test_episodes=24, test_after_model_freeze=True,
                original_cohort=[603, 504, 99], sealed_assets_verified=True,
                active_experiment_processes=[], gpu_processes=[])


@pytest.mark.parametrize("field,value", [
    ("training_runs", 14), ("updates_per_run", 1999), ("evaluations", 14),
    ("test_episodes", 23), ("test_after_model_freeze", False),
    ("sealed_assets_verified", False), ("original_cohort", [603, 503, 100]),
    ("active_experiment_processes", [123]), ("gpu_processes", [123]),
])
def test_incomplete_or_busy_server_cannot_pass_shutdown_gate(delivery, field, value):
    audit = complete_audit()
    audit[field] = value
    with pytest.raises(ValueError):
        delivery.require_complete(audit)


def archive_fixture(tmp_path, name="results/experiment/comparison.json"):
    data = b'{"status":"COMPLETED"}\n'
    manifest = dict(audit=complete_audit(), files=[dict(path=name, size=len(data),
                    sha256=hashlib.sha256(data).hexdigest())])
    archive = tmp_path / "delivery.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for path, content in ((name, data), ("delivery_manifest.json", json.dumps(manifest).encode())):
            info = tarfile.TarInfo(path)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
    expected = hashlib.sha256(archive.read_bytes()).hexdigest()
    return archive, manifest, expected


def test_tampered_archive_cannot_create_verified_backup(delivery, tmp_path):
    archive, manifest, expected = archive_fixture(tmp_path)
    with archive.open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="SHA"):
        delivery.extract_verified(archive, manifest, expected, tmp_path / "payload")


def test_archive_cannot_write_outside_delivery_folder(delivery, tmp_path):
    archive, manifest, expected = archive_fixture(tmp_path, "../outside.json")
    with pytest.raises(ValueError, match="path"):
        delivery.extract_verified(archive, manifest, expected, tmp_path / "payload")
    assert not (tmp_path / "outside.json").exists()


def test_complete_archive_is_verified_on_disk(delivery, tmp_path):
    archive, manifest, expected = archive_fixture(tmp_path)
    target = tmp_path / "payload"
    receipt = delivery.extract_verified(archive, manifest, expected, target)
    assert receipt["verified_files"] == 1
    assert receipt["archive_sha256"] == expected
    assert (target / "results/experiment/comparison.json").read_bytes() == b'{"status":"COMPLETED"}\n'


def test_unlisted_archive_member_is_rejected(delivery, tmp_path):
    archive, manifest, expected = archive_fixture(tmp_path)
    manifest["files"] = []
    with pytest.raises(ValueError):
        delivery.extract_verified(archive, manifest, expected, tmp_path / "payload")


def test_missing_required_migration_file_blocks_package(remote_delivery, tmp_path, monkeypatch):
    monkeypatch.setattr(remote_delivery, "ROOT", tmp_path)
    monkeypatch.setattr(remote_delivery, "OUT", tmp_path / "results")
    folder = tmp_path / "results/remote_migration"
    folder.mkdir(parents=True)
    (folder / "transfer_manifest.json").write_text(json.dumps({"files": [{"path": "missing.pt"}]}))
    with pytest.raises(ValueError, match="required"):
        remote_delivery.package_paths()


@pytest.mark.parametrize("change", ["changed", "added", "removed"])
def test_cached_backup_must_match_current_complete_snapshot(remote_delivery, tmp_path, monkeypatch, change):
    artifact = tmp_path / "a.json"
    artifact.write_text("original")
    manifest = {"files": [{"path": "a.json", "size": artifact.stat().st_size,
                           "sha256": remote_delivery.sha(artifact)}]}
    monkeypatch.setattr(remote_delivery, "ROOT", tmp_path)
    monkeypatch.setattr(remote_delivery, "package_paths", lambda: sorted(tmp_path.glob("*.json")))
    if change == "changed":
        artifact.write_text("changed")
    elif change == "added":
        (tmp_path / "b.json").write_text("new result")
    else:
        artifact.unlink()
    with pytest.raises(ValueError, match="snapshot"):
        remote_delivery.verify_package(manifest)


def test_receipt_is_bound_to_manifest_hash(remote_delivery, tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"files": [{"path": "result"}]}))
    index = {"archive_sha256": "archive", "manifest_sha256": remote_delivery.sha(manifest_path)}
    receipt = {"archive_sha256": "archive", "verified_files": 1,
               "all_files_verified": True, "manifest_sha256": "wrong"}
    with pytest.raises(ValueError, match="receipt"):
        remote_delivery.verify_receipt(receipt, index, manifest_path)


@pytest.mark.parametrize("fault", [None, "empty_episodes", "duplicate_episode", "missing_cycles", "empty_cycles", "wrong_cycle"])
def test_incomplete_evaluation_details_block_delivery(remote_delivery, tmp_path, monkeypatch, fault):
    import numpy as np
    monkeypatch.setattr(remote_delivery, "ROOT", tmp_path)
    data = tmp_path / "record.npz"
    np.savez(data, score_ids=np.arange(4), left=[True, False, False, True],
             right=[False, True, False, False], target=[0., 1., 2., np.nan])
    records = [{"name": "sequence", "path": "record.npz", "frames": 4}]
    predictions = tmp_path / "sequence.npz"
    np.savez(predictions, phase=np.zeros((4, 2)), unwrapped=np.zeros(4), rate=np.ones(4), anchor=np.zeros(4))
    episodes = [{"name": "sequence", "cycles": 1, "predictions_sha256": remote_delivery.sha(predictions)}]
    cycles = [{"sequence_id": "sequence", "start_frame": 0, "end_frame": 3,
               "right_touchdown_count": 1, "evaluable": True, "encoder_completed_one_turn": True}]
    summary = {k: {"observed": n, "evaluable": n, "by_sequence": {}}
               for k, n in (("all", 1), ("consistency", 1), ("mismatch", 0))}
    for key in ("all", "consistency"):
        summary[key]["by_sequence"] = {"sequence": {"observed": 1, "evaluable": 1, "completed": 1}}
    summary["point"] = [{"sequence": "sequence", "windows": 3, "consistency_windows": 3}]
    if fault == "empty_episodes":
        episodes = []
    elif fault == "duplicate_episode":
        episodes *= 2
    elif fault == "empty_cycles":
        cycles = []
    elif fault == "wrong_cycle":
        cycles[0]["end_frame"] = 2
    (tmp_path / "cohort_episodes.json").write_text(json.dumps(episodes))
    if fault != "missing_cycles":
        (tmp_path / "cohort_cycle_records.json").write_text(json.dumps(cycles))
    if fault is None:
        remote_delivery.verify_cohort(tmp_path, "cohort", summary, records)
    else:
        with pytest.raises((ValueError, FileNotFoundError)):
            remote_delivery.verify_cohort(tmp_path, "cohort", summary, records)


def test_watcher_uses_exclusive_os_lock(delivery, tmp_path):
    with delivery.watcher_lock(tmp_path / "watcher.guard"):
        with pytest.raises(RuntimeError, match="already running"):
            with delivery.watcher_lock(tmp_path / "watcher.guard"):
                pytest.fail("second watcher acquired the same OS lock")
    with delivery.watcher_lock(tmp_path / "watcher.guard"):
        pass


def test_no_shebang_shutdown_script_is_invoked_through_bash(remote_delivery, tmp_path, monkeypatch):
    script = tmp_path / "shutdown"
    script.write_text("echo accepted\n")
    calls = []
    monkeypatch.setattr(remote_delivery.subprocess, "run", lambda args, **kwargs: calls.append(args))
    assert remote_delivery.shutdown_command(script, bash="/bin/bash") == ["/bin/bash", str(script)]
    assert calls == [["/bin/bash", "-n", str(script)]]


def test_shutdown_worker_records_actual_execution_failure(remote_delivery, tmp_path, monkeypatch):
    monkeypatch.setattr(remote_delivery, "DELIVERY", tmp_path)
    (tmp_path / "shutdown_request.json").write_text(json.dumps({"attempt": 1, "command": ["invalid-shutdown-command"]}))
    monkeypatch.setattr(remote_delivery.time, "sleep", lambda _: None)
    remote_delivery.execute_shutdown()
    result = json.loads((tmp_path / "shutdown_execution.json").read_text())
    assert result["status"] == "SHUTDOWN_FAILED"
    assert result["command"] == ["invalid-shutdown-command"]
    assert result["error"]


def test_shutdown_request_is_monitored_until_actual_command_fails(delivery, tmp_path, monkeypatch):
    monkeypatch.setattr(delivery, "HERE", tmp_path)
    monkeypatch.setattr(delivery, "DESTINATION", tmp_path / "backup")
    delivery.write(tmp_path / "finalization_status.json", {"status": "SHUTDOWN_REQUESTED", "receipt": {"all_files_verified": True}})
    fake = type("Client", (), {"close": lambda self: None})()
    monkeypatch.setattr(delivery, "remote_call", lambda client, stage: {"status": "SHUTDOWN_FAILED", "attempt": 3, "returncode": 1})
    with pytest.raises(RuntimeError, match="shutdown"):
        delivery.monitor_shutdown(lambda: fake)
    state = json.loads((tmp_path / "finalization_status.json").read_text())
    assert state["status"] == "SHUTDOWN_FAILED"


def test_lost_shutdown_acknowledgement_is_recovered_idempotently(delivery, tmp_path, monkeypatch):
    monkeypatch.setattr(delivery, "HERE", tmp_path)
    monkeypatch.setattr(delivery, "DESTINATION", tmp_path / "backup")
    target = tmp_path / "finalization_status.json"
    delivery.write(target, {"status": "SHUTDOWN_REQUESTING", "receipt": {"all_files_verified": True}})
    def unavailable():
        raise ConnectionRefusedError("temporary network failure")
    assert not delivery.monitor_shutdown(unavailable)
    assert json.loads(target.read_text())["request_unacknowledged"]
    fake = type("Client", (), {"close": lambda self: None})()
    calls = []
    def call(client, stage):
        calls.append(stage)
        return {"status": "SHUTDOWN_REQUESTED", "attempt": 1}
    monkeypatch.setattr(delivery, "remote_call", call)
    assert not delivery.monitor_shutdown(lambda: fake)
    assert calls == ["shutdown"]
    assert json.loads(target.read_text())["status"] == "SHUTDOWN_REQUESTED"
