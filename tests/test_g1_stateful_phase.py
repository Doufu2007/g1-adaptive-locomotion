"""Scientific boundary tests for the separate continuous-phase experiment."""
import importlib
from pathlib import Path

import numpy as np
import pytest
import torch


@pytest.fixture
def study():
    path = Path(__file__).resolve().parents[1] / "scripts/g1_stateful_phase.py"
    assert path.exists(), "continuous-phase implementation is missing"
    return importlib.import_module("scripts.g1_stateful_phase")


def test_features_cannot_read_future(study):
    q = np.arange(40 * 29, dtype=float).reshape(40, 29) / 1000
    quat = np.tile([1., 0., 0., 0.], (40, 1))
    a = study.causal_features(q, quat)
    q[21:] += 20
    quat[21:] = [0., 1., 0., 0.]
    b = study.causal_features(q, quat)
    np.testing.assert_array_equal(a[:21], b[:21])
    np.testing.assert_allclose(a[0, 29:58], 0)
    np.testing.assert_allclose(a[:, 58:61], np.tile([0, 0, -1], (40, 1)))


def test_quaternion_sign_does_not_change_imu_features(study):
    angle = np.arange(12) * .03
    xyzw = np.column_stack((np.zeros(12), np.zeros(12), np.sin(angle / 2), np.cos(angle / 2)))
    wxyz = xyzw[:, [3, 0, 1, 2]]
    signed = wxyz.copy(); signed[::2] *= -1
    a = study.causal_features(np.zeros((12, 29)), wxyz)
    b = study.causal_features(np.zeros((12, 29)), signed)
    np.testing.assert_allclose(a, b, atol=1e-6)
    np.testing.assert_allclose(a[1:, -1], 1.5, atol=1e-6)


def test_contact_debounce_emits_at_confirmation_not_in_past(study):
    raw = np.array([0, 1, 1, 1, 0, 1, 0, 0, 0], bool)
    result = study.debounce_contacts(raw[:, None])[:, 0]
    np.testing.assert_array_equal(result, [0, 0, 0, 1, 1, 1, 1, 1, 0])
    changed = raw.copy(); changed[5:] = 0
    np.testing.assert_array_equal(study.debounce_contacts(changed[:, None])[:5, 0], result[:5])


def test_contact_phase_labels_own_half_open_complete_cycles(study):
    contact = np.zeros((25, 2), bool)
    contact[[2, 12, 22], 0] = True
    contact[[7, 17], 1] = True
    phase, left, right, cycles = study.contact_labels(contact, debounce=1)
    assert [(c[0], c[1], c[2]) for c in cycles] == [(2, 12, 1), (12, 22, 1)]
    assert phase[2] == 0 and phase[12] == 0
    assert phase[11] == pytest.approx(1.8 * np.pi)
    assert np.isnan(phase[22:]).all()


@pytest.mark.parametrize("mode", ["direct", "observer"])
def test_network_is_causal_and_streaming_matches_full_sequence(study, mode):
    torch.set_num_threads(2); torch.manual_seed(7)
    model = study.PhaseNet(29, mode, width=8).eval()
    x = torch.randn(2, 180, 29)
    with torch.no_grad():
        full = model(x)
        first = model(x[:, :73])
        second = model(x[:, 73:], first["state"])
        changed = x.clone(); changed[:, 90:] += 5
        altered = model(changed)
    torch.testing.assert_close(full["phase"][:, :90], altered["phase"][:, :90], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(torch.cat([first["phase"], second["phase"]], 1), full["phase"], atol=2e-5, rtol=2e-5)
    if mode == "observer":
        delta = full["unwrapped"][:, 1:] - full["unwrapped"][:, :-1]
        assert delta.min() >= -1e-6 and delta.max() <= np.pi / 3 + 1e-5


def test_complete_cycle_loss_backpropagates_without_true_phase_initialization(study):
    torch.manual_seed(3)
    model = study.PhaseNet(29, "observer", width=8)
    output = model(torch.randn(2, 100, 29))
    target = torch.zeros(2, 100)
    target[:, 20:80] = torch.arange(60) * (2 * np.pi / 60)
    loss, parts = study.cycle_objective(output, target, torch.tensor([20, 20]), torch.tensor([80, 80]))
    loss.backward()
    assert torch.isfinite(loss) and set(parts) == {"anchor", "point", "increment", "rate", "cycle"}
    assert model.head.weight.grad.abs().sum() > 0


def test_extra_normalization_rejects_nontraining_sequences(study):
    a = {"split": "train", "features": np.ones((20, 64), np.float32)}
    b = {"split": "validation", "features": np.ones((20, 64), np.float32) * 999}
    with pytest.raises(ValueError, match="train"):
        study.fit_normalizer([a, b], np.zeros(29), np.ones(29))
    mean, std = study.fit_normalizer([a], np.zeros(29), np.ones(29))
    np.testing.assert_array_equal(mean[:29], 0)
    np.testing.assert_array_equal(mean[29:], 1)
    assert np.all(std > 0)


def test_fixed_matrix_and_incomplete_dependency_block_gpu(study, tmp_path):
    assert len(study.matrix()) == 15
    assert {j["seed"] for j in study.matrix()} == {20261025, 20261026, 20261027}
    assert study.UPDATES == 2000 and study.BATCH_CYCLES == 8
    assert not study.previous_ready(tmp_path)
    (tmp_path / "queue_status.json").write_text('{"status":"COMPLETED"}')
    assert not study.previous_ready(tmp_path), "status alone cannot prove matrix completion"


def test_sampler_excludes_mismatch_cycles_and_preserves_paired_order(study):
    record = {"name": "motion", "domain": "motion", "split": "train",
              "features": np.zeros((180, 64), np.float32),
              "cycles": [(20, 60, 1), (60, 100, 2), (100, 140, 1)]}
    first = study.CycleSampler([record], seed=17, mixed=False)
    second = study.CycleSampler([record], seed=17, mixed=False)
    aa, bb = first.sample(), second.sample()
    assert [(r[0]["name"], r[1]) for r in aa] == [(r[0]["name"], r[1]) for r in bb]
    assert len(aa) == 8 and all(c[2] == 1 for _, c in aa)


@pytest.fixture
def runner():
    path = Path(__file__).resolve().parents[1] / "scripts/run_g1_stateful_phase_20261025.py"
    assert path.exists(), "continuous-phase runner is missing"
    return importlib.import_module("scripts.run_g1_stateful_phase_20261025")


def test_simulation_splits_have_disjoint_seeds_and_expected_size(runner):
    jobs = {s: runner.simulation_jobs(s) for s in ("train", "development", "test")}
    assert [sum(len(j["seeds"]) * 3 for j in jobs[s]) for s in jobs] == [48, 24, 24]
    seeds = {s: {v for j in jobs[s] for v in j["seeds"]} for s in jobs}
    assert not seeds["train"] & seeds["test"] and not seeds["development"] & seeds["test"]
    assert {j["terrain_seed"] for j in jobs["test"]} == {20261800}


def test_test_collection_is_locked_before_all_final_models(runner, tmp_path):
    with pytest.raises(RuntimeError, match="frozen"):
        runner.require_test_unlocked(tmp_path)


def test_simulation_summary_retains_no_cycle_and_fallen_episodes(runner):
    rows = [dict(stratum="rough_clock_push", completion=None, point_mae_rad=None,
                 cycles=0, fell=True),
            dict(stratum="rough_clock_push", completion=1., point_mae_rad=.1,
                 cycles=10, fell=False)]
    result = runner.summarize_simulation(rows)
    assert result["episodes"] == 2 and result["no_cycle_episodes"] == 1
    assert result["falls"] == 1
    assert result["macro_completion_with_no_cycle_zero"] == .5
    assert result["conditional_macro_completion"] == 1.


def test_process_identity_recognizes_module_invocation(runner, monkeypatch):
    import psutil
    class Process:
        def __init__(self, pid):
            pass
        def cmdline(self):
            return ["python", "-m", "scripts.run_g1_stateful_phase_20261025", "--stage", "queue"]
    monkeypatch.setattr(psutil, "Process", Process)
    assert runner.matching_process(123, "run_g1_stateful_phase_20261025.py")
    assert not runner.matching_process(123, "other_project_training.py")


def test_final_model_lock_rejects_duplicate_matrix_rows(runner, tmp_path):
    import json
    model = tmp_path / "model.pt"; model.write_bytes(b"fixture")
    entry = dict(row="direct_motion", seed=20261025, path=str(model), sha256=runner.core.sha(model))
    (tmp_path / "final_models_frozen.json").write_text(json.dumps(dict(models=[entry] * 15)))
    with pytest.raises(RuntimeError, match="matrix"):
        runner.require_test_unlocked(tmp_path)


def test_opening_stance_is_not_a_touchdown(study):
    contacts = np.ones((30, 2), bool)
    contacts[8:14, 0] = False
    phase, left, right, cycles = study.contact_labels(contacts)
    assert np.flatnonzero(left).tolist() == [16]
    assert not right.any() and not cycles


def test_shared_initial_parameters_are_paired_across_inputs(runner):
    a = runner.initial_model(29, "direct", 7)
    b = runner.initial_model(29, "observer", 7)
    c = runner.initial_model(64, "observer", 7)
    for name, value in a.state_dict().items():
        torch.testing.assert_close(value, b.state_dict()[name], rtol=0, atol=0)
        other = c.state_dict()[name]
        if other.shape != value.shape:
            other = other[:, :29]
        torch.testing.assert_close(value, other, rtol=0, atol=0)


def test_resume_command_points_to_an_executable_module(runner):
    import subprocess
    assert hasattr(runner, "training_resume_command"), "stable resume command is missing"
    command = runner.training_resume_command("observer_motion", 20261025)
    assert "__main__" not in command
    result = subprocess.run(command + ["--help"], cwd=runner.ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_same_count_frame_replacement_is_detected_by_seal(runner, monkeypatch, tmp_path):
    import json
    archive = tmp_path / "filtered.npz"
    np.savez(archive, frame=np.arange(10))
    output = tmp_path / "study"; output.mkdir()
    seal = dict(assets={"filtered.npz": runner.core.sha(archive)})
    (output / "asset_seal.json").write_text(json.dumps(seal))
    monkeypatch.setattr(runner, "ROOT", tmp_path); monkeypatch.setattr(runner, "OUT", output)
    runner.verify_seal()
    np.savez(archive, frame=np.arange(10) + 1)
    with pytest.raises(ValueError, match="sealed asset changed"):
        runner.verify_seal()
