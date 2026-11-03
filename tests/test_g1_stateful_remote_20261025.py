from pathlib import Path


def test_windows_paths_resolve_inside_remote_root():
    from scripts.run_g1_stateful_remote_20261025 import PortableRoot, SOURCE_ROOT
    root = PortableRoot(Path(__file__).resolve().parents[1])
    assert root / r"results\g1_stateful_phase_20261025\asset_seal.json" == root / "results/g1_stateful_phase_20261025/asset_seal.json"
    assert root / (SOURCE_ROOT + r"\scripts\g1_stateful_phase.py") == root / "scripts/g1_stateful_phase.py"


def test_remote_install_keeps_original_seals_and_matrix(monkeypatch):
    import json
    import numpy as np
    import torch
    from scripts import run_g1_stateful_phase_20261025 as runner
    from scripts import run_g1_stateful_remote_20261025 as remote
    root = Path(__file__).resolve().parents[1]
    seals = [root / "results" / n / "asset_seal.json" for n in
             ("g1_stateful_phase_20261025", "g1_phase_controller_adaptation_20261024")]
    before = [p.read_bytes() for p in seals]
    from scripts import g1_phase_adaptation_20261024 as adaptation
    from scripts import run_g1_best_phase_control as base
    from scripts import g1_best_phase_batched as batch
    for module in (runner, adaptation, base, batch):
        monkeypatch.setattr(module, "ROOT", module.ROOT)
    monkeypatch.setattr(runner.core, "make_batch", runner.core.make_batch)
    for module, names in ((runner, ("matching_process", "training_resume_command", "child")),
                          (base, ("bootstrap",)), (adaptation, ("bootstrap",)), (batch, ("bootstrap",))):
        for name in names:
            monkeypatch.setattr(module, name, getattr(module, name))
    remote.install()
    assert len(runner.verify_seal()["assets"]) == 11
    assert len(adaptation.verify_assets()["assets"]) == 35
    assert [p.read_bytes() for p in seals] == before
    assert len(runner.core.matrix()) == 15
    assert runner.core.UPDATES == 2000
    assert runner.core.previous_ready(runner.PREVIOUS)
    assert "scripts.run_g1_stateful_remote_20261025" in runner.training_resume_command("observer_motion", 20261025)
    # Real saved-cycle indices are NumPy integers; the first training update
    # must be serializable without changing any training tensors or frame count.
    record = dict(features=np.zeros((12, 29), np.float32), target=np.arange(12, dtype=np.float32))
    x, target, starts, ends, frames = runner.core.make_batch(
        [(record, np.array([2, 8, 1], dtype=np.int64))],
        np.zeros(29, np.float32), np.ones(29, np.float32), 29, "cpu")
    telemetry = json.loads(json.dumps(dict(update=1, effective_frames=frames)))
    assert telemetry == dict(update=1, effective_frames=6)
    torch.testing.assert_close(x, torch.zeros(1, 8, 29))
    torch.testing.assert_close(target, torch.arange(8, dtype=torch.float32)[None])
    assert starts.tolist() == [2] and ends.tolist() == [8]


def test_remote_process_identity(monkeypatch):
    import psutil
    from scripts.run_g1_stateful_remote_20261025 import matching_process
    class Process:
        def __init__(self, pid):
            pass
        def cmdline(self):
            return ["python", "-m", "scripts.run_g1_stateful_remote_20261025", "--stage", "queue"]
    monkeypatch.setattr(psutil, "Process", Process)
    assert matching_process(123, "run_g1_stateful_phase_20261025.py")
    assert not matching_process(123, "run_g1_phase_adaptation_20261024")
