"""Train, freeze, and evaluate the supervised G1 phase encoder.

The dataset is already split by subject and normalized with train-only
statistics.  This entry point converts cycle labels to radians, fits the two
pre-registered 1D-CNN variants, selects only on validation loss, and writes a
complete checkpoint plus held-out predictions for the next gate.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch

from src.tasks.velocity.phasor.encoder import PhaseEncoder, circular_phase_loss, target_vectors
from src.tasks.velocity.phasor.data import G1_JOINT_ORDER
from scripts.evaluate_phase_study import evaluate_phase_predictions, select_phase_threshold


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "results" / "g1_phase_dataset_20261019"
DATA_PATH = DATA_DIR / "phase_dataset.npz"
MANIFEST_PATH = ROOT / "docs" / "data" / "g1_phase_data_provenance.json"
SEED = 20261019
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fit_model(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, *, preserve_time: bool, epochs: int = 8) -> tuple[PhaseEncoder, float]:
    model = PhaseEncoder(29, 10, 128, preserve_time=preserve_time).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    tx = torch.from_numpy(x_train)
    ty = target_vectors(torch.from_numpy(y_train))
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(tx, ty), batch_size=512, shuffle=True)
    for _ in range(epochs):
        model.train()
        for xb, target in loader:
            xb, target = xb.to(DEVICE), target.to(DEVICE)
            loss = circular_phase_loss(model(xb), target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        val_pred = model(torch.from_numpy(x_val).to(DEVICE))
        val_loss = float(circular_phase_loss(val_pred, target_vectors(torch.from_numpy(y_val).to(DEVICE))).cpu())
    return model, val_loss


def predict(model: PhaseEncoder, x: np.ndarray) -> np.ndarray:
    model.eval()
    out: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x), 4096):
            out.append(model(torch.from_numpy(x[start:start + 4096]).to(DEVICE)).cpu().numpy())
    return np.concatenate(out, axis=0)


def fixed_clock_phase(split: str, frame: np.ndarray, sequence: np.ndarray, arrays: dict[str, np.ndarray]) -> np.ndarray:
    """Construct the frozen train-estimated clock without test touchdown alignment."""
    train_frame = arrays["frame_train"]
    train_seq = arrays["sequence_train"]
    train_td = arrays["left_touchdown_train"].astype(bool)
    periods = []
    origins = []
    for sid in np.unique(train_seq):
        f = train_frame[train_seq == sid]
        t = f[train_td[train_seq == sid]]
        if len(t) >= 3:
            periods.extend(np.diff(t).tolist())
            origins.append(int(t[0]))
    period = float(np.median(periods)) if periods else 50.0
    origin = float(np.median(np.mod(origins, period))) if origins else 0.0
    return (2.0 * np.pi * ((frame.astype(np.float64) - origin) % period) / period).astype(np.float32)


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.set_num_threads(min(8, torch.get_num_threads()))

    d = np.load(DATA_PATH, allow_pickle=True)
    arrays = {k: d[k] for k in d.files}
    y_train = (arrays["y_train_cycles"] * (2.0 * np.pi)).astype(np.float32)
    y_val = (arrays["y_validation_cycles"] * (2.0 * np.pi)).astype(np.float32)
    y_test = (arrays["y_test_cycles"] * (2.0 * np.pi)).astype(np.float32)
    candidates = {}
    losses = {}
    for name, preserve in (("pooled", False), ("preserve_time", True)):
        model, loss = fit_model(arrays["x_train"], y_train, arrays["x_validation"], y_val, preserve_time=preserve)
        candidates[name], losses[name] = model, loss
    selected_name = min(losses, key=lambda n: (losses[n], n))
    model = candidates[selected_name]
    for p in model.parameters():
        p.requires_grad_(False)
    val_pred = predict(model, arrays["x_validation"])
    test_pred = predict(model, arrays["x_test"])
    val_report = evaluate_phase_predictions(
        val_pred, y_val,
        left_touchdown=arrays["left_touchdown_validation"].astype(bool),
        right_touchdown=arrays["right_touchdown_validation"].astype(bool),
    )
    test_clock = fixed_clock_phase("test", arrays["frame_test"], arrays["sequence_test"], arrays)
    test_report = evaluate_phase_predictions(
        test_pred, y_test,
        left_touchdown=arrays["left_touchdown_test"].astype(bool),
        right_touchdown=arrays["right_touchdown_test"].astype(bool),
        fixed_clock_phase=test_clock,
    )
    threshold = select_phase_threshold(val_pred, y_val, quantile=0.95, task_resolution_rad=np.pi / 6.0)
    test_report["phase_threshold_rad"] = threshold
    test_report["phase_pass"] = bool(test_report["circular_phase_error_rad"] <= threshold)
    metadata = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    checkpoint = {
        "state_dict": model.state_dict(), "joint_dim": 29, "window": 10,
        "preserve_time": selected_name == "preserve_time", "selected_architecture": selected_name,
        "joint_names": list(G1_JOINT_ORDER),
        "train_mean": arrays["train_mean"], "train_std": arrays["train_std"],
        # Runtime loader compatibility: observations.py consumes mean/std.
        "mean": arrays["train_mean"], "std": arrays["train_std"], "seed": SEED,
        "phase_unit": "radians", "output_convention": "(sin(phi), cos(phi))",
        "data_manifest_sha256": sha256_file(MANIFEST_PATH), "dataset_sha256": sha256_file(DATA_PATH),
        "train_sequences": sorted(np.unique(arrays["sequence_train"]).tolist()),
        "validation_sequences": sorted(np.unique(arrays["sequence_validation"]).tolist()),
        "test_sequences": sorted(np.unique(arrays["sequence_test"]).tolist()),
    }
    torch.save(checkpoint, DATA_DIR / "phase_encoder_checkpoint.pt")
    np.savez_compressed(DATA_DIR / "phase_predictions.npz", val_pred=val_pred, test_pred=test_pred, y_val=y_val, y_test=y_test, test_clock=test_clock)
    report = {
        "seed": SEED, "device": str(DEVICE), "dataset_sha256": sha256_file(DATA_PATH),
        "manifest_sha256": sha256_file(MANIFEST_PATH), "selected_architecture": selected_name,
        "validation_losses": losses, "validation_report": val_report,
        "test_report": test_report, "phase_gate_threshold_rad": threshold,
        "data_status": metadata.get("status"), "phase_unit": "radians",
        "control_gate_status": "NOT_RUN_UNTIL_PHASE_AND_ROLLOUT_GATES_PASS",
    }
    (DATA_DIR / "phase_encoder_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
