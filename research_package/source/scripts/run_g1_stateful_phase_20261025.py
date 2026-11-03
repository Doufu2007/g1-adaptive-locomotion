"""Separate, serial, fixed-budget continuous-phase experiment; never edits old assets."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from types import SimpleNamespace

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scripts import g1_stateful_phase as core

OUT = ROOT / "results/g1_stateful_phase_20261025"
PREVIOUS = ROOT / "results/g1_phase_controller_adaptation_20261024"
DATA = ROOT / "results/g1_phase_dataset_20261019"
SPEC = ROOT / "docs/superpowers/specs/2026-10-25-g1-stateful-phase-design.md"
SCENARIOS = ("steady", "speed_change", "push")


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def seed_all(seed, device):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(2)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def environment():
    import importlib.metadata
    return dict(python=sys.version, platform=platform.platform(),
                packages={p: importlib.metadata.version(p) for p in ("numpy", "scipy", "torch", "mjlab")},
                threads=torch.get_num_threads(), pythonhashseed=os.environ.get("PYTHONHASHSEED"),
                precision="float32, no AMP, no TF32")


def freeze():
    paths = [Path(__file__), ROOT / "scripts/g1_stateful_phase.py", SPEC,
             ROOT / "docs/superpowers/plans/2026-10-25-g1-stateful-phase.md",
             ROOT / "tests/test_g1_stateful_phase.py"]
    paths += [ROOT / p for p in ("scripts/g1_best_phase_batched.py", "scripts/g1_phase_adaptation_20261024.py",
                                "src/tasks/velocity/phasor/phase_gate_v4_2.py")]
    paths += [DATA / "phase_dataset.npz", DATA / "dataset_metadata.json",
              DATA / "contact_consistency_filter_v1/phase_dataset_train_validation.npz"]
    seal = dict(schema="g1.stateful_phase.seal.v1", frozen_at=now(),
                authorization="2026-10-25 user: 按你说的做", matrix=core.matrix(),
                updates=core.UPDATES, batch_cycles=core.BATCH_CYCLES,
                assets={str(p.relative_to(ROOT)): core.sha(p) for p in paths})
    seal["git_commit"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    seal["git_dirty_state"] = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    path = OUT / "asset_seal.json"
    if path.exists():
        verify_seal()
    else:
        import shutil
        for source in paths[:5]:
            snapshot = OUT / "source_snapshot" / source.relative_to(ROOT)
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, snapshot)
        write(path, seal)


def verify_seal():
    seal = read(OUT / "asset_seal.json")
    for name, expected in seal["assets"].items():
        if core.sha(ROOT / name) != expected:
            raise ValueError(f"new experiment sealed asset changed: {name}")
    return seal


def simulation_jobs(split):
    seeds = {"train": list(range(20261601, 20261605)),
             "development": [20261701, 20261702], "test": [20261801, 20261802]}[split]
    terrain_seed = {"train": 20261600, "development": 20261700, "test": 20261800}[split]
    return [dict(split=split, terrain=terrain, arm=arm, seeds=seeds, terrain_seed=terrain_seed)
            for terrain in ("flat", "rough") for arm in ("clock", "learned")]


def require_test_unlocked(directory=OUT):
    path = Path(directory) / "final_models_frozen.json"
    if not path.exists():
        raise RuntimeError("test is locked until all 15 final models are frozen")
    seal = read(path)
    if len(seal["models"]) != 15:
        raise RuntimeError("all 15 final models must be frozen")
    identities = {(r["row"], r["seed"]) for r in seal["models"]}
    if identities != {(j["row"], j["seed"]) for j in core.matrix()}:
        raise RuntimeError("frozen final model matrix is incomplete or duplicated")
    for record in seal["models"]:
        if core.sha(ROOT / record["path"]) != record["sha256"]:
            raise RuntimeError("frozen final model changed")
    return seal


def summarize_simulation(rows):
    strata = sorted({r["stratum"] for r in rows})
    by = {}
    for name in strata:
        rr = [r for r in rows if r["stratum"] == name]
        valid = [r["completion"] for r in rr if r["completion"] is not None]
        point = [r["point_mae_rad"] for r in rr if r["point_mae_rad"] is not None]
        by[name] = dict(episodes=len(rr), cycles=sum(r["cycles"] for r in rr),
                        no_cycle_episodes=sum(r["cycles"] == 0 for r in rr), falls=sum(r["fell"] for r in rr),
                        completion_with_no_cycle_zero=float(np.mean([r["completion"] or 0. for r in rr])),
                        conditional_completion=float(np.mean(valid)) if valid else None,
                        point_mae_rad=float(np.mean(point)) if point else None,
                        coverage_pass=sum(r["cycles"] for r in rr) >= 10)
    valid = [r["conditional_completion"] for r in by.values() if r["conditional_completion"] is not None]
    point = [r["point_mae_rad"] for r in by.values() if r["point_mae_rad"] is not None]
    return dict(episodes=len(rows), falls=sum(r["fell"] for r in rows),
                no_cycle_episodes=sum(r["cycles"] == 0 for r in rows),
                macro_completion_with_no_cycle_zero=float(np.mean([r["completion_with_no_cycle_zero"] for r in by.values()])),
                conditional_macro_completion=float(np.mean(valid)) if valid else None,
                macro_point_mae_rad=float(np.mean(point)) if point else None, by_stratum=by)


def save_record(record, path):
    keys = ("features", "target", "left", "right", "score_ids")
    np.savez_compressed(path, **{k: record[k] for k in keys}, cycles=np.asarray(record["cycles"], np.int64).reshape(-1, 3))
    return dict(**{k: v for k, v in record.items() if k not in keys and k != "cycles"},
                path=str(path.relative_to(ROOT)), sha256=core.sha(path),
                frames=len(record["features"]), cycles=len(record["cycles"]),
                consistent_cycles=sum(c[2] == 1 for c in record["cycles"]))


def load_records(index_path, split=None):
    records = []
    for entry in read(index_path)["records"]:
        if split is not None and entry["split"] != split:
            continue
        path = ROOT / entry["path"]
        if core.sha(path) != entry["sha256"]:
            raise ValueError("derived data changed")
        with np.load(path, allow_pickle=False) as saved:
            record = {k: saved[k] for k in saved.files}
        record.update({k: v for k, v in entry.items() if k not in ("cycles", "frames")})
        records.append(record)
    return records


def prepare_motion():
    """Only named train/validation arrays; original test arrays are never loaded."""
    from scipy.spatial.transform import Rotation, Slerp
    target = OUT / "data/motion_index.json"
    if target.exists():
        load_records(target)
        return
    folder = target.parent / "motion"; folder.mkdir(parents=True, exist_ok=True)
    metadata = read(DATA / "dataset_metadata.json")
    provenance = {r["sequence_id"]: r for r in metadata["sequences"]}
    entries = []
    with np.load(DATA / "phase_dataset.npz", allow_pickle=False) as source:
        mean, std = source["train_mean"], source["train_std"]
        for split in ("train", "validation"):
            sequences, frames = source["sequence_" + split], source["frame_" + split]
            windows = source["x_" + split]
            for name in sorted(set(sequences.tolist())):
                csv = ROOT / "data/raw/lafan1_g1_walk" / (name + ".csv")
                if core.sha(csv) != provenance[name]["sha256"]:
                    raise ValueError("fixed raw motion source hash changed")
                raw = np.loadtxt(csv, delimiter=",", dtype=np.float64)
                source_t = np.arange(len(raw), dtype=np.float64) / 30.
                count = int(np.floor(source_t[-1] * 50 + 1e-9)) + 1
                t = np.arange(count, dtype=np.float64) / 50.
                # Reproduce the original float32 joint interpolation exactly.
                source_q = raw[:, 7:].astype(np.float32)
                q = np.stack([np.interp(t, source_t, source_q[:, c]) for c in range(29)], axis=1).astype(np.float32)
                quat_xyzw = Slerp(source_t, Rotation.from_quat(raw[:, 3:7]))(t).as_quat()
                ids = np.flatnonzero(sequences == name); end = frames[ids]
                expected = (q[end] - mean) / std
                np.testing.assert_allclose(expected, windows[ids, -1], atol=2e-6, rtol=2e-6)
                reference = np.full(count, np.nan, np.float32)
                reference[end] = source["y_" + split + "_cycles"][ids] * (2 * np.pi)
                left, right = np.zeros(count, bool), np.zeros(count, bool)
                left[end] = source["left_touchdown_" + split][ids].astype(bool)
                right[end] = source["right_touchdown_" + split][ids].astype(bool)
                events = np.flatnonzero(left)
                cycles = [(int(a), int(b), int(right[a:b].sum())) for a, b in zip(events[:-1], events[1:])
                          if b - a >= 3 and np.isfinite(reference[a:b]).all()]
                record = dict(name=name, domain="motion", split=split,
                              features=core.causal_features(q, quat_xyzw[:, [3, 0, 1, 2]]),
                              target=reference, left=left, right=right, cycles=cycles, score_ids=end,
                              source_csv=str(csv.relative_to(ROOT)), source_sha256=core.sha(csv))
                entries.append(save_record(record, folder / (name + ".npz")))
    write(target, dict(schema="g1.stateful_phase.motion.v1", created_at=now(), records=entries,
                       source_archive_sha256=core.sha(DATA / "phase_dataset.npz"),
                       joint_mean=mean.tolist(), joint_std=std.tolist(),
                       original_test_arrays_read=False, raw_joint_endpoint_parity_passed=True))
    print(json.dumps(dict(stage="prepare_motion", records=len(entries), endpoint_parity=True)), flush=True)


def collect(split, terrain, arm, smoke=False):
    if not core.previous_ready(PREVIOUS):
        raise RuntimeError("previous 12+18 experiment must finish before GPU collection")
    from scripts import g1_phase_adaptation_20261024 as adaptation
    adaptation.verify_assets()
    verify_seal()
    if split == "test":
        require_test_unlocked()
    job = next(j for j in simulation_jobs(split) if j["terrain"] == terrain and j["arm"] == arm)
    policy = adaptation.train_dir(terrain, arm, 20261401) / "policy.onnx"
    training = read(policy.parent / "manifest.json")
    if training["status"] != "COMPLETED" or core.sha(policy) != training["policy_sha256"]:
        raise ValueError("fixed collection source actor missing or changed")
    target = OUT / ("smoke/collection" if smoke else "collection") / split / f"{terrain}_{arm}"
    if (target / "manifest.json").exists():
        record = read(target / "manifest.json")
        if record["status"] == "COMPLETED" and record.get("stateful_collection_protocol") == core.sha(SPEC):
            return
        raise FileExistsError("preserve incomplete collection; archive it explicitly before technical retry")
    from scripts import g1_best_phase_batched as batch
    batch.ENCODER = adaptation.RUNTIME
    adaptation.base.ENCODER = adaptation.RUNTIME
    def configuration(t, a, seed, **kwargs):
        cfg = adaptation.make_cfg(t, a, seed, **kwargs)
        if t == "rough":
            cfg.scene.terrain.terrain_generator.seed = job["terrain_seed"]
        return cfg
    batch.make_cfg = configuration
    batch.rollout_batched(SimpleNamespace(output=str(target), policy=str(policy), terrain=terrain, arm=arm,
        label="stateful_data_" + split, seeds=job["seeds"][:1] if smoke else job["seeds"],
        horizon=.2 if smoke else 20., scenarios=["steady"] if smoke else list(SCENARIOS)))
    manifest = read(target / "manifest.json")
    manifest.update(stateful_collection_protocol=core.sha(SPEC), dataset_split=split,
                    terrain_generator_seed=job["terrain_seed"], source_training_seed=20261401)
    write(target / "manifest.json", manifest)


def prepare_simulation(split):
    if split == "test":
        require_test_unlocked()
    target = OUT / "data" / ("simulation_" + split + "_index.json")
    if target.exists():
        load_records(target)
        return
    folder = target.parent / ("simulation_" + split); folder.mkdir(parents=True, exist_ok=True)
    entries = []
    for job in simulation_jobs(split):
        directory = OUT / "collection" / split / f"{job['terrain']}_{job['arm']}"
        manifest = read(directory / "manifest.json")
        if manifest["status"] != "COMPLETED" or manifest.get("stateful_collection_protocol") != core.sha(SPEC):
            raise RuntimeError("simulation collection incomplete or protocol mismatch")
        rows = read(directory / "episodes.json")
        expected = {(s, c) for s in job["seeds"] for c in SCENARIOS}
        if {(r["seed"], r["scenario"]) for r in rows} != expected or len(rows) != len(expected):
            raise ValueError("collected episode matrix mismatch")
        for row in rows:
            raw_path = directory / row["raw_path"]
            if core.sha(raw_path) != row["raw_sha256"]:
                raise ValueError("collection raw trajectory changed")
            with np.load(raw_path, allow_pickle=False) as z:
                raw = {k: z[k] for k in z.files}
            if len(raw["time"]) > 1 and not np.allclose(np.diff(raw["time"]), core.DT):
                raise ValueError("simulation frame gap; cannot span an episode boundary")
            phase, left, right, cycles = core.contact_labels(raw["contact"], debounce=3)
            name = f"sim_{job['terrain']}_{job['arm']}_{row['scenario']}_{row['seed']}"
            record = dict(name=name, domain="simulation", split="train" if split == "train" else split,
                features=core.causal_features(raw["joint_pos"], raw["quaternion"]), target=phase,
                left=left, right=right, cycles=cycles, score_ids=np.arange(len(phase)),
                stratum=f"{job['terrain']}_{job['arm']}_{row['scenario']}", fell=row["fell"],
                scenario=row["scenario"], source_raw=str(raw_path.relative_to(ROOT)),
                source_sha256=core.sha(raw_path), environment_seed=row["seed"])
            entries.append(save_record(record, folder / (name + ".npz")))
    write(target, dict(schema="g1.stateful_phase.simulation.v1", split=split, records=entries,
                       created_at=now(), teacher="feet_ground_contact.current_contact_time>0; causal 3-frame debounce",
                       no_cycle_episodes=sum(r["cycles"] == 0 for r in entries)))


def prepare_training_data():
    prepare_motion()
    prepare_simulation("train"); prepare_simulation("development")
    records = load_records(OUT / "data/motion_index.json", "train")
    records += load_records(OUT / "data/simulation_train_index.json", "train")
    original = read(OUT / "data/motion_index.json")
    mean, std = core.fit_normalizer(records, original["joint_mean"], original["joint_std"])
    path = OUT / "data/normalizer.json"
    normalizer = dict(mean=mean.tolist(), std=std.tolist(), fitted_splits=["motion/train", "simulation/train"],
                      input_indices={str(p.relative_to(ROOT)): core.sha(p) for p in [
                          OUT / "data/motion_index.json", OUT / "data/simulation_train_index.json"]})
    if path.exists() and read(path) != normalizer:
        raise ValueError("frozen normalizer would change")
    write(path, normalizer)
    data_seal = OUT / "data_seal.json"
    inputs = [OUT / "data" / n for n in ("motion_index.json", "simulation_train_index.json",
                                          "simulation_development_index.json", "normalizer.json")]
    seal = dict(frozen_at=now(), inputs={str(p.relative_to(ROOT)): core.sha(p) for p in inputs})
    if data_seal.exists():
        verify_data()
    else:
        write(data_seal, seal)


def verify_data():
    record = read(OUT / "data_seal.json")
    for path, expected in record["inputs"].items():
        if core.sha(ROOT / path) != expected:
            raise ValueError("frozen training data index/normalizer changed")
    return record


def initial_model(channels, mode, seed):
    # Every shared parameter has identical initial values across paired rows.
    torch.manual_seed(seed)
    template = core.PhaseNet(64, mode)
    model = core.PhaseNet(channels, mode)
    source = template.state_dict()
    state = {}
    for name, value in model.state_dict().items():
        other = source[name]
        state[name] = other[:, :channels].clone() if other.shape != value.shape else other.clone()
    model.load_state_dict(state, strict=True)
    return model


def atomic_checkpoint(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary); temporary.replace(path)


def training_resume_command(row, seed):
    return [sys.executable, "-m", "scripts.run_g1_stateful_phase_20261025",
            "--stage", "train", "--row", row, "--seed", str(seed)]


def train(row, seed):
    if (row, seed) not in {(j["row"], j["seed"]) for j in core.matrix()}:
        raise ValueError("run outside frozen matrix")
    if not core.previous_ready(PREVIOUS):
        raise RuntimeError("previous study incomplete; no GPU training")
    verify_seal(); verify_data()
    mode, channels, mixed = core.ROWS[row]
    directory = OUT / "training" / f"{row}_seed{seed}"
    manifest_path = directory / "manifest.json"
    previous = read(manifest_path) if manifest_path.exists() else None
    if previous and previous["status"] == "COMPLETED":
        if (previous["completed_updates"] != core.UPDATES
                or core.sha(directory / "final.pt") != previous["checkpoint_sha256"]):
            raise ValueError("completed training artifact changed")
        return
    if previous:
        import psutil
        pid = previous.get("pid")
        if pid and pid != os.getpid() and psutil.pid_exists(pid):
            try:
                if matching_process(pid, Path(__file__).name):
                    raise RuntimeError("matching training process already active")
            except psutil.NoSuchProcess:
                pass
        write(directory / "attempts" / f"manifest_{time.time_ns()}.json", previous)
    directory.mkdir(parents=True, exist_ok=True)
    device = "cuda"
    seed_all(seed, device)
    model = initial_model(channels, mode, seed).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.0001)
    records = load_records(OUT / "data/motion_index.json", "train")
    records += load_records(OUT / "data/simulation_train_index.json", "train")
    normalizer = read(OUT / "data/normalizer.json")
    mean, std = np.asarray(normalizer["mean"], np.float32), np.asarray(normalizer["std"], np.float32)
    sampler = core.CycleSampler(records, seed=seed + 1000, mixed=mixed)
    completed, frames, cycles = 0, 0, 0
    rejected = []
    for path in sorted(directory.glob("checkpoint_*.pt"), reverse=True):
        try:
            checkpoint = torch.load(path, map_location=device, weights_only=False)
            if (checkpoint["row"], checkpoint["seed"], checkpoint["asset_seal_sha256"], checkpoint["data_seal_sha256"]) != (
                    row, seed, core.sha(OUT / "asset_seal.json"), core.sha(OUT / "data_seal.json")):
                raise ValueError("checkpoint identity mismatch")
            if checkpoint["completed_updates"] > core.UPDATES:
                raise ValueError("checkpoint exceeds budget")
        except Exception as error:
            rejected.append(dict(path=str(path), error=repr(error)))
            continue
        model.load_state_dict(checkpoint["model"]); optimizer.load_state_dict(checkpoint["optimizer"])
        sampler.rng.bit_generator.state = checkpoint["sampler_rng"]
        random.setstate(checkpoint["python_rng"]); np.random.set_state(checkpoint["numpy_rng"])
        torch.set_rng_state(checkpoint["torch_rng"].cpu())
        torch.cuda.set_rng_state_all([r.cpu() for r in checkpoint["cuda_rng"]])
        completed, frames, cycles = checkpoint["completed_updates"], checkpoint["effective_frames"], checkpoint["effective_cycles"]
        break
    if previous and previous.get("completed_updates", 0) and completed == 0:
        raise RuntimeError("failed training has no intact checkpoint; no automatic restart")
    manifest = dict(schema="g1.stateful_phase.run.v1", status="RUNNING", started_at=now(), pid=os.getpid(),
        row=row, seed=seed, device=device, environment=environment(), argv=sys.argv,
        planned_updates=core.UPDATES, batch_cycles=core.BATCH_CYCLES, completed_updates=completed,
        resumed=bool(previous), recovered_updates=completed, rejected_checkpoints=rejected,
        asset_seal_sha256=core.sha(OUT / "asset_seal.json"), data_seal_sha256=core.sha(OUT / "data_seal.json"),
        discarded_update_overhead=max(0, (previous or {}).get("completed_updates", 0) - completed),
        previous_attempt_manifest_archived=bool(previous))
    write(manifest_path, manifest)
    model.train()
    started = time.monotonic()
    try:
        for update in range(completed, core.UPDATES):
            batch = core.make_batch(sampler.sample(), mean, std, channels, device)
            x, reference, starts, ends, count = batch
            optimizer.zero_grad(set_to_none=True)
            loss, parts = core.cycle_objective(model(x), reference, starts, ends)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite training loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            completed = update + 1; frames += count; cycles += core.BATCH_CYCLES
            telemetry = dict(update=completed, loss=float(loss.detach()),
                parts={k: float(v.detach()) for k, v in parts.items()}, gradient_norm=float(norm),
                effective_frames=frames, effective_cycles=cycles, elapsed_seconds=time.monotonic() - started)
            with (directory / "losses.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(telemetry) + "\n")
            if completed % 25 == 0:
                print(json.dumps(dict(row=row, seed=seed, **telemetry)), flush=True)
            if completed % 250 == 0 or completed == core.UPDATES:
                payload = dict(row=row, seed=seed, completed_updates=completed, effective_frames=frames,
                    effective_cycles=cycles, model=model.state_dict(), optimizer=optimizer.state_dict(),
                    sampler_rng=sampler.rng.bit_generator.state, python_rng=random.getstate(),
                    numpy_rng=np.random.get_state(), torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state_all(),
                    asset_seal_sha256=manifest["asset_seal_sha256"], data_seal_sha256=manifest["data_seal_sha256"])
                path = directory / f"checkpoint_{completed:05d}.pt"
                atomic_checkpoint(path, payload)
                manifest.update(completed_updates=completed, effective_frames=frames, effective_cycles=cycles,
                                latest_checkpoint=str(path), latest_checkpoint_sha256=core.sha(path), updated_at=now())
                write(manifest_path, manifest)
        final = directory / "final.pt"
        atomic_checkpoint(final, dict(model=model.state_dict(), channels=channels, mode=mode, row=row, seed=seed,
                                      completed_updates=completed, mean=mean, std=std,
                                      asset_seal_sha256=manifest["asset_seal_sha256"],
                                      data_seal_sha256=manifest["data_seal_sha256"]))
        manifest.update(status="COMPLETED", ended_at=now(), completed_updates=completed,
                        effective_frames=frames, effective_cycles=cycles, checkpoint_sha256=core.sha(final))
    except BaseException as error:
        manifest.update(status="FAILED", ended_at=now(), completed_updates=completed, error=repr(error),
                        traceback=traceback.format_exc(), resume_command=training_resume_command(row, seed))
        raise
    finally:
        write(manifest_path, manifest)


def freeze_models():
    records = []
    for job in core.matrix():
        path = OUT / "training" / f"{job['row']}_seed{job['seed']}"
        manifest = read(path / "manifest.json")
        if manifest["status"] != "COMPLETED" or manifest["completed_updates"] != core.UPDATES:
            raise RuntimeError("cannot freeze incomplete training matrix")
        if core.sha(path / "final.pt") != manifest["checkpoint_sha256"]:
            raise ValueError("final model hash mismatch")
        records.append(dict(**job, path=str((path / "final.pt").relative_to(ROOT)), sha256=manifest["checkpoint_sha256"]))
    target = OUT / "final_models_frozen.json"
    if target.exists():
        require_test_unlocked()
    else:
        write(target, dict(frozen_at=now(), models=records, selection="all final checkpoints; no seed/epoch selection"))


def load_gate():
    path = ROOT / "src/tasks/velocity/phasor/phase_gate_v4_2.py"
    spec = importlib.util.spec_from_file_location("stateful_frozen_gate", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def cycle_summary(rows):
    names = sorted({r["sequence_id"] for r in rows})
    by = {}
    for name in names:
        rr = [r for r in rows if r["sequence_id"] == name]
        valid = [r for r in rr if r["evaluable"]]
        by[name] = dict(observed=len(rr), evaluable=len(valid), completed=sum(r["encoder_completed_one_turn"] for r in valid),
                        completion=float(np.mean([r["encoder_completed_one_turn"] for r in valid])) if valid else 0.)
    strata = {}
    for name in ("short", "medium", "long"):
        rr = [r for r in rows if r["evaluable"] and r["period_length_stratum"] == name]
        strata[name] = dict(cycles=len(rr), completion=float(np.mean([r["encoder_completed_one_turn"] for r in rr])) if rr else None)
    return dict(observed=len(rows), evaluable=sum(r["evaluable"] for r in rows),
                macro_completion=float(np.mean([r["completion"] for r in by.values()])) if by else None,
                by_sequence=by, by_length=strata)


def evaluate_model(row, seed):
    require_test_unlocked(); verify_seal(); verify_data()
    directory = OUT / "evaluation" / f"{row}_seed{seed}"
    if (directory / "completed.json").exists():
        return
    directory.mkdir(parents=True, exist_ok=True)
    source = OUT / "training" / f"{row}_seed{seed}" / "final.pt"
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    seed_all(seed, "cpu")
    model = core.PhaseNet(checkpoint["channels"], checkpoint["mode"]).eval()
    model.load_state_dict(checkpoint["model"])
    gate = load_gate()
    cohorts = {"motion_validation": load_records(OUT / "data/motion_index.json", "validation"),
               "simulation_development": load_records(OUT / "data/simulation_development_index.json"),
               "simulation_test": load_records(OUT / "data/simulation_test_index.json")}
    results = {}
    for cohort, records in cohorts.items():
        cycle_rows, point_rows, episode_rows = [], [], []
        for record in records:
            values = ((record["features"] - checkpoint["mean"]) / checkpoint["std"])[:, :checkpoint["channels"]]
            state = None; predictions = {k: [] for k in ("phase", "unwrapped", "rate", "anchor")}
            with torch.inference_mode():
                for start in range(0, len(values), 256):
                    out = model(torch.from_numpy(values[start:start + 256])[None], state)
                    state = out["state"]
                    for key in predictions:
                        predictions[key].append(out[key][0].numpy())
            predictions = {k: np.concatenate(v) for k, v in predictions.items()}
            if not all(np.isfinite(v).all() for v in predictions.values()):
                raise FloatingPointError("nonfinite held-out predictions")
            path = directory / (record["name"] + ".npz")
            np.savez_compressed(path, **predictions)
            ids = record["score_ids"]
            reference = record["target"].copy()
            # Terminal event is at zero phase; its half-open cycle owns no later frames.
            reference[record["left"]] = 0.
            report = gate.score_phase_cycles_v4_2(predictions["phase"][ids], reference[ids],
                np.full(len(ids), record["name"]), ids, record["left"][ids],
                reference_valid=np.isfinite(reference[ids]), period_cuts_frames=(50.66666666666666, 75.))
            for r in report["cycle_records"]:
                r["right_touchdown_count"] = int(record["right"][r["start_frame"]:r["end_frame"]].sum())
            cycle_rows.extend(report["cycle_records"])
            angle = np.arctan2(predictions["phase"][:, 0], predictions["phase"][:, 1])
            error = np.abs(np.angle(np.exp(1j * (angle - record["target"]))))
            valid = ids[np.isfinite(record["target"][ids])]
            mask = np.zeros(len(reference), bool)
            if record["domain"] == "motion":
                with np.load(DATA / "contact_consistency_filter_v1/phase_dataset_train_validation.npz", allow_pickle=False) as filtered:
                    names, frames = filtered["sequence_validation"], filtered["frame_validation"]
                    mask[frames[names == record["name"]]] = True
            else:
                for a, b, count in record["cycles"]:
                    if count == 1:
                        mask[a:b] = True
            consistent = valid[mask[valid]]
            point_rows.append(dict(sequence=record["name"], windows=len(valid), consistency_windows=len(consistent),
                all_mae_rad=float(error[valid].mean()) if len(valid) else None,
                consistency_mae_rad=float(error[consistent].mean()) if len(consistent) else None))
            evaluable = [r for r in report["cycle_records"] if r["evaluable"]]
            touchdowns = np.flatnonzero(record["left"])
            event_error = float(np.abs(np.angle(np.exp(1j * angle[touchdowns]))).mean()) if len(touchdowns) else None
            response = {}
            if record["domain"] == "simulation":
                switches = (5., 12.) if record["scenario"] == "speed_change" else ((5., 10.) if record["scenario"] == "push" else ())
                for at in switches:
                    ii = valid[(valid * core.DT >= at) & (valid * core.DT < at + 1.)]
                    response[str(at)] = dict(observed_frames=len(ii), mean_rate_hz=float(predictions["rate"][ii].mean()) if len(ii) else None,
                        phase_mae_rad=float(error[ii].mean()) if len(ii) else None)
            episode_rows.append(dict(name=record["name"], stratum=record.get("stratum", record["name"]),
                cycles=len(evaluable), completion=float(np.mean([r["encoder_completed_one_turn"] for r in evaluable])) if evaluable else None,
                point_mae_rad=float(error[valid].mean()) if len(valid) else None,
                touchdown_alignment_rad=event_error, fell=record.get("fell", False), response_windows=response,
                predictions_sha256=core.sha(path)))
        consistent_cycles = [r for r in cycle_rows if r["right_touchdown_count"] == 1]
        mismatch = [r for r in cycle_rows if r["right_touchdown_count"] != 1]
        if cohort == "motion_validation":
            if (len(cycle_rows), len(consistent_cycles), len(mismatch), sum(r["consistency_windows"] for r in point_rows)) != (603, 504, 99, 32878):
                raise ValueError("original common validation cohort changed")
        write(directory / (cohort + "_cycle_records.json"), cycle_rows)
        write(directory / (cohort + "_episodes.json"), episode_rows)
        result = dict(all=cycle_summary(cycle_rows), consistency=cycle_summary(consistent_cycles), mismatch=cycle_summary(mismatch),
                      point=point_rows,
                      all_macro_mae_rad=float(np.mean([r["all_mae_rad"] for r in point_rows if r["all_mae_rad"] is not None])) if any(r["all_mae_rad"] is not None for r in point_rows) else None,
                      consistency_macro_mae_rad=float(np.mean([r["consistency_mae_rad"] for r in point_rows if r["consistency_mae_rad"] is not None])) if any(r["consistency_mae_rad"] is not None for r in point_rows) else None)
        if cohort.startswith("simulation"):
            result["episode_summary"] = summarize_simulation(episode_rows)
        results[cohort] = result
    write(directory / "completed.json", dict(status="COMPLETED", row=row, seed=seed, ended_at=now(),
          checkpoint_sha256=core.sha(source), cohorts=results, environment=environment(),
          final_models_seal_sha256=core.sha(OUT / "final_models_frozen.json")))


def cpu_repro():
    """Two independently initialized small training runs, no CUDA context."""
    reports = []
    for attempt in (1, 2):
        seed_all(20261025, "cpu")
        model = core.PhaseNet(29, "observer", width=8)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.0001)
        x = torch.randn(2, 80, 29)
        reference = torch.zeros(2, 80)
        reference[:, 20:70] = torch.arange(50) * (2 * np.pi / 50)
        losses = []
        started = time.monotonic()
        for _ in range(20):
            optimizer.zero_grad(set_to_none=True)
            loss, _ = core.cycle_objective(model(x), reference, torch.tensor([20, 20]), torch.tensor([70, 70]))
            if not torch.isfinite(loss):
                raise FloatingPointError("CPU smoke nonfinite loss")
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step(); losses.append(float(loss.detach()))
        with torch.inference_mode():
            predictions = model(x)["phase"].numpy()
        state_hash = hashlib.sha256()
        for name, value in sorted(model.state_dict().items()):
            state_hash.update(name.encode()); state_hash.update(value.cpu().numpy().tobytes())
        reports.append(dict(attempt=attempt, model_state_sha256=state_hash.hexdigest(),
            predictions_sha256=hashlib.sha256(predictions.tobytes()).hexdigest(),
            losses_sha256=hashlib.sha256(np.array(losses).tobytes()).hexdigest(),
            first_loss=losses[0], final_loss=losses[-1], elapsed_seconds=time.monotonic() - started))
    keys = ("model_state_sha256", "predictions_sha256", "losses_sha256")
    identical = all(reports[0][key] == reports[1][key] for key in keys)
    write(OUT / "verification/cpu_repro.json", dict(status="PASS" if identical else "FAIL", runs=reports,
          environment=environment(), scope="same-machine CPU reduced-width numerical reproducibility; no quality claim",
          cuda_initialized=torch.cuda.is_initialized()))
    if not identical:
        raise RuntimeError("CPU reproducibility failed")
    print(json.dumps(dict(stage="cpu_repro", identical=identical, runs=reports)), flush=True)


def gpu_smoke():
    if not core.previous_ready(PREVIOUS):
        raise RuntimeError("previous study incomplete")
    verify_seal()
    seed_all(20261025, "cuda")
    model = core.PhaseNet(29, "observer", width=8).cuda()
    x = torch.randn(2, 80, 29, device="cuda")
    reference = torch.zeros(2, 80, device="cuda")
    reference[:, 20:70] = torch.arange(50, device="cuda") * (2 * np.pi / 50)
    loss, _ = core.cycle_objective(model(x), reference, torch.tensor([20, 20], device="cuda"), torch.tensor([70, 70], device="cuda"))
    loss.backward()
    if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
        raise FloatingPointError("GPU observer smoke failed")
    write(OUT / "verification/gpu_smoke.json", dict(status="PASS", ended_at=now(), loss=float(loss.detach()),
          device=torch.cuda.get_device_name(), environment=environment()))


def summarize():
    require_test_unlocked(); verify_seal(); verify_data()
    runs = [read(OUT / "evaluation" / f"{j['row']}_seed{j['seed']}" / "completed.json") for j in core.matrix()]
    pairs = [("observer_motion", "direct_motion", "structure_motion"),
             ("observer_mixed", "direct_mixed", "structure_mixed"),
             ("observer_mixed", "observer_motion", "mixed_training_distribution"),
             ("observer_sensors_mixed", "observer_mixed", "extra_sensors")]
    effects = []
    for treatment, control, label in pairs:
        for cohort in ("motion_validation", "simulation_test"):
            differences = []
            for seed in core.SEEDS:
                values = []
                for row in (treatment, control):
                    run = next(r for r in runs if r["row"] == row and r["seed"] == seed)["cohorts"][cohort]
                    if cohort == "motion_validation":
                        values.append((run["consistency"]["macro_completion"], run["consistency_macro_mae_rad"]))
                    else:
                        values.append((run["episode_summary"]["macro_completion_with_no_cycle_zero"], run["episode_summary"]["macro_point_mae_rad"]))
                differences.append(dict(seed=seed, completion_difference=values[0][0] - values[1][0],
                    point_mae_difference_rad=values[0][1] - values[1][1] if values[0][1] is not None and values[1][1] is not None else None))
            mean_completion = float(np.mean([r["completion_difference"] for r in differences]))
            point = [r["point_mae_difference_rad"] for r in differences if r["point_mae_difference_rad"] is not None]
            mean_point = float(np.mean(point)) if len(point) == 3 else None
            branch = ("IMPROVEMENT_WITHOUT_MEAN_POINT_WORSENING" if mean_point is not None and mean_completion > 0 and mean_point <= 0
                      else "TRADEOFF" if mean_point is not None and mean_completion > 0 and mean_point > 0
                      else "NO_DEMONSTRATED_JOINT_IMPROVEMENT")
            effects.append(dict(comparison=label, treatment=treatment, control=control, cohort=cohort,
                completion_difference_mean=mean_completion,
                completion_difference_seed_sd=float(np.std([r["completion_difference"] for r in differences], ddof=1)),
                point_mae_difference_mean_rad=mean_point, descriptive_branch=branch, per_seed=differences))
    write(OUT / "comparison.json", dict(status="COMPLETED", ended_at=now(), runs=runs, effects=effects,
          interpretation="three optimizer seeds, paired finite cohorts; descriptive mean effects, no independent subject or closed-loop superiority claim"))
    lines = ["# 连续相位联合训练结果", "", "全部固定预算训练、开发集与冻结后的新仿真测试结果；无检查点/seed挑选。", "",
             "| row | seed | 原504完成率 | 原504点MAE rad | 新sim保守完成率 | 新sim点MAE rad |", "|---|---:|---:|---:|---:|---:|"]
    for run in runs:
        motion, sim = run["cohorts"]["motion_validation"], run["cohorts"]["simulation_test"]["episode_summary"]
        lines.append(f"| {run['row']} | {run['seed']} | {motion['consistency']['macro_completion']:.4f} | {motion['consistency_macro_mae_rad']:.4f} | {sim['macro_completion_with_no_cycle_zero']:.4f} | {sim['macro_point_mae_rad']} |")
    lines += ["", "配对差、完整603/99周期、各序列/长度层、跌倒与无周期覆盖见 comparison.json 和各运行原始记录。",
              "这些是离线相位质量结果；优于固定clock的控制主张仍需下一次冻结后的三组等预算闭环比较。"]
    (OUT / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def matching_process(pid, script):
    import psutil
    try:
        identity = Path(script).stem
        return any(Path(token).stem == identity or token.rsplit(".", 1)[-1] == identity
                   for token in psutil.Process(int(pid)).cmdline())
    except (psutil.NoSuchProcess, psutil.AccessDenied, TypeError, ValueError):
        return False


def gpu_busy():
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, check=True)
    return [int(line.strip()) for line in result.stdout.splitlines() if line.strip().isdigit()]


def child(stage, **kwargs):
    command = [sys.executable, "-u", "-m", "scripts.run_g1_stateful_phase_20261025", "--stage", stage]
    for key, value in kwargs.items():
        command += ["--" + key.replace("_", "-"), str(value)]
    name = "_".join([stage] + [str(v) for v in kwargs.values()])
    log = OUT / "logs" / f"{name}_{time.time_ns()}.log"; log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as stream:
        # Hash seeding takes effect in the new interpreter, not this queue process.
        os.environ["PYTHONHASHSEED"] = str(kwargs.get("seed", 20261025))
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        write(OUT / "queue_status.json", dict(status="RUNNING", stage=stage, kwargs=kwargs,
              child_pid=process.pid, argv=command, log_path=str(log), updated_at=now()))
        exit_code = process.wait()
    if exit_code:
        raise RuntimeError(f"child failed ({exit_code}): {log}")


def queue():
    OUT.mkdir(parents=True, exist_ok=True)
    verify_seal()
    if read(OUT / "verification/cpu_repro.json")["status"] != "PASS":
        raise RuntimeError("CPU reproducibility must pass before queue")
    lock = OUT / "queue.lock"
    if lock.exists():
        previous = read(lock)
        if matching_process(previous.get("pid"), Path(__file__).name):
            raise RuntimeError("continuous-phase queue already active")
        lock.rename(OUT / f"stale_queue_lock_{time.time_ns()}.json")
    status = read(OUT / "queue_status.json") if (OUT / "queue_status.json").exists() else {}
    last = status.get("last_stage", status)
    if matching_process(last.get("child_pid"), Path(__file__).name):
        raise RuntimeError("previous queue child remains active; do not duplicate GPU jobs")
    with lock.open("x", encoding="utf-8") as stream:
        json.dump(dict(pid=os.getpid(), started_at=now(), script=str(Path(__file__))), stream)
    try:
        while True:
            status = read(PREVIOUS / "queue_status.json") if (PREVIOUS / "queue_status.json").exists() else {}
            ready = core.previous_ready(PREVIOUS)
            old_queue = read(PREVIOUS / "queue.lock") if (PREVIOUS / "queue.lock").exists() else {}
            old_alive = matching_process(old_queue.get("pid"), "run_g1_phase_adaptation_20261024")
            old_child = matching_process(status.get("child_pid"), "run_g1_phase_adaptation_20261024")
            if ready and not old_alive and not old_child:
                break
            write(OUT / "queue_status.json", dict(status="WAITING_FOR_PREVIOUS_STUDY", updated_at=now(),
                  prior_matrix_complete=ready, prior_queue_alive=old_alive, prior_child_alive=old_child,
                  prior_status=status.get("status"), prior_stage=status.get("stage")))
            if status.get("status") == "FAILED" or not old_alive and not old_child and not ready:
                raise RuntimeError("prior study stopped incomplete; new GPU work remains blocked")
            time.sleep(30)
        from scripts import g1_phase_adaptation_20261024 as adaptation
        adaptation.verify_assets()
        while gpu_busy():
            write(OUT / "queue_status.json", dict(status="WAITING_FOR_GPU", updated_at=now(), compute_pids=gpu_busy()))
            time.sleep(30)
        child("gpu-smoke")
        child("collect-smoke", split="train", terrain="rough", arm="learned")
        for split in ("train", "development"):
            for job in simulation_jobs(split):
                child("collect", split=split, terrain=job["terrain"], arm=job["arm"])
        child("prepare-data")
        for job in core.matrix():
            child("train", **job)
        child("freeze-models")
        for job in simulation_jobs("test"):
            child("collect", split="test", terrain=job["terrain"], arm=job["arm"])
        child("prepare-test")
        for job in core.matrix():
            child("evaluate", **job)
        child("summarize")
        write(OUT / "queue_status.json", dict(status="COMPLETED", ended_at=now(), comparison=str(OUT / "comparison.json")))
    except BaseException as error:
        previous = read(OUT / "queue_status.json") if (OUT / "queue_status.json").exists() else {}
        write(OUT / "queue_status.json", dict(status="FAILED", ended_at=now(), error=repr(error),
              traceback=traceback.format_exc(), last_stage=previous))
        raise
    finally:
        lock.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("prepare-motion", "cpu-repro", "freeze", "queue", "gpu-smoke",
        "collect-smoke", "collect", "prepare-data", "train", "freeze-models", "prepare-test", "evaluate", "summarize"))
    parser.add_argument("--split", choices=("train", "development", "test"), default="train")
    parser.add_argument("--terrain", choices=("flat", "rough"), default="flat")
    parser.add_argument("--arm", choices=("clock", "learned"), default="clock")
    parser.add_argument("--row", choices=core.ROWS, default="observer_motion")
    parser.add_argument("--seed", type=int, default=core.SEEDS[0])
    args = parser.parse_args()
    actions = {"prepare-motion": prepare_motion, "cpu-repro": cpu_repro, "freeze": freeze, "queue": queue,
               "gpu-smoke": gpu_smoke, "prepare-data": prepare_training_data, "freeze-models": freeze_models,
               "prepare-test": lambda: prepare_simulation("test"), "summarize": summarize,
               "collect": lambda: collect(args.split, args.terrain, args.arm),
               "collect-smoke": lambda: collect(args.split, args.terrain, args.arm, smoke=True),
               "train": lambda: train(args.row, args.seed), "evaluate": lambda: evaluate_model(args.row, args.seed)}
    actions[args.stage]()


if __name__ == "__main__":
    main()
