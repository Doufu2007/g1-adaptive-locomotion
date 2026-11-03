"""Frozen inputs and scientific invariants for matched G1 policy adaptation."""
from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

from scripts import run_g1_best_phase_control as base

ROOT = base.ROOT
OUT = ROOT / "results/g1_phase_controller_adaptation_20261024"
PREVIOUS = ROOT / "results/g1_phase_model_comparison_20261023/control_v1"
RUNTIME = ROOT / "results/g1_phase_model_comparison_20261023/runtime_selected/phasor_runtime.pt"
RUNTIME_SHA = "2f315766e6c782bf5d930a1d59f1bc9e782a5e7e194c92c3349b1aa450502339"
TRAIN_SEEDS = (20261401, 20261402, 20261403)
EVAL_SEEDS = tuple(range(20261501, 20261511))
SCENARIOS = ("steady", "speed_change", "push")
ITERATIONS, NUM_ENVS, STEPS = 501, 64, 24
TRANSITIONS = ITERATIONS * NUM_ENVS * STEPS
TRAIN_TERRAIN_SEED, EVAL_TERRAIN_SEED = 20261400, 20261500
bootstrap, sha, write, now = base.bootstrap, base.sha, base.write, base.now


def make_cfg(terrain, arm, seed, *, play, num_envs, horizon):
    """Identical task rewards; phase source is the only intended method change."""
    if terrain not in ("flat", "rough") or arm not in ("clock", "learned"):
        raise ValueError("unknown terrain or phase source")
    cfg = base.make_cfg(terrain, "clock", seed, play=play, num_envs=num_envs, horizon=horizon)
    cfg.rewards.pop("foot_gait")
    if terrain == "rough":
        cfg.scene.terrain.terrain_generator.seed = EVAL_TERRAIN_SEED if play else TRAIN_TERRAIN_SEED
        if not play:
            cfg.sim.nconmax = max(cfg.sim.nconmax or 0, 128)
    if arm == "learned":
        from mjlab.managers.event_manager import EventTermCfg
        from src.tasks.velocity.mdp import observations
        for group in ("actor", "critic"):
            old = cfg.observations[group].terms["phase"]
            cfg.observations[group].terms["phase"] = replace(
                old, func=observations.learned_phase, params={"encoder_path": str(RUNTIME)})
        cfg.events["reset_phasor_phase_history"] = EventTermCfg(
            func=observations.reset_learned_phase_history, mode="reset",
            params={"encoder_path": str(RUNTIME)})
    return cfg


def canonical(value):
    if isinstance(value, dict):
        return {str(k): canonical(v) for k, v in sorted(value.items(), key=lambda x:str(x[0]))}
    if isinstance(value, (list, tuple)):
        return [canonical(v) for v in value]
    if callable(value):
        return f"{value.__module__}.{value.__qualname__}"
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def common_cfg_digest(cfg):
    value = asdict(cfg)
    for group in ("actor", "critic"):
        value["observations"][group]["terms"].pop("phase")
    value["events"].pop("reset_phasor_phase_history", None)
    return hashlib.sha256(json.dumps(canonical(value), sort_keys=True).encode()).hexdigest()


def dependency_ready(directory=PREVIOUS):
    """A missing/dead process alone must never count as a completed study."""
    expected_all = {(s,c) for s in range(20261201,20261231) for c in SCENARIOS}
    try:
        for terrain in ("flat", "rough"):
            for arm in ("clock", "learned"):
                seen = set()
                for batch in range(1,4):
                    path = directory / "rollouts" / f"{terrain}_{arm}_batch{batch:02d}"
                    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
                    rows = json.loads((path / "episodes.json").read_text(encoding="utf-8"))
                    expected_seeds = list(range(20261201+(batch-1)*10,20261201+batch*10))
                    expected = {(s,c) for s in expected_seeds for c in SCENARIOS}
                    keys = {(r["seed"],r["scenario"]) for r in rows}
                    if (manifest.get("status") != "COMPLETED" or len(rows) != 30
                            or manifest.get("episode_count") != 30 or keys != expected
                            or manifest.get("seeds") != expected_seeds
                            or tuple(manifest.get("scenarios",())) != SCENARIOS
                            or seen & keys):
                        return False
                    seen |= keys
                if seen != expected_all:
                    return False
    except (OSError, ValueError, KeyError, TypeError):
        return False
    return True


def training_jobs():
    jobs = []
    for terrain in ("flat", "rough"):
        for index, seed in enumerate(TRAIN_SEEDS):
            for arm in (("clock","learned") if index % 2 == 0 else ("learned","clock")):
                jobs.append(dict(terrain=terrain,arm=arm,seed=seed))
    return jobs


def train_dir(terrain, arm, seed):
    return OUT / "training" / f"{terrain}_{arm}_seed{seed}"


def evaluation_jobs(terrain, seed):
    return [dict(label=label,arm=arm,policy=str(train_dir(terrain,trained,seed)/"policy.onnx"),
                 output=str(OUT/"evaluation"/f"{terrain}_seed{seed}_{label}"))
            for label, trained, arm in (("clock_clock","clock","clock"),
                ("clock_learned","clock","learned"),("learned_learned","learned","learned"))]


def choose_resume_checkpoint(directory,seed,terrain,arm):
    """Preserve corrupt/interrupted files and select the latest intact matching state."""
    import torch
    rejected=[]
    paths=sorted(directory.glob("model_*.pt"),key=lambda p:int(p.stem.split('_')[-1]),reverse=True)
    for candidate in paths:
        try:
            saved=torch.load(candidate,map_location="cpu",weights_only=False)
            metadata=saved["infos"]["adaptation_state"]
            if (metadata["completed_updates"]!=int(saved["iter"])+1
                    or metadata["ppo_seed"]!=seed or metadata["arm"]!=arm
                    or metadata["terrain"]!=terrain or metadata["encoder_sha256"]!=RUNTIME_SHA
                    or not 0<metadata["completed_updates"]<=ITERATIONS):
                raise ValueError("checkpoint identity or update counter mismatch")
        except Exception as error:
            rejected.append(dict(path=str(candidate),error=repr(error)))
            continue
        return candidate,rejected
    raise ValueError(f"no intact matching checkpoint: {rejected}")


def paired_training_effect(rows, terrain, treatment, control, metric):
    """Average paired episodes within a trained policy before estimating uncertainty."""
    import numpy as np
    from scipy.stats import t
    per_seed = []
    for seed in TRAIN_SEEDS:
        groups = []
        for label in (treatment, control):
            group = {}
            for row in rows:
                if (row["terrain"],row["train_seed"],row["label"]) == (terrain,seed,label):
                    key = (row["scenario"],row["seed"])
                    if key in group:
                        raise ValueError("duplicate paired cohort row")
                    group[key] = float(row["fell"] if metric == "fall_rate" else row[metric])
            groups.append(group)
        a,b = groups
        if not a or set(a) != set(b):
            raise ValueError("missing or unmatched paired evaluation cohort")
        per_seed.append(dict(train_seed=seed,difference=float(np.mean([a[k]-b[k] for k in sorted(a)])),
                             paired_episode_count=len(a)))
    differences = np.asarray([r["difference"] for r in per_seed])
    effect = float(differences.mean())
    sd = float(differences.std(ddof=1))
    se = sd / np.sqrt(len(differences))
    half = float(t.ppf(.975,len(differences)-1)*se)
    p = float(2*t.sf(abs(effect/se),len(differences)-1)) if se > 0 else (1. if effect == 0 else 0.)
    return dict(mean_difference=effect,training_seed_count=len(per_seed),
                paired_t_95pct_ci=[effect-half,effect+half],two_sided_p=p,
                per_training_seed=per_seed,metric=metric,terrain=terrain,
                treatment=treatment,control=control,
                interpretation="three paired PPO runs; conditional on the frozen finite evaluation cohort")


def verify_runtime():
    if sha(RUNTIME) != RUNTIME_SHA:
        raise ValueError("frozen selected TCTST runtime changed")


def freeze_assets():
    """Call only after the prior study and independent GPU smoke have completed."""
    verify_runtime()
    paths = [Path(__file__),ROOT/"scripts/run_g1_phase_adaptation_20261024.py",
        ROOT/"scripts/summarize_g1_selected_phase_control_v3.py",
        ROOT/"scripts/g1_best_phase_batched.py",ROOT/"scripts/run_g1_best_phase_control.py",
        ROOT/"scripts/g1_closed_loop_metrics.py",
        ROOT/"docs/superpowers/specs/2026-10-24-g1-phase-policy-adaptation-design.md",
        ROOT/"docs/superpowers/plans/2026-10-24-g1-phase-policy-adaptation.md",
        ROOT/"tests/test_g1_phase_adaptation_20261024.py",RUNTIME]
    paths += list((ROOT/"src/tasks/velocity/phasor").glob("*.py"))
    paths += [ROOT/p for p in ("src/tasks/velocity/mdp/observations.py",
        "src/tasks/velocity/mdp/rewards.py","src/tasks/velocity/velocity_env_cfg.py",
        "src/tasks/velocity/config/g1/env_cfgs.py","src/tasks/velocity/config/g1/rl_cfg.py",
        "src/tasks/velocity/rl/runner.py")]
    paths += [base.POLICIES[t]/"model_9900.pt" for t in ("flat","rough")]
    lock = dict(schema="g1.phase_policy_adaptation.asset_seal.v1",frozen_at=now(),
        user_authorization="2026-10-24: wait for current work, then run the three-arm equal-budget adaptation experiment",
        iterations=ITERATIONS,num_envs=NUM_ENVS,steps_per_iteration=STEPS,transitions_per_run=TRANSITIONS,
        train_seeds=TRAIN_SEEDS,eval_seeds=EVAL_SEEDS,
        assets=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in paths])
    target = OUT/"asset_seal.json"
    if target.exists():
        verify_assets()
    else:
        write(target,lock)
    return lock


def verify_assets():
    lock = json.loads((OUT/"asset_seal.json").read_text(encoding="utf-8"))
    for asset in lock["assets"]:
        p = ROOT/asset["path"]
        if not p.is_file() or sha(p) != asset["sha256"]:
            raise ValueError(f"adaptation sealed asset changed: {asset['path']}")
    verify_runtime()
    return lock
