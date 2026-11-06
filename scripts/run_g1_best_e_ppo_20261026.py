"""Isolated, matched-budget rerun using the offline-best E checkpoint.

Only process-local experiment identity and provenance change. The reviewed
training, online inference, evaluation, budgets and statistics are reused.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODULE = 'scripts.run_g1_best_e_ppo_20261026'
PROTOCOL = ROOT/'docs/superpowers/specs/2026-10-26-g1-best-e-ppo-rerun.md'
SELECTION = ('User-requested offline-best E: observer_sensors_mixed seed20261026; '
             'selected before this PPO rerun; no selection using new PPO outcomes')
AUTHORIZATION = ('2026-10-26 user: use the best E and rerun on '
                 '<ssh-host>:49663; shut down after completion')
_installed = False


def install():
    global _installed
    from scripts.run_g1_stateful_flat_ppo_remote_20261025 import install as linux_install
    from scripts import g1_stateful_flat_ppo as study
    from scripts import run_g1_stateful_flat_ppo_20261025 as runner
    if _installed:
        return runner
    linux_install()
    study.OUT = ROOT/'results/g1_stateful_flat_ppo_best_e_20261026'
    study.ENCODER_SEED = 20261026
    study.RUNTIME = (study.SNAPSHOT/'results/g1_stateful_phase_20261025/training/'
                     'observer_sensors_mixed_seed20261026/final.pt')
    study.RUNTIME_SHA = '8e6b15dcf409f51948f49c18fc04b686ba502e53586476db9c78d8c35a0fc346'
    runner.MODULE = MODULE
    original_assets, original_worker, original_summary = study.asset_paths, runner.worker, runner.summarize

    def assets():
        return original_assets() + [Path(__file__).resolve(), PROTOCOL,
            ROOT/'tests/test_g1_best_e_ppo_20261026.py',
            ROOT/'scripts/g1_best_e_delivery_20261026.py',
            ROOT/'tests/test_g1_best_e_delivery_20261026.py']

    def freeze():
        study.verify_runtime()
        if (study.OUT/'asset_seal.json').exists():
            return study.verify_assets()
        paths = assets()
        lock = dict(schema='g1.stateful_flat_ppo.asset_seal.v1', frozen_at=study.now(),
            authorization=AUTHORIZATION, encoder_selection=SELECTION,
            encoder_seed=study.ENCODER_SEED, encoder_sha256=study.RUNTIME_SHA,
            protocol_amendment_sha256=study.sha(PROTOCOL),
            train_seeds=list(study.TRAIN_SEEDS), eval_seeds=list(study.EVAL_SEEDS),
            iterations=study.ITERATIONS, transitions_per_run=study.TRANSITIONS,
            training_jobs=study.training_jobs(),
            assets=[dict(path=str(p.relative_to(ROOT)), sha256=study.sha(p)) for p in paths])
        for path in paths:
            if path.suffix in ('.py', '.md'):
                target = study.OUT/'source_snapshot'/path.relative_to(ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(path.read_bytes())
        study.write(study.OUT/'asset_seal.json', lock)
        return lock

    def worker(stage, arm, seed, *, resume=False, label=None):
        try:
            return original_worker(stage, arm, seed, resume=resume, label=label)
        finally:
            if stage in ('smoke', 'train'):
                directory = (study.OUT/'smoke'/f'flat_{arm}' if stage == 'smoke'
                             else study.train_dir('flat', arm, seed))
                path = directory/'manifest.json'
                if path.exists():
                    record = runner.read(path)
                    record.update(experiment='offline-best E matched flat PPO rerun',
                        encoder_seed=study.ENCODER_SEED, encoder_selection=SELECTION,
                        protocol_sha256=study.sha(PROTOCOL))
                    study.write(path, record)

    def summarize():
        original_summary()
        path = study.OUT/'comparison.json'
        record = runner.read(path)
        record.update(encoder_seed=study.ENCODER_SEED, encoder_selection=SELECTION,
                      protocol_amendment_sha256=study.sha(PROTOCOL))
        study.write(path, record)
        path = study.OUT/'comparison.md'
        report = path.read_text(encoding='utf-8').replace(
            '候选按最小E种子编号选定，没有按离线或PPO成绩重选。',
            '按用户要求选用离线平地表现最佳的E（seed20261026）；本轮PPO结果未用于挑选模型。')
        path.write_text(report, encoding='utf-8')

    study.asset_paths, study.freeze_assets = assets, freeze
    runner.worker, runner.summarize = worker, summarize
    _installed = True
    return runner


if __name__ == '__main__':
    install().main()
