"""Train the v4 TCTST readout and combine it with immutable completed v3 TCN runs."""
from __future__ import annotations

import json
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_g1_phase_model_comparison_v2 as comparison  # noqa: E402

OUT = ROOT / 'results/g1_phase_model_comparison_20261023/tctst_residual_v4'
OLD_RUNS = ROOT / 'results/g1_phase_model_comparison_20261023/matched_100ep'
PREREG = ROOT / 'results/g1_phase_model_comparison_20261023/preregistration_amendment_v4.md'
V3_PREREG = ROOT / 'results/g1_phase_model_comparison_20261023/preregistration_amendment_v3.md'
PRETRAIN_DIR = ROOT / 'results/g1_phase_model_comparison_20261023/pretrain_v3'
INTERRUPTED_TCTST = OLD_RUNS / 'tctst_repaired_seed_20261021'


def completed_tcn_runs() -> list[dict]:
    records = []
    for family in ('tcn20', 'tcn100_rate'):
        for seed in comparison.SEEDS:
            run_dir = OLD_RUNS / f'{family}_seed_{seed}'
            result_path = run_dir / 'completed.json'
            result = json.loads(result_path.read_text(encoding='utf-8'))
            if (result.get('status') != 'COMPLETED' or result.get('model') != family
                    or int(result.get('seed', -1)) != seed or int(result.get('epochs', 0)) != 100):
                raise ValueError(f'incomplete or mismatched frozen TCN run: {result_path}')
            checkpoint = run_dir / 'final_checkpoint.pt'
            predictions = run_dir / 'direct_predictions.npy'
            if comparison.sha(checkpoint) != result.get('checkpoint_sha256'):
                raise ValueError(f'TCN checkpoint hash mismatch: {checkpoint}')
            if comparison.sha(predictions) != result.get('predictions_sha256'):
                raise ValueError(f'TCN prediction hash mismatch: {predictions}')
            result['reused_from'] = str(result_path.relative_to(ROOT))
            records.append(result)
    prereg_hashes = {r.get('preregistration_sha256') for r in records}
    if prereg_hashes != {comparison.sha(V3_PREREG)}:
        raise ValueError('completed TCN runs do not share the frozen v3 amendment hash')
    return records


def build_asset_seal(records: list[dict]) -> dict:
    verdict_path = PRETRAIN_DIR / 'verdict.json'
    verdict = json.loads(verdict_path.read_text(encoding='utf-8'))
    if verdict.get('verdict') != 'PRETRAIN_COLLAPSE' or verdict.get('checkpoint_written'):
        raise ValueError('TCTST pretraining audit no longer matches the frozen scratch branch')
    return {
        'schema': 'g1.tctst_residual_comparison_asset_seal.v1',
        'amendment_path': str(PREREG.relative_to(ROOT)),
        'amendment_sha256': comparison.sha(PREREG),
        'code_hashes': {
            str(path.relative_to(ROOT)): comparison.sha(path)
            for path in comparison.CODE_PATHS
        },
        'data_hashes': {
            str(path.relative_to(ROOT)): comparison.sha(path)
            for path in comparison.DATA_PATHS
        },
        'pretraining_audit': {
            'verdict': verdict['verdict'],
            'verdict_sha256': comparison.sha(verdict_path),
            'manifest_sha256': comparison.sha(PRETRAIN_DIR / 'manifest.json'),
            'diagnosis_sha256': comparison.sha(PRETRAIN_DIR / 'diagnosis.json'),
        },
        'interrupted_old_tctst': {
            'manifest_sha256': comparison.sha(INTERRUPTED_TCTST / 'manifest.json'),
            'interruption_sha256': comparison.sha(INTERRUPTED_TCTST / 'interruption.json'),
            'partial_checkpoint_sha256': comparison.sha(INTERRUPTED_TCTST / 'partial_checkpoint.pt'),
            'logged_epochs_sha256': comparison.sha(INTERRUPTED_TCTST / 'epochs.jsonl'),
        },
        'reused_tcn_runs': [
            {
                'model': record['model'], 'seed': record['seed'],
                'completed_manifest': record['reused_from'],
                'completed_manifest_sha256': comparison.sha(ROOT / record['reused_from']),
                'checkpoint_sha256': record['checkpoint_sha256'],
                'predictions_sha256': record['predictions_sha256'],
                'source_preregistration_sha256': record['preregistration_sha256'],
            }
            for record in records
        ],
    }


def write_or_verify_seal(payload: dict) -> None:
    path = OUT / 'asset_seal_v4.json'
    if path.exists():
        observed = json.loads(path.read_text(encoding='utf-8'))
        if observed != payload:
            raise ValueError('v4 asset seal differs from the current code or source artifacts')
    else:
        comparison.write_json(path, payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight-only', action='store_true',
                        help='verify frozen assets and write the seal without starting training')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'comparison_summary.json').exists():
        raise FileExistsError(f'refusing to overwrite completed v4 results in {OUT}')
    comparison.OUT = OUT
    comparison.PREREG = PREREG
    comparison.CODE_PATHS = (
        Path(__file__), comparison.ROOT / 'scripts/run_g1_phase_model_comparison_v2.py',
        comparison.ROOT / 'scripts/g1_tctst_faithful_v2.py',
        comparison.ROOT / 'scripts/g1_external_models.py',
        comparison.ROOT / 'scripts/run_g1_external_methods.py',
        comparison.ROOT / 'scripts/run_g1_long_history_comparison.py',
        comparison.ROOT / 'src/tasks/velocity/phasor/encoder_tctst_runtime.py',
        comparison.ROOT / 'src/tasks/velocity/mdp/observations.py',
        comparison.ROOT / 'scripts/prepare_g1_selected_model_runtime_v2.py',
        comparison.ROOT / 'tests/test_g1_tctst_readout.py',
        comparison.ROOT / 'tests/test_g1_long_window_model_runtimes.py',
        comparison.ROOT / 'src/tasks/velocity/phasor/phase_gate_v4_2.py',
    )
    records = completed_tcn_runs()
    seal = build_asset_seal(records)
    write_or_verify_seal(seal)
    if args.preflight_only:
        print(json.dumps({
            'preflight': 'PASS', 'asset_seal': str((OUT / 'asset_seal_v4.json').relative_to(ROOT)),
            'amendment_sha256': seal['amendment_sha256'],
            'reused_tcn_runs': len(records),
            'pretraining_verdict': seal['pretraining_audit']['verdict'],
        }, ensure_ascii=False, indent=2))
        return 0
    matched_runs = OUT / 'matched_100ep'
    if matched_runs.exists():
        if any(matched_runs.iterdir()):
            raise FileExistsError(f'refusing to overwrite existing TCTST runs in {matched_runs}')
        matched_runs.rmdir()
    matched_runs.mkdir()
    train_bank = comparison.base.FrameBank('train')
    val_bank = comparison.base.FrameBank('validation')
    for seed in comparison.SEEDS:
        records.append(comparison.train_one(
            'tctst_repaired', seed, train_bank, val_bank, pretrained_path=None))
    summary = comparison.summarize_matched(records)
    summary.update({
        'schema': 'g1.phase_model_comparison.v4',
        'tctst_pretrained': False,
        'tctst_readout': 'mean over layerwise channel-pooled transformer outputs + sqrt(dim) scaled temporal-token channel mean',
        'reused_tcn_protocol': 'preregistration_amendment_v3.md',
        'current_protocol': str(PREREG.relative_to(ROOT)),
        'training_records': records,
    })
    comparison.write_json(OUT / 'comparison_summary.json', summary)
    selection = {
        'schema': 'g1.phase_model_selection.v2',
        'selected_family': summary['selected_family'],
        'selected_seed_for_control': 20261021,
        'selection_rule': summary['selection_rule'],
        'evidence_path': str((OUT / 'comparison_summary.json').relative_to(ROOT)),
        'requires_separate_control_comparison': True,
        'does_not_claim_deployment_or_freeze_eligibility': True,
        'protocol_sha256': comparison.sha(PREREG),
    }
    comparison.write_json(OUT / 'selected_model.json', selection)
    print(json.dumps({
        'selected_family': selection['selected_family'],
        'selected_seed_for_control': selection['selected_seed_for_control'],
        'models': summary['models'],
        'evidence_path': selection['evidence_path'],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
