"""Merge the frozen six TCN runs and three repaired-TCTST runs for v4 selection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_g1_phase_model_comparison_v2 as comparison  # noqa: E402
from scripts import run_g1_tctst_residual_v4_comparison as v4  # noqa: E402

OUT = ROOT / 'results/g1_phase_model_comparison_20261023/tctst_residual_v4'
TCTST_RUNS = {
    20261021: OUT / 'matched_100ep/tctst_repaired_seed_20261021',
    20261022: OUT / 'recovery_seed_20261022_attempt04/tctst_repaired_seed_20261022',
    20261023: OUT / 'recovery_seed_20261023_attempt01/tctst_repaired_seed_20261023',
}


def load_pretraining_verdict() -> tuple[Path, dict]:
    path = v4.PRETRAIN_DIR / 'verdict.json'
    return path, json.loads(path.read_text(encoding='utf-8'))


def load_tctst_run(seed: int, run_dir: Path, prereg_hash: str) -> dict:
    completed_path = run_dir / 'completed.json'
    manifest_path = run_dir / 'manifest.json'
    checkpoint_path = run_dir / 'final_checkpoint.pt'
    predictions_path = run_dir / 'direct_predictions.npy'
    if not all(path.is_file() for path in (completed_path, manifest_path, checkpoint_path, predictions_path)):
        raise FileNotFoundError(f'TCTST run is incomplete: {run_dir}')
    completed = json.loads(completed_path.read_text(encoding='utf-8'))
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    expected = {
        'status': 'COMPLETED', 'model': 'tctst_repaired', 'seed': seed,
        'epochs': 100, 'window': 100, 'preregistration_sha256': prereg_hash,
    }
    if any(completed.get(key) != value for key, value in expected.items()):
        raise ValueError(f'TCTST completed manifest violates frozen run contract: {completed_path}')
    if manifest.get('status') != 'COMPLETED' or manifest.get('checkpoint_sha256') != completed.get('checkpoint_sha256'):
        raise ValueError(f'TCTST run manifest disagrees with completed record: {manifest_path}')
    if comparison.sha(checkpoint_path) != completed.get('checkpoint_sha256'):
        raise ValueError(f'TCTST checkpoint hash mismatch: {checkpoint_path}')
    if comparison.sha(predictions_path) != completed.get('predictions_sha256'):
        raise ValueError(f'TCTST prediction hash mismatch: {predictions_path}')
    metrics = completed['metrics']
    if (metrics['cycles']['consistency']['observed_cycles'] != 504
            or metrics['cycles']['all']['observed_cycles'] != 603
            or metrics['point']['consistency']['windows'] != 32878):
        raise ValueError(f'TCTST validation cohort differs from frozen common stream: {completed_path}')
    return {
        **completed,
        'reused_from': str(completed_path.relative_to(ROOT)),
        'completed_manifest_sha256': comparison.sha(completed_path),
        'run_manifest_sha256': comparison.sha(manifest_path),
        'run_dir': str(run_dir.relative_to(ROOT)),
        'recovery_method': manifest.get('recovery'),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    v4.OUT = OUT
    v4.PREREG = ROOT / 'results/g1_phase_model_comparison_20261023/preregistration_amendment_v4.md'
    v4.V3_PREREG = ROOT / 'results/g1_phase_model_comparison_20261023/preregistration_amendment_v3.md'
    prereg_hash = comparison.sha(v4.PREREG)
    tcn_records = v4.completed_tcn_runs()
    verdict_path, verdict = load_pretraining_verdict()
    if verdict.get('verdict') != 'PRETRAIN_COLLAPSE' or verdict.get('checkpoint_written'):
        raise ValueError('masked pretraining audit does not match the frozen scratch branch')
    tctst_records = [load_tctst_run(seed, path, prereg_hash) for seed, path in TCTST_RUNS.items()]
    records = tcn_records + tctst_records
    if len(records) != 9 or sorted(r['seed'] for r in records if r['model'] == 'tctst_repaired') != [20261021, 20261022, 20261023]:
        raise ValueError('comparison must contain all nine frozen seed runs')
    if args.preflight_only:
        print(json.dumps({
            'preflight': 'PASS', 'tcn_runs': len(tcn_records), 'tctst_runs': len(tctst_records),
            'pretraining_verdict': verdict['verdict'],
            'selection_rule': 'highest mean completion across 504 consistent cycles; tie by common-frame MAE; final alphabetical tie',
        }, ensure_ascii=False, indent=2))
        return 0
    summary_path = OUT / 'comparison_summary.json'
    selection_path = OUT / 'selected_model.json'
    if summary_path.exists() or selection_path.exists():
        raise FileExistsError('refusing to overwrite v4 comparison or selection artifacts')
    summary = comparison.summarize_matched(records)
    summary.update({
        'schema': 'g1.phase_model_comparison.v4',
        'tctst_pretrained': False,
        'tctst_readout': 'mean over layerwise channel-pooled transformer outputs + sqrt(dim) scaled temporal-token channel mean',
        'masked_pretraining_verdict': verdict['verdict'],
        'reused_tcn_protocol': 'preregistration_amendment_v3.md',
        'current_protocol': str(v4.PREREG.relative_to(ROOT)),
        'training_records': records,
        'merge_script_sha256': comparison.sha(Path(__file__)),
    })
    comparison.write_json(summary_path, summary)
    selection = {
        'schema': 'g1.phase_model_selection.v2',
        'selected_family': summary['selected_family'],
        'selected_seed_for_control': 20261021,
        'selection_rule': summary['selection_rule'],
        'evidence_path': str(summary_path.relative_to(ROOT)),
        'requires_separate_control_comparison': True,
        'does_not_claim_deployment_or_freeze_eligibility': True,
        'protocol_sha256': prereg_hash,
    }
    comparison.write_json(selection_path, selection)
    print(json.dumps({
        'selected_family': selection['selected_family'],
        'selected_seed_for_control': selection['selected_seed_for_control'],
        'models': summary['models'], 'evidence_path': selection['evidence_path'],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
