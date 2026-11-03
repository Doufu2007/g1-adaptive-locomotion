"""Matched-budget TCN20 / TCN100-rate / repaired-TCTST comparison.

Only original train and validation arrays are loaded. All models use the same
consistent training endpoints and the same complete validation stream.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('PYTHONHASHSEED', '20261021')

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_g1_external_methods as base  # noqa: E402
from scripts.g1_external_models import phase_rate_output  # noqa: E402
from scripts.g1_tctst_faithful_v2 import ChannelIdentifiedTCTST  # noqa: E402
from scripts.run_g1_long_history_comparison import LongContextTCNEncoder  # noqa: E402

OUT = ROOT / 'results/g1_phase_model_comparison_20261023'
PREREG = OUT / 'preregistration_amendment_v3.md'
SEEDS = (20261021, 20261022, 20261023)
EPOCHS = 100
EFFECTIVE_BATCH = 512
MICRO = 128
LR = 6e-4
WARMUP_EPOCHS = 20
MODELS = ('tcn20', 'tcn100_rate', 'tctst_repaired')
CODE_PATHS = (
    Path(__file__), ROOT / 'scripts/diagnose_g1_tctst_pretraining_v2.py',
    ROOT / 'scripts/g1_tctst_faithful_v2.py', ROOT / 'scripts/g1_external_models.py',
    ROOT / 'scripts/run_g1_external_methods.py',
    ROOT / 'scripts/run_g1_long_history_comparison.py',
    ROOT / 'scripts/run_g1_contact_followup_comparison.py',
    ROOT / 'src/tasks/velocity/phasor/phase_gate_v4_2.py',
)
DATA_PATHS = (
    ROOT / 'results/g1_phase_dataset_20261019/phase_dataset.npz',
    ROOT / 'results/g1_phase_dataset_20261019/contact_consistency_filter_v1/phase_dataset_train_validation.npz',
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               allow_nan=False) + '\n', encoding='utf-8')


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)


def model_for(name: str) -> tuple[nn.Module, int]:
    if name == 'tcn20':
        return LongContextTCNEncoder(), 20
    if name == 'tcn100_rate':
        return base.LongRateTCN(), 100
    if name == 'tctst_repaired':
        return ChannelIdentifiedTCTST(), 100
    raise ValueError(name)


def lr_scale(epoch: int) -> float:
    if epoch >= WARMUP_EPOCHS:
        return 1.0
    return .2 + .8 * (1 - np.cos(np.pi * epoch / (WARMUP_EPOCHS - 1))) / 2


def initialize_tctst(model: ChannelIdentifiedTCTST, pretrained_path: Path | None) -> None:
    if pretrained_path is None:
        return
    checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=False)
    if checkpoint.get('verdict') != 'PRETRAIN_PASS':
        raise ValueError('pretraining checkpoint did not pass the frozen audit')
    state = checkpoint['model']
    fresh_head = model.head
    model.load_state_dict(state, strict=True)
    # The reconstruction stage never trains the phase head. Reinitialize it
    # from the supervised seed so all three runs start with a fresh head.
    model.head = fresh_head


def loss_for(output: torch.Tensor, bank, ids: torch.Tensor) -> torch.Tensor:
    phase_loss = (1.0 - (output[:, :2] * bank.target[ids]).sum(dim=-1)).mean()
    if output.shape[1] == 3:
        valid = bank.rate_mask[ids]
        phase_loss = phase_loss + .02 * (
            (output[:, 2] - bank.rate_tensor[ids]).square() * valid
        ).sum() / valid.sum().clamp_min(1)
    return phase_loss


def atomic_torch_save(value, path: Path) -> None:
    temp = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temp)
    temp.replace(path)


def predict(model: nn.Module, bank, window: int, micro: int) -> np.ndarray:
    model.eval()
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(bank.ends), micro):
            ids = torch.arange(start, min(start + micro, len(bank.ends)),
                               device=base.DEVICE)
            with base.amp():
                output = model(bank.windows(ids, window))
            chunks.append(output.float().cpu().numpy())
    return np.concatenate(chunks, axis=0)


def train_one(name: str, seed: int, train_bank, val_bank,
              pretrained_path: Path | None) -> dict:
    run_dir = OUT / 'matched_100ep' / f'{name}_seed_{seed}'
    run_dir.mkdir(parents=True, exist_ok=False)
    seed_all(seed)
    model, window = model_for(name)
    if name == 'tctst_repaired':
        initialize_tctst(model, pretrained_path)
    model = model.to(base.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    ids = train_bank.ids
    if len(ids) != 55820:
        raise ValueError(f'locked consistent train size changed: {len(ids)}')
    generator = torch.Generator().manual_seed(seed)
    history = []
    write_json(run_dir / 'manifest.json', {
        'schema': 'g1.phase_model_comparison_run.v2', 'status': 'RUNNING',
        'started_at': now(), 'model': name, 'seed': seed, 'window': window,
        'epochs': EPOCHS, 'effective_batch': EFFECTIVE_BATCH, 'microbatch': MICRO,
        'optimizer': 'Adam', 'learning_rate': LR,
        'lr_schedule': 'cosine warm-up 0.2 to 1.0 for 20 epochs; fixed thereafter',
        'training_endpoints': int(len(ids)), 'normalization': 'frozen training mean/std',
        'pretrained_checkpoint': str(pretrained_path) if name == 'tctst_repaired' and pretrained_path else None,
        'pretrained_checkpoint_sha256': sha(pretrained_path) if name == 'tctst_repaired' and pretrained_path else None,
        'preregistration_sha256': sha(PREREG),
        'code_hashes': {str(p.relative_to(ROOT)): sha(p) for p in CODE_PATHS},
        'data_hashes': {str(p.relative_to(ROOT)): sha(p) for p in DATA_PATHS},
    })
    try:
        for epoch in range(EPOCHS):
            started = time.perf_counter()
            model.train()
            order = ids[torch.randperm(len(ids), generator=generator).to(base.DEVICE)]
            actual_lr = LR * lr_scale(epoch)
            for group in optimizer.param_groups:
                group['lr'] = actual_lr
            loss_sum, seen = 0.0, 0
            for start in range(0, len(order), EFFECTIVE_BATCH):
                effective = order[start:start + EFFECTIVE_BATCH]
                optimizer.zero_grad(set_to_none=True)
                for small in range(0, len(effective), MICRO):
                    ii = effective[small:small + MICRO]
                    with base.amp():
                        output = model(train_bank.windows(ii, window)).float()
                        loss = loss_for(output, train_bank, ii)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f'nonfinite loss at epoch {epoch + 1}')
                    (loss * (len(ii) / len(effective))).backward()
                    loss_sum += float(loss.detach()) * len(ii)
                    seen += len(ii)
                norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0,
                                                error_if_nonfinite=True)
                optimizer.step()
            if base.DEVICE.type == 'cuda':
                torch.cuda.synchronize()
            row = {'epoch': epoch + 1, 'loss': loss_sum / seen, 'samples': seen,
                   'seconds': time.perf_counter() - started,
                   'grad_norm_last': float(norm), 'learning_rate': actual_lr}
            history.append(row)
            with (run_dir / 'epochs.jsonl').open('a', encoding='utf-8') as f:
                f.write(json.dumps(row) + '\n')
            print(json.dumps({'model': name, 'seed': seed, **row}), flush=True)
            if (epoch + 1) % 10 == 0:
                atomic_torch_save({'epoch': epoch + 1, 'model': model.state_dict(),
                                   'optimizer': optimizer.state_dict(),
                                   'torch_rng': torch.get_rng_state(),
                                   'cuda_rng': torch.cuda.get_rng_state_all() if base.DEVICE.type == 'cuda' else None,
                                   'generator_rng': generator.get_state()},
                                  run_dir / 'partial_checkpoint.pt')
        predictions = predict(model, val_bank, window, 64 if name == 'tctst_repaired' else 512)
        metrics = base.evaluate(predictions, val_bank, run_dir, 'direct', window)
        checkpoint = {
            'model_name': name, 'seed': seed, 'window': window,
            'model': model.state_dict(), 'normalization_mean': train_bank.mean,
            'normalization_std': train_bank.std, 'pretrained_checkpoint_sha256':
                sha(pretrained_path) if name == 'tctst_repaired' and pretrained_path else None,
        }
        atomic_torch_save(checkpoint, run_dir / 'final_checkpoint.pt')
        result = {
            'schema': 'g1.phase_model_comparison_run.v2', 'status': 'COMPLETED',
            'ended_at': now(), 'model': name, 'seed': seed, 'window': window,
            'metrics': metrics, 'final_training_loss': history[-1]['loss'],
            'epochs': len(history), 'checkpoint_sha256': sha(run_dir / 'final_checkpoint.pt'),
            'predictions_sha256': sha(run_dir / 'direct_predictions.npy'),
            'pretrained_checkpoint_sha256': sha(pretrained_path)
                if name == 'tctst_repaired' and pretrained_path else None,
            'preregistration_sha256': sha(PREREG),
        }
        write_json(run_dir / 'completed.json', result)
        write_json(run_dir / 'manifest.json', result)
        del model, optimizer
        if base.DEVICE.type == 'cuda':
            torch.cuda.empty_cache()
        return result
    except BaseException as exc:
        write_json(run_dir / 'failure.json', {'type': type(exc).__name__, 'message': str(exc)})
        raise


def reevaluate_historical(train_bank, val_bank) -> list[dict]:
    """Re-score fixed historical TCN checkpoints on the common complete stream."""
    sweep = ROOT / 'results/g1_contact_followup_20261022/long_history_v1_replay_fixed'
    external = ROOT / 'results/g1_external_methods_trials_20261022/run_v2'
    out = OUT / 'historical_common_stream_v2'
    out.mkdir(parents=True, exist_ok=False)
    records = []
    for seed in SEEDS:
        name = f'tcn_window20_seed{seed}'
        path = sweep / f'{name}.pt'
        model = LongContextTCNEncoder().to(base.DEVICE)
        model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=True)
        pred = predict(model, val_bank, 20, 512)
        run_dir = out / name
        run_dir.mkdir()
        metrics = base.evaluate(pred, val_bank, run_dir, 'direct', 20)
        records.append({'model': 'tcn20', 'seed': seed, 'budget_epochs': 8,
                        'source_checkpoint': str(path.relative_to(ROOT)),
                        'source_checkpoint_sha256': sha(path), 'metrics': metrics})
        del model
        if base.DEVICE.type == 'cuda': torch.cuda.empty_cache()

    for seed in SEEDS:
        name = f'tcn100_rate_seed_{seed}'
        completed_path = external / name / 'completed.json'
        completed = json.loads(completed_path.read_text(encoding='utf-8'))
        attempt = Path(completed['attempt'].replace('\\', '/'))
        if not attempt.is_absolute():
            attempt = ROOT / attempt
        run_dir_src = attempt
        pred_path = run_dir_src / 'direct_predictions.npy'
        pred = np.load(pred_path, allow_pickle=False)
        if pred.shape != (45236, 3):
            raise ValueError(f'unexpected historical TCN100 prediction shape: {pred.shape}')
        run_dir = out / name
        run_dir.mkdir()
        metrics = base.evaluate(pred, val_bank, run_dir, 'direct', 100)
        records.append({'model': 'tcn100_rate', 'seed': seed, 'budget_epochs': 32,
                        'source_predictions': str(pred_path.relative_to(ROOT)),
                        'source_predictions_sha256': sha(pred_path),
                        'source_completed_manifest': str(completed_path.relative_to(ROOT)),
                        'source_protocol_sha256': completed['protocol_sha256'],
                        'metrics': metrics})

    write_json(out / 'reevaluation.json', {
        'schema': 'g1.phase_model_historical_common_stream.v1',
        'validation_rows': int(len(val_bank.ends)),
        'consistent_rows': int(val_bank.consistent.sum()),
        'all_complete_cycles': 603, 'consistent_cycles': 504, 'mismatch_cycles': 99,
        'preregistration_sha256': sha(PREREG),
        'code_hashes': {str(p.relative_to(ROOT)): sha(p) for p in CODE_PATHS},
        'data_hashes': {str(p.relative_to(ROOT)): sha(p) for p in DATA_PATHS},
        'records': records,
    })
    return records


def summarize_matched(records: list[dict]) -> dict:
    by_model = {}
    for name in MODELS:
        rr = [r for r in records if r['model'] == name]
        clean_gate = [r['metrics']['cycles']['consistency']['macro_completion'] for r in rr]
        clean_mae = [r['metrics']['point']['consistency']['macro_mae_rad'] for r in rr]
        all_gate = [r['metrics']['cycles']['all']['macro_completion'] for r in rr]
        by_model[name] = {
            'n_seeds': len(rr), 'seeds': [r['seed'] for r in rr],
            'consistent_cycle_macro_mean': float(np.mean(clean_gate)),
            'consistent_cycle_macro_sd': float(np.std(clean_gate, ddof=1)),
            'consistent_frame_macro_mae_mean_rad': float(np.mean(clean_mae)),
            'consistent_frame_macro_mae_sd_rad': float(np.std(clean_mae, ddof=1)),
            'all_603_cycle_macro_mean': float(np.mean(all_gate)),
            'per_seed': [{'seed': r['seed'],
                          'consistent_cycle_macro': r['metrics']['cycles']['consistency']['macro_completion'],
                          'consistent_frame_macro_mae_rad': r['metrics']['point']['consistency']['macro_mae_rad'],
                          'all_603_cycle_macro': r['metrics']['cycles']['all']['macro_completion'],
                          'official_full_freeze_eligible': r['metrics']['official_full_freeze_eligible']}
                         for r in rr],
        }
    chosen = min(MODELS, key=lambda n: (
        -by_model[n]['consistent_cycle_macro_mean'],
        by_model[n]['consistent_frame_macro_mae_mean_rad'], n))
    return {'models': by_model, 'selection_rule':
            'highest mean 504-cycle consistency-subset macro completion; ties use lower common-frame macro MAE; final tie alphabetical',
            'selected_family': chosen,
            'selected_seed_for_control': 20261021}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--historical-only', action='store_true')
    parser.add_argument('--train-matched', action='store_true')
    parser.add_argument('--pretrained', type=Path)
    parser.add_argument('--scratch-tctst', action='store_true')
    args = parser.parse_args()
    if args.historical_only == args.train_matched:
        parser.error('select exactly one of --historical-only or --train-matched')
    base.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.train_matched and not args.scratch_tctst:
        if args.pretrained is None or not args.pretrained.exists():
            parser.error('--train-matched requires a passed --pretrained checkpoint')
    if args.train_matched and args.scratch_tctst and args.pretrained is not None:
        parser.error('--scratch-tctst and --pretrained cannot be combined')
    train_bank = base.FrameBank('train')
    val_bank = base.FrameBank('validation')
    if args.historical_only:
        reevaluate_historical(train_bank, val_bank)
        return 0
    pretrained = args.pretrained
    records = []
    for name in MODELS:
        for seed in SEEDS:
            current_pretrained = pretrained if name == 'tctst_repaired' else None
            records.append(train_one(name, seed, train_bank, val_bank,
                                     current_pretrained))
    summary = summarize_matched(records)
    summary['runs'] = records
    summary['pretrained_tctst'] = bool(pretrained)
    write_json(OUT / 'matched_100ep' / 'comparison_summary.json', summary)
    write_json(OUT / 'selected_model.json', {
        'schema': 'g1.phase_model_selection.v1',
        'selected_family': summary['selected_family'],
        'selected_seed_for_control': summary['selected_seed_for_control'],
        'selection_rule': summary['selection_rule'],
        'evidence_path': 'matched_100ep/comparison_summary.json',
        'requires_separate_control_comparison': True,
        'does_not_claim_deployment_or_freeze_eligibility': True,
    })
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
