"""Resume or run a preregistered TCTST v4 seed without changing its recipe.

Recovery is isolated from the failed attempt. It restores the last complete
optimizer and RNG checkpoint, and records any OOM rewind before retrying.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import sys
import time

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_g1_phase_model_comparison_v2 as comparison  # noqa: E402
from scripts import run_g1_external_methods as base  # noqa: E402


def configure_protocol() -> None:
    comparison.OUT = ROOT / 'results/g1_phase_model_comparison_20261023/tctst_residual_v4'
    comparison.PREREG = ROOT / 'results/g1_phase_model_comparison_20261023/preregistration_amendment_v4.md'
    comparison.CODE_PATHS = (
        ROOT / 'scripts/run_g1_tctst_residual_v4_comparison.py',
        ROOT / 'scripts/run_g1_phase_model_comparison_v2.py',
        ROOT / 'scripts/g1_tctst_faithful_v2.py',
        ROOT / 'scripts/g1_external_models.py',
        ROOT / 'scripts/run_g1_external_methods.py',
        ROOT / 'scripts/run_g1_long_history_comparison.py',
        ROOT / 'src/tasks/velocity/phasor/encoder_tctst_runtime.py',
        ROOT / 'src/tasks/velocity/mdp/observations.py',
        ROOT / 'scripts/prepare_g1_selected_model_runtime_v2.py',
        ROOT / 'tests/test_g1_tctst_readout.py',
        ROOT / 'tests/test_g1_long_window_model_runtimes.py',
        ROOT / 'src/tasks/velocity/phasor/phase_gate_v4_2.py',
    )
    comparison.DATA_PATHS = (
        ROOT / 'results/g1_phase_dataset_20261019/phase_dataset.npz',
        ROOT / 'results/g1_phase_dataset_20261019/contact_consistency_filter_v1/phase_dataset_train_validation.npz',
    )


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def save_partial(model, optimizer, generator, epoch: int, path: Path) -> None:
    comparison.atomic_torch_save({
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'torch_rng': torch.get_rng_state(),
        'cuda_rng': torch.cuda.get_rng_state_all() if base.DEVICE.type == 'cuda' else None,
        'generator_rng': generator.get_state(),
    }, path)


def restore_partial(model, optimizer, generator, path: Path) -> int:
    state = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(state['model'], strict=True)
    optimizer.load_state_dict(state['optimizer'])
    torch.set_rng_state(state['torch_rng'])
    if base.DEVICE.type == 'cuda':
        if not state['cuda_rng']:
            raise ValueError('CUDA recovery checkpoint lacks CUDA RNG state')
        torch.cuda.set_rng_state_all(state['cuda_rng'])
    generator.set_state(state['generator_rng'])
    return int(state['epoch'])


def truncate_epoch_log(path: Path, through_epoch: int) -> list[dict]:
    rows = read_jsonl(path)
    discarded = [row for row in rows if int(row['epoch']) > through_epoch]
    kept = [row for row in rows if int(row['epoch']) <= through_epoch]
    content = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in kept)
    path.write_text(content, encoding='utf-8')
    return discarded


def load_source_prefix(source_checkpoint: Path, seed: int) -> tuple[dict, list[dict]]:
    source_checkpoint = source_checkpoint.resolve(strict=True)
    source_run = source_checkpoint.parent
    failure_path = source_run / 'failure.json'
    source_manifest = json.loads((source_run / 'manifest.json').read_text(encoding='utf-8'))
    failure = json.loads(failure_path.read_text(encoding='utf-8'))
    if failure.get('type') != 'RuntimeError' or 'out of memory' not in failure.get('message', '').lower():
        raise ValueError('source attempt is not the recorded CUDA OOM failure')
    if source_manifest.get('model') != 'tctst_repaired' or int(source_manifest.get('seed', -1)) != seed:
        raise ValueError('source attempt model/seed does not match the requested recovery')
    checkpoint = torch.load(source_checkpoint, map_location='cpu', weights_only=False)
    resume_epoch = int(checkpoint['epoch'])
    if not 0 < resume_epoch < comparison.EPOCHS:
        raise ValueError(f'invalid recovery epoch {resume_epoch}')
    prefix = read_jsonl(source_run / 'epochs.jsonl')
    complete_prefix = [row for row in prefix if int(row['epoch']) <= resume_epoch]
    if len(complete_prefix) != resume_epoch or [int(r['epoch']) for r in complete_prefix] != list(range(1, resume_epoch + 1)):
        raise ValueError('source checkpoint does not have a complete matching epoch log prefix')
    if comparison.sha(source_run / 'partial_checkpoint.pt') != comparison.sha(source_checkpoint):
        raise ValueError('source checkpoint hash changed during recovery preflight')
    return {
        'path': source_checkpoint,
        'source_run': source_run,
        'failure_path': failure_path,
        'failure_sha256': comparison.sha(failure_path),
        'manifest_sha256': comparison.sha(source_run / 'manifest.json'),
        'checkpoint_sha256': comparison.sha(source_checkpoint),
        'checkpoint': checkpoint,
        'resume_epoch': resume_epoch,
        'prefix': complete_prefix,
        'observed_failed_attempt_epochs': prefix,
    }, complete_prefix


def append_event(path: Path, event: dict) -> None:
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(event, ensure_ascii=False) + '\n')


def clear_cuda_oom_state() -> list[str]:
    """Clear recoverable CUDA OOM state while preserving unrelated failures."""
    cleanup_errors = []
    for name, operation in (
        ('synchronize', torch.cuda.synchronize),
        ('empty_cache', torch.cuda.empty_cache),
    ):
        try:
            operation()
        except RuntimeError as error:
            if 'out of memory' not in str(error).lower():
                raise
            cleanup_errors.append(f'{name}: {error}')
    return cleanup_errors


def run_seed(seed: int, run_dir: Path, resume_from: Path | None, preflight: bool) -> int:
    configure_protocol()
    comparison.base.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if resume_from is not None and comparison.base.DEVICE.type != 'cuda':
        raise RuntimeError('frozen exact-resume path requires the original CUDA device')
    run_dir = run_dir.resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f'recovery run directory is not empty: {run_dir}')

    source = None
    prefix: list[dict] = []
    if resume_from is not None:
        source, prefix = load_source_prefix(resume_from, seed)
    train_bank = comparison.base.FrameBank('train')
    val_bank = comparison.base.FrameBank('validation')
    observed_counts = {
        'train_rows': len(train_bank.ends),
        'train_consistent_rows': int(train_bank.consistent.sum()),
        'validation_rows': len(val_bank.ends),
        'validation_consistent_rows': int(val_bank.consistent.sum()),
    }
    if observed_counts != {
        'train_rows': 73226, 'train_consistent_rows': 55820,
        'validation_rows': 45236, 'validation_consistent_rows': 32878,
    }:
        raise ValueError(f'frozen train/validation endpoint counts changed: {observed_counts}')
    if preflight:
        free = total = None
        if comparison.base.DEVICE.type == 'cuda':
            free, total = torch.cuda.mem_get_info()
        print(json.dumps({
            'preflight': 'PASS', 'seed': seed, 'device': str(comparison.base.DEVICE),
            'run_dir': str(run_dir), 'resume_epoch': source['resume_epoch'] if source else 0,
            'source_checkpoint_sha256': source['checkpoint_sha256'] if source else None,
            'cuda_free_bytes': free, 'cuda_total_bytes': total,
            'training_endpoints': 55820, 'epochs': comparison.EPOCHS,
            'effective_batch': comparison.EFFECTIVE_BATCH, 'microbatch': comparison.MICRO,
            'observed_counts': observed_counts,
        }, ensure_ascii=False, indent=2))
        return 0

    run_dir.mkdir(parents=True, exist_ok=False)
    comparison.seed_all(seed)
    model, window = comparison.model_for('tctst_repaired')
    comparison.initialize_tctst(model, None)
    model = model.to(comparison.base.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=comparison.LR)
    generator = torch.Generator().manual_seed(seed)
    epoch_log = run_dir / 'epochs.jsonl'
    event_log = run_dir / 'recovery_events.jsonl'
    if source is not None:
        shutil.copy2(source['path'], run_dir / 'partial_checkpoint.pt')
        epoch_log.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in prefix), encoding='utf-8')
        start_epoch = restore_partial(model, optimizer, generator, run_dir / 'partial_checkpoint.pt')
        if start_epoch != source['resume_epoch']:
            raise ValueError('restored epoch differs from the sealed source checkpoint')
    else:
        start_epoch = 0
        save_partial(model, optimizer, generator, start_epoch, run_dir / 'partial_checkpoint.pt')

    manifest = {
        'schema': 'g1.phase_model_comparison_run.v2', 'status': 'RUNNING',
        'started_at': comparison.now(), 'model': 'tctst_repaired', 'seed': seed,
        'window': window, 'epochs': comparison.EPOCHS,
        'effective_batch': comparison.EFFECTIVE_BATCH, 'microbatch': comparison.MICRO,
        'optimizer': 'Adam', 'learning_rate': comparison.LR,
        'lr_schedule': 'cosine warm-up 0.2 to 1.0 for 20 epochs; fixed thereafter',
        'training_endpoints': 55820, 'normalization': 'frozen training mean/std',
        'pretrained_checkpoint': None, 'pretrained_checkpoint_sha256': None,
        'preregistration_sha256': comparison.sha(comparison.PREREG),
        'code_hashes': {str(p.relative_to(ROOT)): comparison.sha(p) for p in comparison.CODE_PATHS},
        'data_hashes': {str(p.relative_to(ROOT)): comparison.sha(p) for p in comparison.DATA_PATHS},
        'recovery_script_sha256': comparison.sha(Path(__file__)),
        'recovery': None if source is None else {
            'source_failed_run': str(source['source_run'].relative_to(ROOT)),
            'source_failure_sha256': source['failure_sha256'],
            'source_manifest_sha256': source['manifest_sha256'],
            'source_checkpoint_sha256': source['checkpoint_sha256'],
            'resumed_from_epoch': start_epoch,
            'resume_preserved_model_optimizer_torch_cpu_cuda_and_order_rng': True,
            'recipe_changed': False,
        },
        'device': str(comparison.base.DEVICE), 'torch_version': torch.__version__,
        'cuda_device': torch.cuda.get_device_name() if comparison.base.DEVICE.type == 'cuda' else None,
        'source_protocol_manifest': str((source['source_run'] / 'manifest.json').relative_to(ROOT)) if source else None,
    }
    comparison.write_json(run_dir / 'manifest.json', manifest)

    attempts = 0
    latest_saved_epoch = start_epoch
    retry_budget = 5
    while start_epoch < comparison.EPOCHS:
        epoch = start_epoch
        started = time.perf_counter()
        model.train()
        order = train_bank.ids[torch.randperm(len(train_bank.ids), generator=generator).to(comparison.base.DEVICE)]
        actual_lr = comparison.LR * comparison.lr_scale(epoch)
        for group in optimizer.param_groups:
            group['lr'] = actual_lr
        loss_sum, seen = 0.0, 0
        try:
            for batch_start in range(0, len(order), comparison.EFFECTIVE_BATCH):
                effective = order[batch_start:batch_start + comparison.EFFECTIVE_BATCH]
                optimizer.zero_grad(set_to_none=True)
                for small_start in range(0, len(effective), comparison.MICRO):
                    ids = effective[small_start:small_start + comparison.MICRO]
                    with comparison.base.amp():
                        output = model(train_bank.windows(ids, window)).float()
                        loss = comparison.loss_for(output, train_bank, ids)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f'nonfinite loss at epoch {epoch + 1}')
                    (loss * (len(ids) / len(effective))).backward()
                    loss_sum += float(loss.detach()) * len(ids)
                    seen += len(ids)
                norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
            if comparison.base.DEVICE.type == 'cuda':
                torch.cuda.synchronize()
        except RuntimeError as error:
            if 'out of memory' not in str(error).lower() or comparison.base.DEVICE.type != 'cuda':
                raise
            error_message = str(error)
            attempts += 1
            event = {
                'event': 'CUDA_OOM_REWIND', 'at_epoch': epoch + 1,
                'retry': attempts, 'last_complete_checkpoint_epoch': latest_saved_epoch,
                'message': error_message, 'timestamp': comparison.now(),
            }
            append_event(event_log, event)
            if attempts > retry_budget:
                comparison.write_json(run_dir / 'failure.json', {
                    'type': 'CUDA_OOM_RETRY_EXHAUSTED', 'attempts': attempts,
                    'last_complete_epoch': latest_saved_epoch, 'message': error_message,
                })
                manifest.update(status='FAILED_RECOVERABLE', ended_at=comparison.now(),
                                failure='CUDA_OOM_RETRY_EXHAUSTED',
                                last_complete_epoch=latest_saved_epoch)
                comparison.write_json(run_dir / 'manifest.json', manifest)
                raise
            error.__traceback__ = None
            try:
                del loss
            except UnboundLocalError:
                pass
            try:
                del output
            except UnboundLocalError:
                pass
            try:
                del ids
            except UnboundLocalError:
                pass
            try:
                del effective
            except UnboundLocalError:
                pass
            try:
                del order
            except UnboundLocalError:
                pass
            try:
                del norm
            except UnboundLocalError:
                pass
            optimizer.zero_grad(set_to_none=True)
            gc.collect()
            cleanup_errors = clear_cuda_oom_state()
            restored_epoch = restore_partial(model, optimizer, generator, run_dir / 'partial_checkpoint.pt')
            discarded = truncate_epoch_log(epoch_log, restored_epoch)
            append_event(event_log, {
                'event': 'CHECKPOINT_RESTORED', 'checkpoint_epoch': restored_epoch,
                'discarded_logged_epochs': discarded, 'cleanup_errors': cleanup_errors,
                'timestamp': comparison.now(),
            })
            start_epoch = restored_epoch
            print(json.dumps({
                'status': 'OOM_RECOVERY', 'seed': seed, 'rewind_to_epoch': restored_epoch,
                'retry': attempts, 'wait_seconds': 30,
            }), flush=True)
            time.sleep(30)
            continue

        row = {'epoch': epoch + 1, 'loss': loss_sum / seen, 'samples': seen,
               'seconds': time.perf_counter() - started, 'grad_norm_last': float(norm),
               'learning_rate': actual_lr}
        with epoch_log.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row) + '\n')
        print(json.dumps({'model': 'tctst_repaired', 'seed': seed, **row}), flush=True)
        start_epoch = epoch + 1
        if start_epoch % 10 == 0:
            save_partial(model, optimizer, generator, start_epoch, run_dir / 'partial_checkpoint.pt')
            latest_saved_epoch = start_epoch
            attempts = 0

    replay_losses = {}
    if source is not None:
        old = {int(row['epoch']): row for row in source['observed_failed_attempt_epochs']}
        new = {int(row['epoch']): row for row in read_jsonl(epoch_log)}
        for epoch in (11, 12):
            if epoch in old and epoch in new:
                replay_losses[str(epoch)] = {
                    'source_loss': old[epoch]['loss'], 'replayed_loss': new[epoch]['loss'],
                    'absolute_difference': abs(old[epoch]['loss'] - new[epoch]['loss']),
                    'within_1e-8': abs(old[epoch]['loss'] - new[epoch]['loss']) <= 1e-8,
                }

    predictions = comparison.predict(model, val_bank, window, 64)
    metrics = comparison.base.evaluate(predictions, val_bank, run_dir, 'direct', window)
    checkpoint = {
        'model_name': 'tctst_repaired', 'seed': seed, 'window': window,
        'model': model.state_dict(), 'normalization_mean': train_bank.mean,
        'normalization_std': train_bank.std, 'pretrained_checkpoint_sha256': None,
    }
    comparison.atomic_torch_save(checkpoint, run_dir / 'final_checkpoint.pt')
    completed = {
        'schema': 'g1.phase_model_comparison_run.v2', 'status': 'COMPLETED',
        'ended_at': comparison.now(), 'model': 'tctst_repaired', 'seed': seed,
        'window': window, 'metrics': metrics, 'final_training_loss': read_jsonl(epoch_log)[-1]['loss'],
        'epochs': len(read_jsonl(epoch_log)),
        'checkpoint_sha256': comparison.sha(run_dir / 'final_checkpoint.pt'),
        'predictions_sha256': comparison.sha(run_dir / 'direct_predictions.npy'),
        'pretrained_checkpoint_sha256': None, 'preregistration_sha256': comparison.sha(comparison.PREREG),
        'recovery_script_sha256': comparison.sha(Path(__file__)),
        'recovery': manifest['recovery'], 'replayed_source_losses': replay_losses,
        'recovery_events_sha256': comparison.sha(event_log) if event_log.exists() else None,
    }
    comparison.write_json(run_dir / 'completed.json', completed)
    comparison.write_json(run_dir / 'manifest.json', {**manifest, **completed})
    print(json.dumps({
        'status': 'COMPLETED', 'seed': seed, 'run_dir': str(run_dir),
        'checkpoint_sha256': completed['checkpoint_sha256'],
        'consistent_cycle_macro_completion': metrics['cycles']['consistency']['macro_completion'],
        'consistent_frame_macro_mae_rad': metrics['point']['consistency']['macro_mae_rad'],
        'all_603_cycle_macro_completion': metrics['cycles']['all']['macro_completion'],
        'official_full_freeze_eligible': metrics['official_full_freeze_eligible'],
        'replayed_source_losses': replay_losses,
    }, ensure_ascii=False, indent=2), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, required=True, choices=(20261021, 20261022, 20261023))
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--resume-from', type=Path)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    if args.seed == 20261022 and args.resume_from is None:
        parser.error('seed 20261022 recovery requires --resume-from pointing to the failed run checkpoint')
    if args.seed != 20261022 and args.resume_from is not None:
        parser.error('--resume-from is reserved for the documented seed 20261022 OOM recovery')
    return run_seed(args.seed, args.run_dir, args.resume_from, args.preflight_only)


if __name__ == '__main__':
    raise SystemExit(main())
