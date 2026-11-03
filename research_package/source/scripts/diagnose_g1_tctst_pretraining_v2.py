"""Sequence-held-out masked-pretraining audit for the TCTST comparison.

Uses one entire training recording for audit and excludes every endpoint from
that recording from masked-pretraining updates. The same fixed 10/29 masks are
used for audit and both baselines. No phase labels or validation/test arrays
are read.
"""

from __future__ import annotations

import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('PYTHONHASHSEED', '20261021')

import argparse
import contextlib
import json
import math
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_g1_external_methods as base  # noqa: E402
from scripts.g1_tctst_faithful_v2 import (  # noqa: E402
    JOINT_CHANNELS, MASK_COUNT, ChannelIdentifiedTCTST, MaskedPretrainer,
    masked_reconstruction_loss,
)

PREREG = ROOT / 'results/g1_phase_model_comparison_20261023/preregistration_amendment_v3.md'
DEFAULT_OUT = ROOT / 'results/g1_phase_model_comparison_20261023/pretrain'
SMOKE_OUT = ROOT / 'results/g1_phase_model_comparison_20261023/smoke'

# Pre-declared thresholds.  Do not edit without a new preregistration.
PRETRAIN_PASS_FRACTION = .90
S5_COLLAPSE_SD = .2   # amendment 1: see preregistration


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
                    encoding='utf-8')


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for f in self.streams:
            f.write(s)
            f.flush()

    def flush(self):
        for f in self.streams:
            f.flush()


# ── trivial baselines, computed once before training ────────────────────────

def losses_on(joint: torch.Tensor, mask: torch.Tensor, prediction: torch.Tensor) -> float:
    selected = mask[..., None]
    return float(((prediction - joint).square() * selected).sum()
                 / (mask.sum() * joint.shape[-1]).clamp_min(1))


def visible_mean_baseline(joint: torch.Tensor, mask: torch.Tensor) -> float:
    visible = (~mask)[..., None]
    mean = (joint * visible).sum(1) / visible.sum(1).clamp_min(1)
    return losses_on(joint, mask, mean[:, None, :].expand_as(joint))


def ridge_baseline(joint: torch.Tensor) -> float:
    """Per-timestep ridge from the other 28 joints.

    Evaluated over every entry and averaged equally across channels.  With a
    uniform 10-of-29 mask that is an unbiased estimate of the masked-entry
    loss the pretrainer is scored on.
    """
    x = joint.permute(0, 2, 1).reshape(-1, JOINT_CHANNELS).double()
    half = len(x) // 2
    fit, evaluate = x[:half], x[half:]
    errors = []
    eye = torch.eye(JOINT_CHANNELS + 1, dtype=x.dtype, device=x.device)
    for channel in range(JOINT_CHANNELS):
        features_fit = fit.clone()
        features_fit[:, channel] = 0
        a_fit = torch.cat((features_fit, torch.ones(len(fit), 1, dtype=x.dtype, device=x.device)), 1)
        weights = torch.linalg.solve(a_fit.T @ a_fit + 1e-2 * eye, a_fit.T @ fit[:, channel])
        features_eval = evaluate.clone()
        features_eval[:, channel] = 0
        a_eval = torch.cat((features_eval, torch.ones(len(evaluate), 1, dtype=x.dtype,
                                                      device=x.device)), 1)
        errors.append((a_eval @ weights - evaluate[:, channel]).square().mean())
    return float(torch.stack(errors).mean())


def feature_stats(model: ChannelIdentifiedTCTST, joint: torch.Tensor) -> dict:
    """Cross-sample variability and effective rank of the channel tokens."""
    model.eval()
    with torch.no_grad():
        tokens = model.encode_tokens(joint).float()
    flat = tokens.reshape(len(tokens), -1)
    flat = flat - flat.mean(0, keepdim=True)
    gram = flat @ flat.T
    values = torch.linalg.eigvalsh(gram).clamp_min(0)
    total = values.sum().clamp_min(1e-12)
    share = values / total
    effective = torch.exp(-(share * share.clamp_min(1e-12).log()).sum())
    return {
        'across_sample_sd': float(flat.std(0).mean()),
        'effective_rank': float(effective),
        'effective_rank_ratio': float(effective / min(flat.shape)),
        'mean_token_norm': float(tokens.norm(dim=-1).mean()),
    }


@torch.no_grad()
def ridge_readout(model: ChannelIdentifiedTCTST, bank, ids) -> dict:
    """Closed-form linear readout from frozen features to end-point sin/cos."""
    model.eval()
    features, targets = [], []
    for start in range(0, len(ids), 512):
        ii = ids[start:start + 512]
        joint, _ = bank.pretraining_joint(ii, model.window)
        features.append(model.encode_tokens(joint[:, :JOINT_CHANNELS]).mean(1).float())
        targets.append(bank.target[ii].float())
    x, y = torch.cat(features), torch.cat(targets)
    half = len(x) // 2
    fit, evaluate = x[:half], x[half:]
    target_fit, target_eval = y[:half], y[half:]
    mean = fit.mean(0)
    scale = fit.std(0).clamp_min(1e-6)
    a_fit = torch.cat(((fit - mean) / scale, torch.ones(len(fit), 1, device=x.device)), 1)
    a_eval = torch.cat(((evaluate - mean) / scale, torch.ones(len(evaluate), 1, device=x.device)), 1)
    identity = torch.eye(a_fit.shape[1], device=x.device)
    weights = torch.linalg.solve(a_fit.T @ a_fit + 1e-2 * identity, a_fit.T @ target_fit)
    prediction = a_eval @ weights
    residual = (prediction - target_eval).square().sum(0)
    total = (target_eval - target_eval.mean(0)).square().sum(0)
    theta = torch.atan2(prediction[:, 0], prediction[:, 1])
    truth = torch.atan2(target_eval[:, 0], target_eval[:, 1])
    error = torch.atan2(torch.sin(theta - truth), torch.cos(theta - truth)).abs().mean()
    return {'r2': (1 - residual / total).tolist(), 'ridge_mae_rad': float(error),
            'n_features': int(x.shape[1]), 'n_samples': int(len(x))}


def spread(values: torch.Tensor, count: int) -> torch.Tensor:
    """Evenly spaced subsample so the audit batch spans the full recording."""
    if len(values) <= count:
        return values
    positions = torch.linspace(0, len(values) - 1, count).round().long()
    return values[positions]


def sample_mask(count: int, generator: torch.Generator) -> torch.Tensor:
    mask = torch.zeros(count, JOINT_CHANNELS, dtype=torch.bool, device=base.DEVICE)
    mask.scatter_(1, torch.rand(count, JOINT_CHANNELS, generator=generator,
                                device=base.DEVICE).topk(MASK_COUNT, dim=1).indices, True)
    return mask


def warmup_scale(epoch: int) -> float:
    return .2 + .8 * (1 - math.cos(math.pi * min(epoch / 19., 1.))) / 2


def audit_loss_batched(pretrainer, joint: torch.Tensor, mask: torch.Tensor,
                       micro: int) -> float:
    total, count = 0.0, 0
    for start in range(0, len(joint), micro):
        x = joint[start:start + micro]
        m = mask[start:start + micro]
        with base.amp():
            prediction = pretrainer(x, m)
            loss = masked_reconstruction_loss(prediction.float(), x, m)
        total += float(loss) * len(x)
        count += len(x)
    return total / max(count, 1)


def train(args, bank, model, pretrainer, train_ids, audit_joint, audit_mask, run_dir):
    params = list(pretrainer.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)
    order_generator = torch.Generator().manual_seed(args.seed)
    mask_generator = torch.Generator(device=base.DEVICE).manual_seed(args.seed + 1)
    order_all = train_ids[::5]
    history = []
    stop_reason = None
    for epoch in range(args.epochs):
        started = time.perf_counter()
        model.train(); pretrainer.train()
        actual_lr = args.lr * warmup_scale(epoch)
        for group in optimizer.param_groups:
            group['lr'] = actual_lr
        order = order_all[torch.randperm(len(order_all), generator=order_generator).to(base.DEVICE)]
        loss_sum, seen, step = 0., 0, 0
        for start in range(0, len(order), 512):
            effective = order[start:start + 512]
            optimizer.zero_grad(set_to_none=True)
            for small in range(0, len(effective), args.micro):
                ii = effective[small:small + args.micro]
                joint, _ = bank.pretraining_joint(ii, model.window)
                joint = joint[:, :JOINT_CHANNELS]
                mask = sample_mask(len(ii), mask_generator)
                with base.amp():
                    prediction = pretrainer(joint, mask)
                    loss = masked_reconstruction_loss(prediction.float(), joint, mask)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'nonfinite loss at epoch {epoch}')
                (loss * (len(ii) / len(effective))).backward()
                loss_sum += float(loss.detach()) * len(ii)
                seen += len(ii)
                step += 1
            norm = nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
            optimizer.step()
        if base.DEVICE.type == 'cuda':
            torch.cuda.synchronize()
        model.eval(); pretrainer.eval()
        with torch.no_grad():
            audit = audit_loss_batched(pretrainer, audit_joint, audit_mask, args.micro)
        row = {'epoch': epoch + 1, 'train_loss': loss_sum / seen, 'audit_loss': audit,
               'seconds': time.perf_counter() - started, 'grad_norm_last': float(norm),
               'learning_rate': actual_lr, 'masked_entries_per_sample': MASK_COUNT}
        history.append(row)
        with (run_dir / 'epochs.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)

    return history, stop_reason


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=20261021)
    parser.add_argument('--lr', type=float, default=6e-4)
    parser.add_argument('--micro', type=int, default=128)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--audit-sequence', default='walk3_subject5')
    parser.add_argument('--smoke', action='store_true',
                        help='tiny CPU run that exercises the code path; never a verdict')
    args = parser.parse_args()

    if args.smoke:
        args.out = args.out or SMOKE_OUT
        args.epochs, args.micro = 2, 32
        args.device = 'cpu'
    else:
        args.out = args.out or DEFAULT_OUT
    args.out = args.out.resolve()
    if args.out.exists():
        raise FileExistsError(f'refusing to overwrite {args.out}')
    args.out.mkdir(parents=True)

    sys.stdout.reconfigure(encoding='utf-8')
    base.DEVICE = torch.device(args.device)
    with (args.out / 'stdout.log').open('w', encoding='utf-8') as log:
        with contextlib.redirect_stdout(Tee(sys.stdout, log)):
            return run(args)


def run(args) -> int:
    torch.manual_seed(args.seed)
    if not args.smoke:
        base.seed_all(args.seed)
    else:
        torch.use_deterministic_algorithms(False)  # smoke only, never a verdict

    manifest = {
        'schema': 'g1.tctst_pretrain_diagnosis.v1',
        'started_at': now(),
        'argv': sys.argv,
        'smoke': args.smoke,
        'seed': args.seed,
        'epochs': args.epochs,
        'lr': args.lr,
        'microbatch': args.micro,
        'device': str(base.DEVICE),
        'mask_count': MASK_COUNT,
        'joint_channels': JOINT_CHANNELS,
        'preregistration_sha256': base.digest(PREREG),
        'code_hashes': {str(p.relative_to(ROOT)): base.digest(p) for p in (
            Path(__file__), ROOT / 'scripts/g1_tctst_faithful_v2.py',
            ROOT / 'scripts/g1_external_models.py',
            ROOT / 'scripts/run_g1_external_methods.py')},
        'source_hashes': {str(p.relative_to(ROOT)): base.digest(p) for p in (
            base.SOURCE, base.FILTERED)},
        'environment': {'python': sys.version, 'torch': torch.__version__,
                        'numpy': base.np.__version__, 'os': platform.platform()},
    }
    write_json(args.out / 'manifest.json', manifest)

    bank = base.FrameBank('train')
    if len(bank.ids) != 55820:
        raise ValueError(f'locked train sample count changed: {len(bank.ids)}')
    consistent_rows = np.flatnonzero(bank.consistent)
    audit_rows = consistent_rows[bank.sequence[consistent_rows] == args.audit_sequence]
    train_rows = consistent_rows[bank.sequence[consistent_rows] != args.audit_sequence]
    if len(audit_rows) < 1000 or len(train_rows) < 10000:
        raise ValueError('sequence-held-out pretraining split is too small')
    audit_rows = spread(torch.as_tensor(audit_rows, dtype=torch.long), 2048).cpu().numpy()
    if args.smoke:
        audit_rows = audit_rows[:64]
        train_rows = train_rows[::200]
    audit_ids = torch.as_tensor(audit_rows, dtype=torch.long, device=base.DEVICE)
    train_ids = torch.as_tensor(train_rows, dtype=torch.long, device=base.DEVICE)
    audit_joint = bank.windows(audit_ids, 100).transpose(1, 2).contiguous()
    audit_generator = torch.Generator(device=base.DEVICE).manual_seed(args.seed + 7)
    audit_mask = sample_mask(len(audit_ids), audit_generator)

    baselines = {
        'B0_predict_zero': losses_on(audit_joint, audit_mask,
                                     torch.zeros_like(audit_joint)),
        'B1_visible_contemporaneous_mean': visible_mean_baseline(audit_joint, audit_mask),
    }
    args.best_baseline = min(baselines.values())
    baselines['best'] = args.best_baseline
    baselines['audit_samples'] = int(len(audit_ids))
    baselines['audit_sequence'] = args.audit_sequence
    baselines['training_pretrain_samples_before_stride'] = int(len(train_ids))
    baselines['note'] = ('B* is the better of zero and visible-channel contemporaneous-mean '
                         'predictors, scored with the exact same fixed masks on this held-out '
                         'recording. No predictor sees unmasked target joints.')
    write_json(args.out / 'baselines.json', baselines)
    print(json.dumps({'baselines': baselines}, ensure_ascii=False), flush=True)

    model = ChannelIdentifiedTCTST().to(base.DEVICE)
    pretrainer = MaskedPretrainer(model).to(base.DEVICE)
    print(json.dumps({'parameters': sum(p.numel() for p in model.parameters()),
                      'pretrainer_parameters': sum(p.numel() for p in pretrainer.parameters())}),
          flush=True)

    before = feature_stats(model, audit_joint[:256])

    history, stop_reason = train(args, bank, model, pretrainer, train_ids,
                                 audit_joint, audit_mask, args.out)

    after = feature_stats(model, audit_joint[:256])
    final_audit = history[-1]['audit_loss']
    variance_ratio = after['across_sample_sd'] / max(before['across_sample_sd'], 1e-12)

    verdict, reasons = 'PRETRAIN_PASS', []
    if args.smoke:
        verdict = 'SMOKE_ONLY'
    else:
        if stop_reason is not None:
            verdict = 'PRETRAIN_FAIL'; reasons.append(stop_reason)
        if final_audit > PRETRAIN_PASS_FRACTION * args.best_baseline:
            verdict = 'PRETRAIN_FAIL'
            reasons.append(f'final audit {final_audit:.4f} > '
                           f'{PRETRAIN_PASS_FRACTION:.2f}*{args.best_baseline:.4f}')
        if variance_ratio < S5_COLLAPSE_SD and final_audit > PRETRAIN_PASS_FRACTION * args.best_baseline:
            verdict = 'PRETRAIN_COLLAPSE'
            reasons.append(f"feature sd ratio {variance_ratio:.3f} < {S5_COLLAPSE_SD} "
                           f'and held-out loss did not pass')

    diagnosis = {
        'verdict': verdict, 'reasons': reasons, 'epochs_run': len(history),
        'final_audit_loss': final_audit, 'best_baseline': args.best_baseline,
        'audit_over_baseline': final_audit / args.best_baseline,
        'first_audit_loss': history[0]['audit_loss'],
        'train_loss_first_last': [history[0]['train_loss'], history[-1]['train_loss']],
        'feature_stats_before': before, 'feature_stats_after': after,
        'feature_sd_ratio': variance_ratio,
        'note': ('Feature spread and effective rank are diagnostics, not standalone collapse '
                 'criteria. The audit recording is sequence-held-out from masked-pretraining updates.'),
    }
    write_json(args.out / 'diagnosis.json', diagnosis)
    write_json(args.out / 'verdict.json', {
        'verdict': verdict, 'reasons': reasons, 'smoke': args.smoke,
        'preregistration_sha256': manifest['preregistration_sha256'],
        'checkpoint_written': verdict == 'PRETRAIN_PASS',
        'finished_at': now(),
    })

    if verdict == 'PRETRAIN_PASS':
        torch.save({
            'verdict': verdict,
            'model': model.state_dict(),
            'window': model.window, 'joint_channels': JOINT_CHANNELS,
            'seed': args.seed, 'epochs': len(history),
            'audit_loss': final_audit, 'baselines': baselines,
            'preregistration_sha256': manifest['preregistration_sha256'],
            'code_hashes': manifest['code_hashes'],
            'source_hashes': manifest['source_hashes'],
        }, args.out / 'pretrained_encoder.pt')

    print(json.dumps({'verdict': verdict, 'reasons': reasons,
                      'final_audit_loss': final_audit,
                      'best_baseline': args.best_baseline,
                      'epochs_run': len(history)}, ensure_ascii=False), flush=True)
    return 0 if verdict in ('PRETRAIN_PASS', 'SMOKE_ONLY') else 2


if __name__ == '__main__':
    raise SystemExit(main())
