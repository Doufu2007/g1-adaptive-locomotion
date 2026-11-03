"""Package the preregistered winning phase checkpoint for simulation runtime."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_g1_external_methods as base  # noqa: E402
from scripts.g1_tctst_faithful_v2 import ChannelIdentifiedTCTST  # noqa: E402
from scripts.run_g1_external_methods import LongRateTCN  # noqa: E402
from scripts.run_g1_long_history_comparison import LongContextTCNEncoder  # noqa: E402
from src.tasks.velocity.phasor.data import G1_JOINT_ORDER  # noqa: E402
from src.tasks.velocity.phasor.encoder_tcn_context import (  # noqa: E402
    LongContextTCNEncoder as RuntimeTCN20,
)
from src.tasks.velocity.phasor.encoder_tcn_long_rate import LongRateTCNEncoder as RuntimeTCN100  # noqa: E402
from src.tasks.velocity.phasor.encoder_tctst_runtime import RepairedTCTSTEncoder  # noqa: E402

OUT = ROOT / 'results/g1_phase_model_comparison_20261023/runtime_selected'
SELECTION = ROOT / 'results/g1_phase_model_comparison_20261023/selected_model.json'
RUNS = ROOT / 'results/g1_phase_model_comparison_20261023/matched_100ep'
REFERENCE = ROOT / 'results/g1_phase_dataset_20261019/phase_encoder_runtime_v1.pt'


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def training_model(name: str) -> torch.nn.Module:
    if name == 'tcn20': return LongContextTCNEncoder()
    if name == 'tcn100_rate': return LongRateTCN()
    if name == 'tctst_repaired': return ChannelIdentifiedTCTST()
    raise ValueError(name)


def runtime_model(name: str, window: int) -> torch.nn.Module:
    if name == 'tcn20': return RuntimeTCN20()
    if name == 'tcn100_rate': return RuntimeTCN100()
    if name == 'tctst_repaired': return RepairedTCTSTEncoder(window=window)
    raise ValueError(name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, default=SELECTION)
    parser.add_argument('--output', type=Path, default=OUT / 'phasor_runtime.pt')
    parser.add_argument('--runs-dir', type=Path, default=RUNS,
                        help='folder containing <family>_seed_<seed>/final_checkpoint.pt')
    args = parser.parse_args()
    selection_path = args.selection.resolve(strict=True)
    selection = json.loads(selection_path.read_text(encoding='utf-8'))
    family = selection['selected_family']
    seed = int(selection['selected_seed_for_control'])
    source = args.runs_dir.resolve() / f'{family}_seed_{seed}' / 'final_checkpoint.pt'
    source = source.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f'refusing to overwrite {output}')
    run = torch.load(source, map_location='cpu', weights_only=False)
    state = run['model']
    window = int(run['window'])
    source_model = training_model(family).eval()
    source_model.load_state_dict(state, strict=True)
    encoder = runtime_model(family, window).eval()
    encoder.load_state_dict(state, strict=True)
    torch.manual_seed(20261023)
    example = torch.randn(8, window, 29)
    with torch.inference_mode():
        expected = source_model(example).float()[:, :2]
        actual = encoder(example).float()
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-6)

    ref = torch.load(REFERENCE, map_location='cpu', weights_only=False)
    mean = torch.as_tensor(run['normalization_mean'], dtype=torch.float32).reshape(-1)
    std = torch.as_tensor(run['normalization_std'], dtype=torch.float32).reshape(-1)
    ref_mean = torch.as_tensor(ref['mean'], dtype=torch.float32).reshape(-1)
    ref_std = torch.as_tensor(ref['std'], dtype=torch.float32).reshape(-1)
    torch.testing.assert_close(mean, ref_mean, rtol=0, atol=0)
    torch.testing.assert_close(std, ref_std, rtol=0, atol=0)
    if mean.shape != (29,) or std.shape != (29,) or not torch.all(std > 0):
        raise ValueError('frozen normalization does not match the 29-joint runtime')
    if family == 'tcn20':
        architecture, schema = 'tcn_long_context', 'g1.phase_runtime_checkpoint.experimental.v2'
    elif family == 'tcn100_rate':
        architecture, schema = 'tcn_long_rate', 'g1.phase_runtime_checkpoint.experimental.v3'
    else:
        architecture, schema = 'tctst_repaired', 'g1.phase_runtime_checkpoint.experimental.v4'
    ref_contract = ref['runtime_contract']
    contract = {
        'schema': schema,
        'source_checkpoint_sha256': sha(source),
        'dataset_sha256': ref['dataset_sha256'],
        'metadata_sha256': ref_contract['metadata_sha256'],
        'manifest_sha256': ref['data_manifest_sha256'],
        'reward_template_sha256': ref_contract['reward_template_sha256'],
        'joint_order': list(G1_JOINT_ORDER),
        'joint_dim': 29,
        'window': window,
        'phase_output_convention': '(sin(phi), cos(phi))',
        'control_dt_seconds': .02,
        'selected_model_family': family,
        'selected_seed': seed,
        'source_validation_selection_sha256': sha(selection_path),
    }
    checkpoint = {
        'state_dict': state,
        'architecture': architecture,
        'joint_dim': 29,
        'window': window,
        'preserve_time': False,
        'mean': mean,
        'std': std,
        'train_mean': mean,
        'train_std': std,
        'dataset_sha256': ref['dataset_sha256'],
        'data_manifest_sha256': ref['data_manifest_sha256'],
        'joint_names': list(G1_JOINT_ORDER),
        'output_convention': '(sin(phi), cos(phi))',
        'runtime_contract': contract,
        'source_checkpoint_path': str(source.relative_to(ROOT)),
        'source_checkpoint_sha256': sha(source),
        'selected_model_family': family,
        'selected_seed': seed,
        'selection_rule': selection['selection_rule'],
        'encoder_freeze_eligible': False,
        'note': 'Experimental MuJoCo phase-source checkpoint; not deployment authorization.',
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    loaded = torch.load(output, map_location='cpu', weights_only=False)
    runtime_model(family, window).load_state_dict(loaded['state_dict'], strict=True)
    print(json.dumps({
        'output': str(output), 'sha256': sha(output), 'family': family,
        'seed': seed, 'window': window, 'runtime_parity': 'PASS',
        'source_checkpoint': str(source.relative_to(ROOT)),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
