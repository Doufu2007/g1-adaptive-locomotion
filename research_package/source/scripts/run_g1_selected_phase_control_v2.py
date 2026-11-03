"""Paired fixed-clock vs selected-phase MuJoCo runs for the 2026-10-23 model."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_g1_best_phase_control as control  # noqa: E402

OUT = ROOT / 'results/g1_phase_model_comparison_20261023/control_v1'
RUNTIME = ROOT / 'results/g1_phase_model_comparison_20261023/runtime_selected/phasor_runtime.pt'
SELECTION = ROOT / 'results/g1_phase_model_comparison_20261023/tctst_residual_v4/selected_model.json'
SEEDS = tuple(range(20261201, 20261231))
SCENARIOS = ('steady', 'speed_change', 'push')
TERRAINS = ('flat', 'rough')
BATCH_SIZE = 10


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if not RUNTIME.is_file():
        raise FileNotFoundError(f'selected runtime checkpoint is missing: {RUNTIME}')
    selection = json.loads(SELECTION.read_text(encoding='utf-8'))
    if int(selection['selected_seed_for_control']) != 20261021:
        raise ValueError('control seed differs from the frozen selection rule')
    import torch
    runtime = torch.load(RUNTIME, map_location='cpu', weights_only=False)
    if (runtime.get('selected_model_family') != selection['selected_family']
            or int(runtime.get('selected_seed', -1)) != int(selection['selected_seed_for_control'])):
        raise ValueError('runtime checkpoint does not match the frozen model selection')
    contract = runtime.get('runtime_contract', {})
    if (contract.get('selected_model_family') != selection['selected_family']
            or int(contract.get('selected_seed', -1)) != int(selection['selected_seed_for_control'])):
        raise ValueError('runtime contract does not match the frozen model selection')
    if args.dry_run:
        print(json.dumps({'family': selection['selected_family'], 'seed': SEEDS,
                          'scenarios': SCENARIOS, 'terrains': TERRAINS,
                          'batches': len(SEEDS) // BATCH_SIZE,
                          'runtime': str(RUNTIME)}, ensure_ascii=False, indent=2))
        return 0
    if OUT.exists():
        raise FileExistsError(f'refusing to overwrite {OUT}')
    OUT.mkdir(parents=True)
    control.ENCODER = RUNTIME
    control.OUT = OUT / 'rollouts'
    control.OUT.mkdir()
    # Import after setting ENCODER/OUT because the batched runner imports these
    # names from the frozen pilot module.
    from scripts.g1_best_phase_batched import rollout_batched

    for terrain in TERRAINS:
        for batch_start in range(0, len(SEEDS), BATCH_SIZE):
            batch = SEEDS[batch_start:batch_start + BATCH_SIZE]
            batch_id = batch_start // BATCH_SIZE + 1
            for arm in ('clock', 'learned'):
                output = control.OUT / f'{terrain}_{arm}_batch{batch_id:02d}'
                rollout_batched(SimpleNamespace(
                    terrain=terrain, arm=arm, label='selected_phase_direct_swap',
                    seeds=list(batch), scenarios=list(SCENARIOS), horizon=20.,
                    output=str(output), policy=None,
                ))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
