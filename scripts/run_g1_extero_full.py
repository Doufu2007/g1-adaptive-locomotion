"""Escalate the terrain-scan comparison to the 10,001-update budget."""
from __future__ import annotations

import copy
import json
from pathlib import Path

from scripts import run_g1_extero_long as long

FULL_ITERATIONS = 10001
FULL_OUTPUT = long.ROOT / 'results/g1_extero_full_20261105'


def full_protocol():
    protocol = copy.deepcopy(long.long_protocol())
    protocol['schema'] = 'g1.extero_split.full_screening.v1'
    protocol['status'] = 'FULL_SCREENING_FROZEN'
    protocol['training']['pilot_iterations'] = FULL_ITERATIONS
    protocol['training']['formal_iterations'] = FULL_ITERATIONS
    protocol['long_screening']['iterations'] = FULL_ITERATIONS
    protocol['long_screening']['purpose'] = 'escalation after the 3000-update paired screening was inconclusive'
    return protocol


def run_training(output=FULL_OUTPUT, *, device='cuda:0', resume=True):
    long.PROTOCOL = full_protocol()
    return long.run_training(output, device=device, resume=resume)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=FULL_OUTPUT)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--no-resume', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run_training(args.output, device=args.device, resume=not args.no_resume), ensure_ascii=False))
