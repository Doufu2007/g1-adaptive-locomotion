"""Hash the terrain hfield arrays for the training, S and new maps and check
that the ten new maps are distinct from every training / S-evaluation map.

The check validates itself first: the six maps already recorded in
``map_identities.json`` are recomputed with the same construction; if those six
reproduce, the same method is trustworthy for the ten new maps.

Run with the project root as cwd so ``scripts`` imports resolve.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(r'C:\Users\郑傅韬\Desktop\大学生活\论文\宇树\phasor')
sys.path.insert(0, str(ROOT))

from scripts import g1_phase_lr_factorial_20261027 as a
from scripts import g1_rough_domain_contract_20261028 as d
from scripts import g1_rough_confirmation_experiment_20261028 as ce

KEYS = ('hfield_data', 'hfield_size', 'hfield_nrow', 'hfield_ncol', 'hfield_adr')
STORED = json.loads(
    (ROOT / 'results/g1_rough_domain_adaptation_20261028/map_identities.json')
    .read_text(encoding='utf-8'))


def digest_for(study, seed):
    from mjlab.envs import ManagerBasedRlEnv
    cfg = ce.make_cfg(study, 'RE_live', seed, 15)
    env = ManagerBasedRlEnv(cfg=cfg, device='cpu')
    try:
        arrays = ce.physical_snapshot(env)
        return d.r.array_digest({k: arrays[k] for k in KEYS})
    finally:
        env.close()


def main():
    study, _ = a.setup()
    seeds = list(dict.fromkeys(
        list(d.TRAIN_MAPS) + [20264701, 20264702, 20264703]
        + list(range(20269101, 20269111))))
    out = {}
    for seed in seeds:
        try:
            out[seed] = digest_for(study, seed)
            print(f'{seed} {out[seed]}', flush=True)
        except Exception as error:  # noqa: BLE001 - report and continue
            out[seed] = f'ERROR {error!r}'
            print(f'{seed} ERROR {error!r}', flush=True)

    print('\n=== self-check against stored map_identities.json ===', flush=True)
    ok = True
    for key, value in STORED.items():
        got = out.get(int(key))
        same = got == value
        ok &= same
        print(f'  {key}: stored {value[:16]} recomputed {str(got)[:16]} {"MATCH" if same else "MISMATCH"}')

    print('\n=== distinctness ===', flush=True)
    by_value = {}
    for seed, value in out.items():
        by_value.setdefault(value, []).append(seed)
    collisions = {v: s for v, s in by_value.items() if len(s) > 1}
    print('  collisions:', collisions if collisions else 'none')
    new = list(range(20269101, 20269111))
    print('  new maps all distinct:', not any(set(s) & set(new) and len(s) > 1 for s in by_value.values()))

    json.dump(dict(digests=out, stored_match=ok, collisions=collisions),
              (ROOT / 'results/g1_period_sweep_cpu_20261028/map_identity_probe_20261029.json')
              .open('w', encoding='utf-8'), indent=1)
    return 0 if ok else 2


if __name__ == '__main__':
    sys.exit(main())
