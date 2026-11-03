"""Record version 2 of the period-sweep probe without touching lock.json.

Version 1 failed its own reproduction gate.  The cause was a slot-enumeration
error: v1 drew its episodes from the confirmation contract's ``world_slots()``,
which enumerates the confirmation evaluation seeds 20264911-20264915, and
applied them to the S evaluation maps.  The audited S condition uses
20264811-20264815.  V1 is therefore archived rather than edited, and version 2
differs from it only in the slot enumeration plus the child-process environment.

No P or M outcome had been inspected for any non-nominal period or any new map
at the time this was written.
"""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/g1_prereg_20261028'


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    path = OUT / 'lock_v2.json'
    if path.exists():
        raise SystemExit('lock_v2.json exists; versions are immutable')
    v1 = ROOT / 'scripts/g1_period_sweep_20261028.py'
    v2 = ROOT / 'scripts/g1_period_sweep_20261028_v2.py'
    archived = ROOT / 'results/g1_period_sweep_v1_gate_failed_20261028'
    v1_gate = archived / 'P' / 'RC_clock_terrain20264703_p0.60' / 'manifest.json'
    value = dict(
        status='LOCKED_BEFORE_NON_NOMINAL_PERIOD_OR_NEW_MAP_OUTCOMES_READ',
        written_at=datetime.now(timezone.utc).isoformat(),
        supersedes='results/g1_prereg_20261028/lock.json (v1 probe script only)',
        v1=dict(path=str(v1), sha256=sha(v1), status='SUPERSEDED'),
        v2=dict(path=str(v2), sha256=sha(v2), status='ACTIVE'),
        archived_v1_output=str(archived),
        v1_gate_result=dict(
            completed=27, fell=48, displacement_m=4.473807045900159,
            expected_completed=22, expected_fell=46, expected_displacement_m=4.2437,
            passed=False),
        diagnosis=(
            'v1 enumerated confirmation evaluation seeds 20264911-20264915 and '
            'applied them to the S evaluation maps 20264701-20264703. The audited '
            'S condition uses evaluation seeds 20264811-20264815. The rollout code '
            'is unchanged between versions; only the slot enumeration differs.'),
        v2_change=(
            'probe_slots("P") now draws train_seed/eval_seed/patch_col from '
            'g1_rough_domain_contract_20261028, matching the audited S run; the M '
            'probe keeps the confirmation slots; child processes now inherit the '
            'same PYTHONHASHSEED/OMP/MKL settings as the S and confirmation parents.'),
        unchanged=dict(
            periods=[0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80],
            s_maps=[20264701, 20264702, 20264703],
            new_maps=list(range(20269101, 20269111)),
            nominal_period=0.60,
            primary_endpoint='completion-rate difference, paired McNemar, alpha=.05'),
    )
    OUT.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1), encoding='utf-8')
    print(json.dumps(dict(written_at=value['written_at'],
                          v2_sha256=value['v2']['sha256'][:16])))


if __name__ == '__main__':
    main()
