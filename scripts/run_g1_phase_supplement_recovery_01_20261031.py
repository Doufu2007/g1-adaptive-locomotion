"""One reviewed zero-sampling recovery. No retry loop and no disk-floor change.

Original formal launch failed its disk gate before creating any training cell.
All smoke evidence, source and scientific constants stay pinned to that seal.
Only the new formal output/control roots and child entry point are redirected.
"""
from pathlib import Path
import json
import os
import site
import sys

site.addsitedir(str(Path.home() / 'AppData/Roaming/Python/Python313/site-packages'))
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import g1_phase_supplement_20261031 as p

BASE_CONTROL = p.CONTROL
BASE_OUT = p.OUT
ORIGINAL_SHA = '31535c957252fd9a31d33db630ecb21a53e88941fb77ee4bebec9498750cb1d8'
CONTROL = BASE_CONTROL / 'formal_recovery_01'
OUT = BASE_OUT / 'formal_recovery_01'
SELF = Path(__file__).resolve()


def prepare():
    import psutil
    if p.a.sha(BASE_CONTROL / 'protocol.json') != ORIGINAL_SHA:
        raise ValueError('original protocol changed')
    original = p.verify_seal()
    status = p.a.read(BASE_CONTROL / 'queue_status.json')
    if status['status'] != 'FAILED' or status['completed_jobs'] != 0:
        raise ValueError('only the documented zero-job disk failure is recoverable here')
    if 'below locked 10GiB free disk floor' not in status['error']:
        raise ValueError('not the reviewed technical failure')
    if list((BASE_OUT / 'training').glob('*/manifest.json')) or list((BASE_OUT / 'rollouts').rglob('manifest.json')):
        raise ValueError('formal sampling evidence exists; cannot claim zero-sampling restart')
    pid = p.a.read(BASE_OUT / 'queue_started.json')['pid']
    if psutil.pid_exists(pid):
        try:
            if 'g1_phase_supplement' in ' '.join(psutil.Process(pid).cmdline()):
                raise ValueError('original queue still alive')
        except psutil.NoSuchProcess:
            pass
    review_path = BASE_CONTROL / 'technical_recovery_01_review.json'
    review = p.a.read(review_path)
    if review['status'] != 'PASS' or review['wrapper_sha256'] != p.a.sha(SELF):
        raise ValueError('fresh recovery implementation review missing/changed')
    CONTROL.mkdir(parents=True, exist_ok=False)
    OUT.mkdir(parents=True, exist_ok=False)
    value = dict(original)
    value.update(status='LOCKED_BEFORE_FORMAL_RECOVERY_01', at=p.a.now(),
        recovery_of_protocol_sha256=ORIGINAL_SHA, original_formal_train_transitions=0,
        original_formal_eval_transitions=0, original_formal_PPO_updates=0, original_formal_Adam_steps=0,
        simulation_resume=False, budget_change=False,
        recovery_rule='one new attempt only after original C>=10GiB/E>=15GiB disk gates pass; no automatic retry',
        control=str(CONTROL), output=str(OUT))
    value['files'] = list(original['files']) + [dict(path=str(SELF), sha256=p.a.sha(SELF)),
        dict(path=str(review_path), sha256=p.a.sha(review_path))]
    with (CONTROL / 'protocol.json').open('x', encoding='utf8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
    p.a.write(CONTROL / 'recovery_ready.json', dict(status='PREPARED_WAITING_FOR_DISK', at=p.a.now(),
        C_free_GiB=p.shutil.disk_usage(ROOT).free / 1024**3,
        E_free_GiB=p.shutil.disk_usage('E:/').free / 1024**3, actual_sampling=0))


def child_args(command, **kwargs):
    args = [sys.executable, '-X', 'utf8', str(SELF), command]
    for key, value in kwargs.items():
        if value is True:
            args.append('--' + key.replace('_', '-'))
        elif value is not None and value is not False:
            args.extend(['--' + key.replace('_', '-'), str(value)])
    return args


def configure():
    p.OUT = OUT; p.CONTROL = CONTROL; p.child_args = child_args


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'prepare':
        prepare(); return
    configure()
    if len(sys.argv) > 1 and sys.argv[1] == 'status':
        print(json.dumps(dict(queue=p.a.read(CONTROL / 'queue_status.json') if (CONTROL / 'queue_status.json').exists() else None,
            ready=p.a.read(CONTROL / 'recovery_ready.json'),
            C_free_GiB=p.shutil.disk_usage(ROOT).free / 1024**3,
            E_free_GiB=p.shutil.disk_usage('E:/').free / 1024**3), ensure_ascii=False, indent=2))
        return
    # Gate before launch reserves a flag or starts any process. queue() and each
    # run_child repeat the same gate; failures preserve this attempt and stop.
    p.check_disk()
    if len(sys.argv) < 2 or sys.argv[1] not in ('launch', 'queue', 'train', 'rollout', 'analyze', 'audit'):
        raise ValueError('recovery entry permits only the reviewed formal pipeline')
    p.main()


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        # No automatic relaunch. Resource evidence stays with this attempt.
        if CONTROL.exists():
            configure()
            try:
                p.failure_evidence(error)
            except Exception:
                import traceback
                traceback.print_exc()
        raise
