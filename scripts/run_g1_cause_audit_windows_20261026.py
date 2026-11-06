"""Technical-only recovery: reuse matching DOS aliases and one bootstrap module."""
import argparse
import ctypes
from ctypes import wintypes
import importlib.util
import json
from pathlib import Path

from scripts import diagnose_g1_best_e_closed_loop_20261026 as audit

_MODULE = None


def device_targets(letter):
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    query = kernel.QueryDosDeviceW
    query.argtypes = (wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD)
    query.restype = wintypes.DWORD
    buffer = ctypes.create_unicode_buffer(65536)
    size = query(letter+':', buffer, len(buffer))
    if not size:
        error = ctypes.get_last_error()
        if error == 2:
            return []
        raise ctypes.WinError(error)
    return [p for p in buffer[:size].split('\0') if p]


def bootstrap():
    global _MODULE
    if _MODULE is None:
        harness = audit.ROOT/'results/phasor_pilot_20261016/run_pilot.py'
        spec = importlib.util.spec_from_file_location('_cause_audit_native_bootstrap', harness)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        original_define = module._define_dos_device
        def define_once(letter, target):
            expected = '\\??\\'+str(Path(target).resolve())
            targets = device_targets(letter)
            if targets:
                if targets[0].casefold() != expected.casefold():
                    raise RuntimeError('refusing to replace unrelated DOS alias '+letter)
                return  # Borrow an identical mapping; do not add or remove mappings.
            original_define(letter, target)
        module._define_dos_device = define_once
        _MODULE = module
    np, torch = _MODULE.import_project()
    from mjlab.utils.torch import configure_torch_backends
    configure_torch_backends()
    torch.set_num_threads(2)
    from src.tasks.velocity.phasor.mujoco_io import install_mjlab_scene_xml_unicode_compat
    install_mjlab_scene_xml_unicode_compat()
    return np, torch


def provenance():
    return dict(recovery_launcher_sha256=audit.sha(__file__),
        original_phase_script_sha256=audit.sha(audit.ROOT/'scripts/diagnose_g1_best_e_closed_loop_20261026.py'),
        original_training_script_sha256=audit.sha(audit.ROOT/'scripts/diagnose_g1_best_e_training_20261026.py'),
        change='Reuse already matching DOS aliases and cache the original native bootstrap module. '
               'No change to model, task, physics, calibration, seed, reward condition or update budget.')


def install():
    from scripts import run_g1_best_phase_control as base
    base.bootstrap = bootstrap
    original_rollout = audit.local_rollout
    def tagged_rollout(args):
        try:
            return original_rollout(args)
        finally:
            path = audit.OUT/'rollouts'/args.name/'manifest.json'
            if path.exists():
                value = json.loads(path.read_text(encoding='utf-8'))
                value['technical_runtime_recovery'] = provenance()
                path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    audit.local_rollout = tagged_rollout
    from scripts import diagnose_g1_best_e_training_20261026 as training
    original_train = training.train
    def tagged_train(arm, seed, mode, *, smoke=False):
        try:
            return original_train(arm, seed, mode, smoke=smoke)
        finally:
            path = training.OUT/('smoke' if smoke else 'training')/f'{arm}_{mode}_{seed}'/'manifest.json'
            if path.exists():
                value = json.loads(path.read_text(encoding='utf-8'))
                value['technical_runtime_recovery'] = provenance()
                training.write(path, value)
    training.train = tagged_train
    return training


def selftest():
    before = {letter:device_targets(letter) for letter in ('X','Y','Z')}
    assert all(before.values()), 'The recovery test expects existing mappings from the failed run'
    bootstrap()
    first = id(_MODULE)
    bootstrap()
    after = {letter:device_targets(letter) for letter in ('X','Y','Z')}
    assert before == after, 'Repeated initialization must not grow the alias stacks'
    assert id(_MODULE) == first
    audit.write('windows_bootstrap_recovery_test.json',dict(status='PASS',at=audit.now(),
        aliases_unchanged=True,module_reused=True,
        mapping_counts={k:len(v) for k,v in after.items()},**provenance()))
    print('PASS: bootstrap repeated with unchanged DOS aliases and reused module', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',required=True,choices=['selftest','phase-queue','training-queue'])
    args = parser.parse_args()
    if args.stage == 'selftest':
        selftest()
    else:
        training = install()
        if args.stage == 'phase-queue':
            audit.queue()
        else:
            training.queue()
