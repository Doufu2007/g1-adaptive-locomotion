"""Isolate where the run-to-run divergence enters.

The v2 probe showed that two identical invocations of the full rollout differ
from the first ``env.step()`` onward, with bit-identical initial state.  That
leaves two candidates:

  1. the physics solver itself is not reproducible, or
  2. something upstream of the action (observation assembly, policy inference)
     is not reproducible, and the solver merely amplifies it.

This script removes the policy entirely: it steps the same environment with a
*pre-generated, fixed* action sequence, so any divergence between two runs must
come from the simulator.  It can also switch the backend, to test whether the
GPU path is the source.

Usage:
    run     --tag a --device cuda:0   # then repeat with --tag b
    compare                              # bitwise comparison of two tags
"""
from __future__ import annotations

from pathlib import Path
import argparse
import json

import numpy as np

from scripts import g1_phase_lr_factorial_20261027 as a
from scripts import g1_rough_confirmation_contract_20261028 as c
from scripts import g1_rough_confirmation_experiment_20261028 as ce

ROOT = a.ROOT
OUT = ROOT / 'results/g1_determinism_diagnosis_20261028'


def probe(tag, device, steps, worlds, terrain):
    study, torch = a.setup()
    from mjlab.envs import ManagerBasedRlEnv

    out = OUT / tag
    if (out / 'trace.npz').exists():
        raise ValueError('output exists; refusing to overwrite: ' + str(out))
    out.mkdir(parents=True, exist_ok=True)

    cfg = ce.make_cfg(study, 'RC_clock', terrain, worlds)
    record = dict(tag=tag, device=device, steps=steps, worlds=worlds,
                  terrain=terrain, status='RUNNING', started_at=a.now(),
                  common_cfg_sha256=study.common_cfg_digest(cfg))
    a.write(out / 'manifest.json', record)
    env = None
    try:
        ce.seed_all(torch, 20264200)
        env = ManagerBasedRlEnv(cfg=cfg, device=device)
        if abs(float(env.step_dt) - .02) > 1e-9:
            raise ValueError('50Hz required')
        dim = int(env.action_manager.total_action_dim)
        # Fixed action sequence: same numbers on every run and every device.
        rng = np.random.default_rng(20261028)
        actions = rng.normal(0., .5, size=(steps, worlds, dim)).astype(np.float32)
        record['action_dim'] = dim
        record['action_sha256'] = c.array_digest({'actions': actions})

        slots = c.world_slots()[:worlds]
        for i, s in enumerate(slots):
            ce.seed_all(torch, s['eval_seed'])
            env.reset(seed=s['eval_seed'], env_ids=torch.tensor([i], device=env.device))
        record['slots'] = slots

        robot = env.scene['robot']
        # Pre-step state, so a differing trace can be blamed on the solver
        # rather than on a differing reset.
        trace = {'joint_pos': [], 'joint_vel': [], 'root_pos': [], 'qacc': []}
        initial = dict(
            init_joint_pos=robot.data.joint_pos.detach().cpu().numpy().copy(),
            init_joint_vel=robot.data.joint_vel.detach().cpu().numpy().copy(),
            init_root_pos=robot.data.root_link_pos_w.detach().cpu().numpy().copy(),
            init_root_quat=robot.data.root_link_quat_w.detach().cpu().numpy().copy())
        for step in range(steps):
            env.command_manager.get_command('twist')[:] = torch.tensor(
                [.5, 0., 0.], device=env.device)
            action = torch.as_tensor(actions[step], device=env.device)
            env.step(action)
            trace['joint_pos'].append(robot.data.joint_pos.detach().cpu().numpy().copy())
            trace['joint_vel'].append(robot.data.joint_vel.detach().cpu().numpy().copy())
            trace['root_pos'].append(robot.data.root_link_pos_w.detach().cpu().numpy().copy())
            qacc = getattr(env.sim.data, 'qacc', None)
            trace['qacc'].append(np.zeros(1, np.float32) if qacc is None
                                 else qacc.detach().cpu().numpy().copy())
        arrays = {k: np.asarray(v) for k, v in trace.items()}
        arrays.update(initial)
        np.savez_compressed(out / 'trace.npz', **arrays)
        record.update(status='COMPLETED', ended_at=a.now(),
                      trace_sha256=c.array_digest(arrays))
    except BaseException as error:
        record.update(status='FAILED', ended_at=a.now(), error=repr(error))
        raise
    finally:
        a.write(out / 'manifest.json', record)
        if env is not None:
            env.close()


def compare(tag_a, tag_b):
    A = np.load(OUT / tag_a / 'trace.npz')
    B = np.load(OUT / tag_b / 'trace.npz')
    print(f'{tag_a} vs {tag_b}')
    ma = json.loads((OUT / tag_a / 'manifest.json').read_text(encoding='utf-8'))
    mb = json.loads((OUT / tag_b / 'manifest.json').read_text(encoding='utf-8'))
    print(f'  device {ma["device"]} vs {mb["device"]}  '
          f'actions identical: {ma["action_sha256"] == mb["action_sha256"]}')
    first = None
    for key in A.files:
        x, y = A[key], B[key]
        if x.shape != y.shape:
            print(f'  {key}: shape mismatch {x.shape} vs {y.shape}')
            continue
        diff = np.abs(x.astype(np.float64) - y.astype(np.float64))
        worst = float(diff.max()) if diff.size else 0.
        per_step = diff.reshape(diff.shape[0], -1).max(axis=1) if diff.ndim else diff
        bad = np.flatnonzero(per_step > 0)
        step = int(bad[0]) if bad.size else None
        print(f'  {key:10s} bitwise-identical={worst == 0}  max|d|={worst:.3e}  '
              f'first differing step={step}')
        if step is not None and first is None:
            first = step
    print(f'  => first divergence at step {first}' if first is not None
          else '  => identical over the whole trace')


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('run', 'compare'))
    parser.add_argument('--tag', default='a')
    parser.add_argument('--other', default='b')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--worlds', type=int, default=2)
    parser.add_argument('--terrain', type=int, default=20264701)
    args = parser.parse_args(argv)
    if args.command == 'run':
        probe(args.tag, args.device, args.steps, args.worlds, args.terrain)
    else:
        compare(args.tag, args.other)


if __name__ == '__main__':
    main()
