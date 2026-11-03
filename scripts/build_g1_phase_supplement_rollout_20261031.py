"""One-time exact adaptation of the frozen raw rollout; fail if anchors drift."""
from pathlib import Path


def build():
    source = Path('scripts/g1_period_sweep_20261028_v3.py').read_text(encoding='utf8')
    body = source[source.index('def rollout('):source.index('\n\ndef child(')]
    def change(old, new):
        nonlocal body
        if old not in body:
            raise ValueError('missing adaptation anchor: ' + old[:90])
        body = body.replace(old, new)
    begin = body.index('    study, torch = a.setup()')
    body = '''def rollout(split, mode, terrain_seed, seed, smoke=False, repeat=False):
    if not smoke:
        verify_seal()
    check_disk()
    method = mode
    device = 'cpu'
    tag = split
    period = .55 if mode == 'clock55' else .60
''' + body[begin:]
    p = body.index("    learned = method.startswith('RE')")
    q = body.index('    options = ort.SessionOptions()')
    body = body[:p] + '''    learned = mode in ('live', 'constant', 'replay')
    if not smoke and (terrain_seed not in c.MAPS[split] or seed not in c.NEW_SEEDS):
        raise ValueError('outside locked evaluation split')
    if split == 'calibration' and mode != 'live':
        raise ValueError('only live calibration')
    if split == 'validation' and mode not in ('clock55', 'clock60'):
        raise ValueError('only clock validation')
    out = rollout_folder('smoke' if smoke else split,
                         mode + ('_repeat' if repeat else ''), terrain_seed, seed)
    reserve(out)
    slots = ([dict(train_seed=seed, eval_seed=20271911, patch_col=0)] if smoke
             else c.slots(split, seed))
    batch = len(slots)
    limit = 200 if smoke else 1000
    scenarios = ('push06',) if smoke else (('steady',) if split in ('calibration', 'validation') else c.SCENARIOS)
    entries = [policy_entry(split, mode, s['train_seed'], smoke) for s in slots]
    increments = bank(smoke) if mode in ('constant', 'replay') else None
''' + body[q:]
    change('cfg = ce.make_cfg(study, method, terrain_seed, batch)\n    if not learned:\n        apply_period(cfg, period)',
           'cfg = eval_cfg(study, mode, terrain_seed, batch)')
    change('ce.seed_all(torch, 20264200)', 'ce.seed_all(torch, c.CONSTRUCT_SEED)')
    change("ce.pairing_guard(OUT / tag / 'pairing' /\n                             f'terrain{terrain_seed}_{scenario}.json', pairing)",
           "map_identity(terrain_seed, physical)\n            ce.pairing_guard(OUT / ('smoke_pairing' if smoke else 'pairing') /\n                             f'{tag}_terrain{terrain_seed}_seed{seed}_{scenario}.json', pairing)\n            a.write(out / f'{scenario}_initial.json', pairing)")
    change("'foot_contact_interior', 'terminated', 'timed_out')",
           "'foot_contact_interior', 'terminated', 'timed_out', 'live_phase', 'actor_input', 'action', 'push_delta', 'root_velocity_world')")
    change('            for step in range(limit):', '            initial_angle = None\n            for step in range(limit):')
    change("                if scenario == 'push' and step in (250, 500):",
           '                delta_push = c.push_delta(scenario, step, smoke)\n                if delta_push:')
    change("velocity[:, 1] += .3 if step == 250 else -.3", "velocity[:, 1] += delta_push")
    change("                phase_np = live_phase.cpu().numpy().copy()",
           "                live_np = live_phase.cpu().numpy().copy()\n                phase_np = live_np.copy()")
    change('                action = c.route_actions(obs, sessions, names)', '''                if initial_angle is None:
                    initial_angle = np.arctan2(live_np[:, 0], live_np[:, 1]).astype(float)
                if increments is not None:
                    phase_np = c.open_loop_phase(initial_angle,
                        [s['train_seed'] for s in slots], increments, step, mode)
                    obs = c.replace_phase(obs, phase_np)
                action = oldc.route_actions(obs, sessions, names)''')
    change('                              phase_used=phase_np,', '''                              phase_used=phase_np, live_phase=live_np, actor_input=obs.copy(), action=action.copy(),
                              push_delta=np.full(batch, delta_push),
                              root_velocity_world=robot.data.root_link_vel_w.cpu().numpy().copy(),''')
    change("                result = env.step(torch.as_tensor(action, device=env.device))\n                record['transitions'] += active",
           "                result = env.step(torch.as_tensor(action, device=env.device))\n                raw_capacity(env)\n                record['transitions'] += active\n                record['simulator_world_steps'] = record.get('simulator_world_steps', 0) + batch\n                if step % 100 == 0:\n                    a.write(out / 'progress.json', dict(at=a.now(), scenario=scenario, step=step+1, transitions=record['transitions'], simulator_world_steps=record['simulator_world_steps']))")
    change("score = c.episode_metrics(value, 20., scenario, bool(fell[i]),\n                                              bool(terrain_seed))",
           "score = oldc.episode_metrics(value, 20., 'steady', bool(fell[i]), True)\n                    score['push_recovery'] = c.recovery(value['time'], value['velocity'], bool(fell[i]))")
    change("score.update(probe=tag, method=method, terrain_seed=terrain_seed,", "score.update(split=split, mode=mode, probe=tag, method=method, terrain_seed=terrain_seed,")
    change("record.update(status='COMPLETED', ended_at=a.now(), episode_count=len(episodes))",
           "record.update(status='COMPLETED', ended_at=a.now(), episode_count=len(episodes), protocol_sha256=(a.sha(CONTROL / 'protocol.json') if (CONTROL / 'protocol.json').exists() else None))")
    change("record.update(status='FAILED', ended_at=a.now(), error=repr(error),",
           "record.update(status='FAILED', ended_at=a.now(), error=repr(error),")
    target = Path('scripts/g1_phase_supplement_20261031.py')
    existing = target.read_text(encoding='utf8')
    if '\ndef rollout(' in existing:
        raise ValueError('rollout already generated; preserve existing implementation')
    target.write_text(existing + '\n\n' + body + '\n', encoding='utf8')


if __name__ == '__main__':
    build()
