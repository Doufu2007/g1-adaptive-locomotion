"""Metrics from saved simulator telemetry; no model fitting or label calibration."""
from __future__ import annotations

import numpy as np


def fixed_horizon_mae(velocity: np.ndarray, commands: np.ndarray) -> float:
    if len(velocity) > len(commands) or not len(commands):
        raise ValueError('invalid fixed horizon')
    padded = np.zeros(len(commands), dtype=float)
    padded[:len(velocity)] = velocity
    return float(np.mean(np.abs(padded - commands)))


def phase_metrics(times: np.ndarray, angle: np.ndarray, contacts: np.ndarray) -> dict:
    if len(angle) != len(times) or contacts.shape != (len(times), 2):
        raise ValueError('telemetry shapes do not match')
    if not np.isfinite(angle).all() or not np.isfinite(times).all():
        raise ValueError('non-finite phase telemetry')
    # A contact already present at reset is not an observed touchdown.
    events = np.flatnonzero(contacts[1:, 0] & ~contacts[:-1, 0]) + 1
    kept = []
    for index in events:
        if not kept or times[index] - times[kept[-1]] >= .08:
            kept.append(int(index))
    unwrapped = np.unwrap(angle)
    cycles = []
    for start, end in zip(kept[:-1], kept[1:]):
        period = float(times[end] - times[start])
        if period < .15:
            continue
        net = float(unwrapped[end-1] - unwrapped[start])
        # Net progression uses the half-open gate convention, but frequency
        # must use the matching elapsed time. Dividing by the full contact
        # period creates a spurious correlation even for a perfect clock.
        observed_duration = float(times[end-1] - times[start])
        cycles.append(dict(start=start, end=end, period_seconds=period,
                           measured_frequency_hz=1/period,
                           predicted_frequency_hz=net/(2*np.pi*observed_duration),
                           net_progression_rad=net,
                           one_turn=bool(abs(net-2*np.pi) <= np.pi/6)))
    n = len(cycles)
    result = dict(status='measured' if n >= 3 else 'insufficient_cycles',
                  complete_cycles=n, left_touchdown_count=len(kept), cycles=cycles,
                  one_turn_fraction=float(np.mean([c['one_turn'] for c in cycles])) if n else None,
                  touchdown_phase_concentration=float(abs(np.exp(1j*angle[kept]).mean())) if kept else None,
                  frequency_correlation=None)
    if n >= 3:
        actual = np.array([c['measured_frequency_hz'] for c in cycles])
        predicted = np.array([c['predicted_frequency_hz'] for c in cycles])
        # Float32 sin/cos observations leave sub-microhertz clock roundoff.
        # A correlation with an effectively constant signal is undefined.
        if actual.std() > 1e-5 and predicted.std() > 1e-5:
            result['frequency_correlation'] = float(np.corrcoef(actual, predicted)[0, 1])
    return result


def command_schedule(steps: int, dt: float, scenario: str) -> np.ndarray:
    t = np.arange(steps)*dt
    if scenario == 'speed_change':
        return np.where(t < 5, .3, np.where(t < 12, .8, .4))
    if scenario not in ('steady', 'push'):
        raise ValueError(scenario)
    return np.full(steps, .5)


def summarize_episode(raw: dict, horizon_commands: np.ndarray, dt: float,
                      terminated: bool) -> dict:
    velocity = raw['velocity'][:, 0]
    position = raw['position']
    n = len(velocity)
    moving = float(np.mean(velocity > .1))
    result = dict(observed_steps=n, horizon_steps=len(horizon_commands),
                  fell=bool(terminated), survival_seconds=n*dt,
                  mae_forward_velocity_mps=fixed_horizon_mae(velocity, horizon_commands),
                  mean_observed_forward_velocity_mps=float(velocity.mean()),
                  forward_displacement_m=float(position[-1, 0]-position[0, 0]),
                  moving_fraction=moving,
                  phase=phase_metrics(raw['time'], raw['angle'], raw['contact']),
                  push_recovery=[])
    result['locomotion_observed'] = bool(result['forward_displacement_m'] > 1.0 and moving > .5)
    if raw['scenario'] == 'speed_change':
        result['speed_switch_exposed'] = {str(at): round(at/dt) < n for at in (5., 12.)}
    if raw['scenario'] == 'push':
        for at in (5., 10.):
            start = round(at/dt)
            end = min(n, round(10./dt)) if at == 5. else n
            recovered = None
            if start < n:
                error = np.hypot(velocity[start:end] - horizon_commands[start:end],
                                 raw['velocity'][start:end, 1])
                stable = round(1./dt)
                for k in range(max(0, len(error)-stable+1)):
                    if np.all(error[k:k+stable] < .15):
                        recovered = k*dt
                        break
            result['push_recovery'].append(dict(at_seconds=at, exposed=start < n,
                                                 recovered_seconds=recovered,
                                                 observation_end_seconds=end*dt,
                                                 censored_at_next_push=recovered is None and at == 5. and end == round(10./dt)))
    return result
