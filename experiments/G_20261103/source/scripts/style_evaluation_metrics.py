"""Evaluation helpers that exclude resets from fault recovery claims."""
import torch


def alive_rmse(values, alive, start, end):
    selected = values[start:end][alive[start:end]]
    return float(selected.square().mean().sqrt()) if selected.numel() else None


def recovery_without_falls(errors, alive, start, end, dt):
    cohort = alive[max(0, start-1)]
    eligible = int(cohort.sum())
    horizon = errors.shape[0] - end
    hold = max(1, round(0.1/dt))
    baseline = errors[max(0,start-round(1.0/dt)):start].mean(0)
    threshold = torch.maximum(baseline+0.15, torch.full_like(baseline,0.25))
    recovered = torch.zeros_like(cohort)
    first = torch.full(cohort.shape, horizon, dtype=torch.long)
    for index in range(end, errors.shape[0]-hold+1):
        stable = torch.all(errors[index:index+hold] <= threshold,dim=0)
        stable &= alive[index+hold-1] & cohort
        newly = stable & ~recovered
        first[newly] = index-end
        recovered |= stable
    return {'recovery_without_fall_rate':float(recovered.sum())/eligible if eligible else None,
            'recovery_eligible_envs':eligible,
            'recovery_restricted_mean_time_s':float(first[cohort].float().mean())*dt if eligible else None,
            'recovery_censor_horizon_s':horizon*dt}
