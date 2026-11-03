"""Finish independent read-only analyses once the fixed simulation queue is ready."""
from pathlib import Path
import time

from scripts import g1_phase_feedback_probe_20261026 as exp


def wait_for(predicate):
    while not predicate():
        path=exp.OUT/'queue_status.json'
        if path.exists() and exp.read(path)['status']=='FAILED':
            raise RuntimeError('Simulation failed; preserve attempts and investigate before continuing')
        time.sleep(10)


if __name__=='__main__':
    donors=[exp.OUT/'rollouts'/f'{arm}_{mode}_{seed}'/'manifest.json'
            for seed in exp.SEEDS for arm,mode in (('clock','clock'),('learned','live'))]
    wait_for(lambda: all(p.exists() and exp.read(p)['status']=='COMPLETED' for p in donors))
    from scripts import analyze_g1_phase_feedback_probe_20261026 as analysis
    import torch
    torch.set_num_threads(2)
    analysis.objective_audit()
    analysis.sensitivity()
    print('COMPLETED: all same-state sensitivity comparisons',flush=True)
    wait_for(lambda: exp.read(exp.OUT/'queue_status.json')['status']=='COMPLETED')
    analysis.collect()
    print('COMPLETED: all 180 episodes verified and all nine contrasts summarized',flush=True)
