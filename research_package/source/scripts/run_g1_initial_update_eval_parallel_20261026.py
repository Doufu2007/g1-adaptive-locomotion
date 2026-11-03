"""Evaluate the last four locked cells in one separate process, without extra trials."""
import contextlib
import gc
from pathlib import Path
from scripts import diagnose_g1_initial_update_20261026 as exp


def main():
    protocol=exp.lock();jobs=protocol['evaluation_order'][-4:]
    path=exp.OUT/'parallel_evaluation_execution.json'
    record=dict(status='LOCKED_BEFORE_DISPATCH',at=exp.now(),jobs=jobs,
        script_sha256=exp.sha(__file__),protocol_sha256=exp.sha(exp.OUT/'prospective_protocol.json'),
        reason='Execution-only overlap with serial worker; unchanged seeds, three-world configuration and 36-episode budget',
        maximum_concurrent_evaluation_processes=2,
        limitation='Actual completion order overlaps; no cross-process bitwise physics identity claim',
        extra_environment_training_transitions=0,extra_evaluation_episodes=0)
    assert not path.exists()
    for seed,condition in jobs:assert not (exp.OUT/'rollouts'/f'{condition}_{seed}').exists()
    exp.write(path,record)
    completed=[]
    try:
        for seed,condition in jobs:
            name=f'evaluate_{condition}_{seed}'
            with (exp.OUT/'logs'/(name+'.log')).open('x',encoding='utf-8') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
                exp.evaluate(seed,condition)
            completed.append(name);gc.collect()
            import torch
            torch.cuda.empty_cache()
            exp.write(exp.OUT/'parallel_evaluation_status.json',dict(status='RUNNING',completed=completed,at=exp.now()))
        exp.write(exp.OUT/'parallel_evaluation_status.json',dict(status='COMPLETED',completed=completed,at=exp.now()))
    except BaseException as error:
        import traceback
        exp.write(exp.OUT/'parallel_evaluation_status.json',dict(status='FAILED',completed=completed,at=exp.now(),error=repr(error),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':main()
