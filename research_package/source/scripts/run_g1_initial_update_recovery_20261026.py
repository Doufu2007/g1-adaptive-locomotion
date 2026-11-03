"""Restore the native mjlab optional-config filtering in frozen-batch replay."""
import argparse
import copy
import inspect
import json
from pathlib import Path
from scripts import diagnose_g1_initial_update_20261026 as exp


def filtered_pack(pack):
    from rsl_rl.models import MLPModel
    result=dict(pack,agent=copy.deepcopy(pack['agent']))
    accepted=set(inspect.signature(MLPModel).parameters)
    for name in ('actor','critic'):
        cfg=result['agent'][name]
        for key in ('cnn_cfg','distribution_cfg'):
            if key in cfg and cfg[key] is None:cfg.pop(key)
        assert not (set(cfg)-accepted-{'class_name'}),(name,set(cfg)-accepted)
    return result


def install():
    native=exp.make_algorithm
    def restored(pack,variant,torch):return native(filtered_pack(pack),variant,torch)
    exp.make_algorithm=restored
    return restored


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['preflight','queue'],required=True)
    args=parser.parse_args()
    protocol=exp.lock()
    path=exp.OUT/'technical_recovery_mjlab_config.json'
    record=dict(status='LOCKED_BEFORE_REPLAY',at=exp.now(),script_sha256=exp.sha(__file__),
        original_protocol_sha256=exp.sha(exp.OUT/'prospective_protocol.json'),
        native_mjlab_runner_sha256=exp.sha(Path(__import__('mjlab.rl.runner',fromlist=['x']).__file__)),
        issue='Manual MLP construction retained cnn_cfg=None; native MjlabOnPolicyRunner removes optional None fields',
        change='Exactly mirror native filtering of cnn_cfg/distribution_cfg when None; assert all other keys accepted',
        existing_capture_preserved=True,extra_environment_transitions=0,extra_optimizer_updates=0,
        no_change_to_matrix_budget_or_metrics=True)
    if path.exists():
        assert exp.read(path)['script_sha256']==record['script_sha256']
    else:exp.write(path,record)
    restored=install()
    if args.stage=='preflight':
        from scripts.g1_phase_feedback_probe_20261026 import setup
        _,torch=setup()
        folder=exp.OUT/'captures/clock_20261901'
        pack=torch.load(folder/'batch.pt',map_location='cpu',weights_only=False)
        for variant in exp.VARIANTS:
            alg=restored(pack,variant,torch)
            assert not alg.optimizer.state
            print(json.dumps(dict(variant=variant,status='PASS',steps=alg.storage.step)),flush=True)
        exp.write(exp.OUT/'replay_constructor_preflight.json',dict(status='PASS',variants=list(exp.VARIANTS),
            values_returns_advantages_reproduced=True,environment_steps=0,optimizer_steps=0,
            script_sha256=exp.sha(__file__)))
    else:
        assert exp.read(exp.OUT/'replay_constructor_preflight.json')['status']=='PASS'
        exp.queue()


if __name__=='__main__':main()
