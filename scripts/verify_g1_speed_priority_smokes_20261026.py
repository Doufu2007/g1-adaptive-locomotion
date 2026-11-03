"""Compare the two already completed technical smoke runs without new training."""
import hashlib
import json
import numpy as np
import torch
from scripts import diagnose_g1_speed_priority_20261026 as exp
from scripts.audit_g1_training_mechanism_20261026 import forward,longpath,pool


def fingerprint(state):
    digest=hashlib.sha256()
    for key,value in sorted(state.items()):
        digest.update(key.encode());digest.update(np.ascontiguousarray(value.numpy()).tobytes())
    return digest.hexdigest()


def run():
    torch.set_num_threads(2);torch.set_grad_enabled(False)
    folders=[exp.OUT/f'smoke_{i}/smoke/learned_original_reward_20261903' for i in (1,2)]
    records=[exp.read(p/'manifest.json') for p in folders]
    assert all(r['status']=='COMPLETED' and r['additional_completed_updates']==2 for r in records)
    for field in ('source_checkpoint_sha256','initial_actor_sha256','initial_critic_sha256',
                  'initial_learning_rate','continuation_rng_seed','startup_model_hashes'):
        assert records[0][field]==records[1][field]
    states=[torch.load(longpath(p/'model_502.pt'),map_location='cpu',weights_only=False) for p in folders]
    obs,_=pool();actions=[forward(s['actor_state_dict'],obs) for s in states]
    difference=actions[0]-actions[1]
    actor=[fingerprint(s['actor_state_dict']) for s in states]
    critic=[fingerprint(s['critic_state_dict']) for s in states]
    value=dict(status='COMPLETED',script_sha256=exp.sha(__file__),
        source_initial_conditions_equal=True,actor_hashes=actor,critic_hashes=critic,
        final_actor_bitwise_equal=actor[0]==actor[1],final_critic_bitwise_equal=critic[0]==critic[1],
        same_pool_action_rms_difference=float(difference.square().mean().sqrt()),
        same_pool_action_max_abs_difference=float(difference.abs().max()),
        scope='Two existing2-update smoke runs in independent directories and newly created environments, '
              'same process/session; no claim of independent-process, full100-update, or physical bitwise reproducibility.')
    exp.write(exp.OUT/'smoke_repeat_verification.json',value)
    print(json.dumps(value,indent=2),flush=True)


if __name__=='__main__':
    run()
