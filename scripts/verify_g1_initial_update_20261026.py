"""Recheck saved batches, instrumentation, source seals and the full analysis family."""
from pathlib import Path
import json
import numpy as np
from scipy.stats import t
from statsmodels.stats.multitest import multipletests
from scripts import diagnose_g1_initial_update_20261026 as exp


def main():
    import torch
    torch.set_num_threads(2)
    protocol=exp.lock()
    result=exp.read(exp.OUT/'analysis.json')
    assert result['status']=='COMPLETED' and len(result['episodes'])==36
    import os
    hash_before=hash('g1_initial_update_hash_environment_probe')
    launch_env_seed=os.environ.get('PYTHONHASHSEED')
    os.environ['PYTHONHASHSEED']='20261999'
    hash_after=hash('g1_initial_update_hash_environment_probe')
    assert hash_before==hash_after
    if launch_env_seed is None:os.environ.pop('PYTHONHASHSEED')
    else:os.environ['PYTHONHASHSEED']=launch_env_seed
    # mjlab.seed_rng mutates getenv, which does not reseed an existing Python interpreter.
    # The recovery process restarted after the first capture. Both launches passed 20263200.
    expected_env_seed='20263200';checks=[]
    for index,(arm,seed) in enumerate(protocol['capture_order']):
        if index==1:expected_env_seed='20263200'
        folder=exp.OUT/'captures'/f'{arm}_{seed}'
        m=exp.read(folder/'manifest.json')
        assert exp.sha(folder/'batch.pt')==m['batch_sha256']
        batch=torch.load(folder/'batch.pt',map_location='cpu',weights_only=False)
        for key,value in batch['observations'].items():
            torch.testing.assert_close(value,torch.stack([s[key] for s in batch['pre_step_snapshots']]),rtol=0,atol=0)
        for value in (*batch['observations'].values(),*batch['buffers'].values()):assert torch.isfinite(value).all()
        assert m['pythonhashseed']==expected_env_seed
        expected_env_seed=str(seed)
        assert m['backend']==dict(tf32_matmul=False,cudnn_deterministic=True,cudnn_benchmark=False)
        actual=torch.load(folder/'actual_post.pt',map_location='cpu',weights_only=False)
        native=torch.load(exp.OUT/'replays'/f'{arm}_{seed}_default_fresh/model_0.pt',map_location='cpu',weights_only=False)
        for name in ('actor','critic'):
            for key,v in actual[name].items():torch.testing.assert_close(v,native[name+'_state_dict'][key],rtol=0,atol=0)
        targets={}
        for variant in exp.VARIANTS:
            replay=exp.OUT/'replays'/f'{arm}_{seed}_{variant}'
            meta=exp.read(replay/'manifest.json')
            for file,key in [('model_0.pt','checkpoint_sha256'),('policy.onnx','policy_sha256')]:assert exp.sha(replay/file)==meta[key]
            targets[variant]=torch.load(replay/'targets.pt',map_location='cpu',weights_only=False)
        a,b=targets['default_fresh'],targets['low_lr_fresh']
        for key in ('values','returns','advantages'):torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
        a,b=targets['default_source_critic'],targets['low_lr_source_critic']
        for key in ('values','returns','advantages'):torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
        a=targets['default_fresh']['advantages'].numpy().ravel()
        b=targets['default_source_critic']['advantages'].numpy().ravel()
        checks.append(dict(arm=arm,train_seed=seed,pre_step_observation_max_error=0,
            recorded_vs_unrecorded_model_max_error=0,
            recorded_mutable_hashseed_environment=m['pythonhashseed'],
            normalized_advantage_fresh_source_correlation=float(np.corrcoef(a,b)[0,1]),
            normalized_advantage_sign_agreement=float(((a>=0)==(b>=0)).mean())))
    source_count=0
    for folder in (exp.OUT/'rollouts').iterdir():
        if not folder.is_dir():continue
        m=exp.read(folder/'manifest.json');assert m['status']=='COMPLETED'
        for source,digest in m['source_hashes'].items():
            assert exp.sha(exp.ROOT/source)==digest;source_count+=1
    contrasts=result['contrasts'];assert [r['name'] for r in contrasts]==protocol['primary_family']
    for contrast in contrasts:
        values=np.array([contrast['per_training_seed'][str(seed)] for seed in exp.SEEDS])
        mean=float(values.mean());se=float(values.std(ddof=1)/np.sqrt(3));stat=mean/se
        np.testing.assert_allclose(contrast['paired_t_95pct_ci_mps'],[mean-t.ppf(.975,2)*se,mean+t.ppf(.975,2)*se],atol=1e-12,rtol=0)
        assert abs(contrast['two_sided_p']-2*t.sf(abs(stat),2))<1e-12
        condition=contrast['name'].removesuffix('_minus_default_fresh')
        for seed,value in zip(exp.SEEDS,values):
            row=lambda c:next(r for r in result['aggregates'] if r['train_seed']==seed and r['condition']==c)
            assert abs(value-(row(condition)['mae_forward_velocity_mps']-row('default_fresh')['mae_forward_velocity_mps']))<1e-12
    np.testing.assert_allclose([r['q_bh_two_contrasts'] for r in contrasts],multipletests([r['two_sided_p'] for r in contrasts],method='fdr_bh')[1],atol=1e-12,rtol=0)
    recovery=exp.read(exp.OUT/'technical_recovery_mjlab_config.json')
    amendment=protocol['technical_amendment']
    for key,relative in (
        ('previous_protocol_sha256','prospective_protocol.json'),
        ('failed_manifest_sha256','captures/clock_20261901/manifest.json'),
        ('second_failure_protocol_sha256','local_reconstruction_v2/prospective_protocol.json'),
        ('second_failure_manifest_sha256','local_reconstruction_v2/captures/clock_20261901/manifest.json')):
        assert exp.sha(exp.OUT.parent/relative)==amendment[key]
    assert exp.sha(exp.ROOT/'scripts/run_g1_initial_update_recovery_20261026.py')==recovery['script_sha256']
    assert exp.sha(__import__('mjlab.rl.runner',fromlist=['x']).__file__)==recovery['native_mjlab_runner_sha256']
    parallel=exp.read(exp.OUT/'parallel_evaluation_execution.json')
    assert parallel['jobs']==protocol['evaluation_order'][-4:]
    assert exp.read(exp.OUT/'parallel_evaluation_status.json')['status']=='COMPLETED'
    assert exp.sha(exp.ROOT/'scripts/run_g1_initial_update_eval_parallel_20261026.py')==parallel['script_sha256']
    # Superseded locks are checked against their preserved source snapshots.
    for protocol_folder,snapshot in ((exp.OUT.parent,exp.OUT.parent/'technical_failure_01'),
                                    (exp.OUT.parent/'local_reconstruction_v2',exp.OUT.parent/'technical_failure_02')):
        old=exp.read(protocol_folder/'prospective_protocol.json')
        for row in old['sources']:
            p=Path(row['path']);preserved=snapshot/p.name
            assert exp.sha(preserved if preserved.exists() else p)==row['sha256']
    exp.write(exp.OUT/'sealed_input_recheck.json',dict(status='PASS',at=exp.now(),
        current_protocol_sources=len(protocol['sources']),rollout_source_checks=source_count,
        exact_batch_and_instrumentation_checks=checks,complete_primary_family_checked=True,
        superseded_sources_verified_against_snapshots=True,
        hash_seed_interpretation=dict(process_launch_environment_from_tool_record='20263200',
            manifest_field='Mutable os.getenv, rewritten by mjlab.seed_rng; not the interpreter hash secret',
            mutation_probe_unchanged_hash=True,probe_scope='Post-run demonstration; no original startup hash probe was recorded',
            framework_seed_module_sha256=exp.sha(Path(__import__('mjlab.utils.random',fromlist=['x']).__file__))),
        recovery_script_sha256=recovery['script_sha256'],parallel_execution_sha256=parallel['script_sha256'],
        verifier_sha256=exp.sha(__file__)))
    files=[]
    for p in sorted(exp.OUT.rglob('*')):
        if p.is_file() and p.name not in ('artifact_seal.json','verification_console.log','verification_recovery_console.log'):
            files.append(dict(path=str(p.relative_to(exp.OUT)),sha256=exp.sha(p),bytes=p.stat().st_size))
    exp.write(exp.OUT/'artifact_seal.json',dict(status='PASS',at=exp.now(),files=files,
        limitation='Integrity and numerical checks passed; exploratory statistical evidence remains n=3'))
    print(json.dumps(dict(status='PASS',artifacts=len(files),source_checks=source_count,
        batches=6,formal_episodes=36,contrast_family=2),indent=2),flush=True)


if __name__=='__main__':main()
