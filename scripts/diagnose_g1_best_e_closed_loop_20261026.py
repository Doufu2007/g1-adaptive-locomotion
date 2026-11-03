"""Read-only replay and isolated interventions for the sealed best-E study."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
ORIGINAL = ROOT/'results/g1_stateful_flat_ppo_best_e_20261026'
SNAPSHOT = ORIGINAL/'remote_download/snapshot'
OUT = ROOT/'results/g1_best_e_cause_audit_20261026'
ENCODER = ROOT/'deliveries/g1_stateful_phase_20261025/snapshot/results/g1_stateful_phase_20261025/training/observer_sensors_mixed_seed20261026/final.pt'
ENCODER_SHA = '8e6b15dcf409f51948f49c18fc04b686ba502e53586476db9c78d8c35a0fc346'


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(name, value):
    path = OUT/name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def circular_stats(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return None
    mean = np.angle(np.exp(1j*values).mean())
    return dict(mean_rad=float(mean), concentration=float(abs(np.exp(1j*values).mean())),
                mae_rad=float(np.abs(np.angle(np.exp(1j*values))).mean()),
                aligned_mae_rad=float(np.abs(np.angle(np.exp(1j*(values-mean)))).mean()))


def rotate_phase_vectors(phase, offset):
    result = phase.new_empty(phase.shape)
    c, s = float(np.cos(offset)), float(np.sin(offset))
    result[..., 0] = phase[..., 0]*c + phase[..., 1]*s
    result[..., 1] = phase[..., 1]*c - phase[..., 0]*s
    return result


def checkpoint_policy(arm, seed, step):
    folder = SNAPSHOT/'results/g1_stateful_flat_ppo_best_e_20261026/training'/f'flat_{arm}_seed{seed}'
    manifest = json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    if step == 500:
        policy = folder/'policy.onnx'
        assert sha(policy) == manifest['policy_sha256']
        return policy
    if step != 250:
        raise ValueError('diagnostic checkpoint must be predeclared: 250 or 500')
    import torch
    from tensordict import TensorDict
    from rsl_rl.models import MLPModel
    import onnxruntime as ort
    source = folder/f'model_{step}.pt'
    export_manifest = json.loads((ORIGINAL/'remote_download/export_manifest.json').read_text(encoding='utf-8'))
    expected = next(r['sha256'] for r in export_manifest['files']
                    if r['path'] == source.relative_to(SNAPSHOT).as_posix())
    assert sha(source) == expected
    target = OUT/'exported'/f'{arm}_{seed}_{step}'
    policy = target/'policy.onnx'
    if policy.exists():
        old = json.loads((target/'parity.json').read_text(encoding='utf-8'))
        assert old['checkpoint_sha256'] == expected and old['onnx_sha256'] == sha(policy)
        return policy
    target.mkdir(parents=True,exist_ok=False)
    saved = torch.load(source,map_location='cpu',weights_only=False)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20262600)
        model = MLPModel(TensorDict({'actor':torch.zeros(1,98)},batch_size=[1]),
            {'actor':['actor']},'actor',29,hidden_dims=(512,256,128),activation='elu',
            obs_normalization=True,distribution_cfg=dict(class_name='GaussianDistribution',init_std=1.,std_type='scalar'))
        model.load_state_dict(saved['actor_state_dict'],strict=True)
        model.eval()
        exported = model.as_onnx(verbose=False).eval()
        torch.onnx.export(exported,exported.get_dummy_inputs(),str(policy),export_params=True,
            opset_version=18,input_names=exported.input_names,output_names=exported.output_names,
            dynamic_axes={},dynamo=False)
        values = model.obs_normalizer._mean + torch.randn(64,98)*model.obs_normalizer._std
        with torch.inference_mode():
            reference = model(TensorDict({'actor':values},batch_size=[64]),stochastic_output=False).numpy()
        options = ort.SessionOptions(); options.intra_op_num_threads=options.inter_op_num_threads=1
        session = ort.InferenceSession(str(policy),sess_options=options,providers=['CPUExecutionProvider'])
        actual = np.concatenate([session.run(None,{'obs':x[None].numpy()})[0] for x in values])
        np.testing.assert_allclose(actual,reference,rtol=2e-4,atol=2e-5)
    write(target.relative_to(OUT)/'parity.json',dict(status='PASS',checkpoint_sha256=expected,
        onnx_sha256=sha(policy),max_abs_error=float(np.abs(actual-reference).max()),step=step,
        completed_updates=step+1,actor_normalization_preserved=True,random_inputs=64))
    return policy


def local_rollout(args):
    """Run existing frozen-policy evaluator with an isolated phase rotation."""
    from dataclasses import replace
    import importlib.metadata
    from types import SimpleNamespace
    from scripts import run_g1_best_phase_control as base
    native_bootstrap = base.bootstrap
    from scripts.run_g1_best_e_ppo_20261026 import install
    install()
    from scripts import g1_stateful_flat_ppo as study
    from scripts import g1_best_phase_batched as batch
    # The original Windows bootstrap handles Unicode/cache paths and keeps GLFW.
    study.base.bootstrap = native_bootstrap
    study.previous.bootstrap = native_bootstrap
    np, torch = study.bootstrap()
    from scripts.g1_stateful_online import learned_phase
    from src.tasks.velocity.mdp import observations
    policy = checkpoint_policy(args.trained,args.train_seed,args.checkpoint)
    assert sha(ENCODER) == ENCODER_SHA
    if args.phase != 'aligned' and args.offset != 0:
        raise ValueError('nonzero rotation requires aligned treatment')
    offset = float(args.offset)
    def aligned_phase(env, encoder_path):
        return rotate_phase_vectors(learned_phase(env, encoder_path), offset)
    def configuration(terrain, arm, seed, **kwargs):
        cfg = study.make_cfg(terrain, arm, seed, **kwargs)
        if args.phase == 'aligned':
            for group in ('actor','critic'):
                cfg.observations[group].terms['phase'] = replace(
                    cfg.observations[group].terms['phase'], func=aligned_phase)
        return cfg
    previous_phase = observations.learned_phase
    if args.phase == 'aligned':
        observations.learned_phase = aligned_phase
    batch.bootstrap = study.bootstrap
    batch.make_cfg = configuration
    batch.ENCODER = ENCODER
    base.ENCODER = ENCODER
    target = OUT/'rollouts'/args.name
    if target.exists():
        raise FileExistsError('preserve previous attempt: '+str(target))
    provenance = dict(at=now(), phase=args.phase, phase_offset_rad=offset,
        trained=args.trained, train_seed=args.train_seed, checkpoint=args.checkpoint,seeds=args.eval_seeds,
        policy_sha256=sha(policy), encoder_sha256=ENCODER_SHA,
        script_sha256=sha(__file__), gpu=torch.cuda.get_device_name(),
        packages={n:importlib.metadata.version(n) for n in
                  ('torch','mjlab','mujoco','warp-lang','rsl-rl-lib','onnxruntime')})
    write(Path('launches')/(args.name+'.json'), provenance)
    try:
        batch.rollout_batched(SimpleNamespace(output=str(target),policy=str(policy),
            terrain='flat',arm='clock' if args.phase=='clock' else 'learned',
            label=args.name,seeds=args.eval_seeds,horizon=args.horizon,scenarios=args.scenarios))
    finally:
        observations.learned_phase = previous_phase
    path = target/'manifest.json'
    manifest = json.loads(path.read_text(encoding='utf-8'))
    manifest.update(cause_audit=provenance, original_study_modified=False)
    path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def calibrate():
    import torch
    from scripts.g1_stateful_phase import PhaseNet,causal_features
    model = PhaseNet(64,'observer').eval()
    checkpoint = torch.load(ENCODER,map_location='cpu',weights_only=False)
    model.load_state_dict(checkpoint['model'],strict=True)
    rows, sums = [], []
    for seed in (20261901,20261902,20261903):
        folder = OUT/'rollouts'/f'development_clock_{seed}'
        episodes = json.loads((folder/'episodes.json').read_text(encoding='utf-8'))
        for episode in episodes:
            path = folder/episode['raw_path']
            assert sha(path) == episode['raw_sha256']
            with np.load(path,allow_pickle=False) as raw:
                values=(causal_features(raw['joint_pos'],raw['quaternion'])-checkpoint['mean'])/checkpoint['std']
                with torch.inference_mode():
                    phase=model(torch.from_numpy(values)[None])['unwrapped'][0].numpy()
                mask=raw['time']>=3.
                if not mask.any():
                    raise ValueError('development trajectory lacks calibration history')
                differences=phase[mask]-raw['angle'][mask]
                sums.append(np.exp(1j*differences).mean())
                rows.append(dict(train_seed=seed,eval_seed=episode['seed'],scenario=episode['scenario'],
                    raw_sha256=episode['raw_sha256'],E_minus_clock=circular_stats(differences)))
    correction=-float(np.angle(np.mean(sums)))
    result=dict(status='LOCKED_BEFORE_HOLDOUT',at=now(),phase_offset_rad=correction,
        method='negative circular mean of E-minus-clock; equal episode weights; after 3 seconds',
        development_eval_seeds=[20262601,20262602,20262603],
        holdout_eval_seeds=list(range(20262611,20262621)),rows=rows,
        script_sha256=sha(__file__),encoder_sha256=ENCODER_SHA)
    write('calibration.json',result)
    return correction


def queue():
    import contextlib
    import gc
    import traceback
    from types import SimpleNamespace
    from scripts import run_g1_best_phase_control as base
    native_bootstrap=base.bootstrap
    status_path='queue_status.json'
    completed=[]
    def run(name,trained,seed,phase,seeds,offset=0.,checkpoint=500,horizon=20.,scenarios=None):
        directory=OUT/'rollouts'/name
        if (directory/'manifest.json').exists():
            manifest=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
            if manifest['status']=='COMPLETED':
                completed.append(name)
                return
            raise FileExistsError('incomplete diagnostic preserved: '+name)
        write(status_path,dict(status='RUNNING',at=now(),current=name,completed=completed))
        (OUT/'logs').mkdir(parents=True,exist_ok=True)
        base.bootstrap=native_bootstrap
        args=SimpleNamespace(name=name,trained=trained,train_seed=seed,phase=phase,
            eval_seeds=seeds,offset=offset,checkpoint=checkpoint,horizon=horizon,
            scenarios=scenarios or ['steady','speed_change','push'])
        with (OUT/'logs'/(name+'.log')).open('w',encoding='utf-8') as stream:
            with contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
                local_rollout(args)
        completed.append(name)
        print(json.dumps(dict(completed=name,groups_done=len(completed))),flush=True)
        gc.collect()
    try:
        run('smoke_aligned','clock',20261901,'aligned',[20262601],.2,horizon=.2,scenarios=['steady'])
        for seed in (20261901,20261902,20261903):
            run(f'development_clock_{seed}','clock',seed,'clock',[20262601,20262602,20262603])
        if (OUT/'calibration.json').exists():
            calibration=json.loads((OUT/'calibration.json').read_text(encoding='utf-8'))
            if calibration['script_sha256'] != sha(__file__):
                raise ValueError('script changed after calibration lock')
            offset=calibration['phase_offset_rad']
        else:
            offset=calibrate()
        held=list(range(20262611,20262621))
        treatments=[('clock','clock'),('clock','raw'),('clock','aligned'),
                    ('learned','raw'),('learned','aligned')]
        for seed in (20261901,20261902,20261903):
            for trained,phase in treatments:
                run(f'holdout_{trained}_{phase}_{seed}',trained,seed,phase,held,
                    offset if phase=='aligned' else 0.)
        for seed in (20261901,20261902,20261903):
            for trained,phase in [('clock','clock'),('learned','raw')]:
                run(f'midpoint_{trained}_{seed}',trained,seed,phase,held,checkpoint=250)
        write(status_path,dict(status='COMPLETED',at=now(),completed=completed,
            phase_offset_rad=offset,holdout_episodes=630,development_episodes=27))
    except BaseException as error:
        write(status_path,dict(status='FAILED',at=now(),completed=completed,
            error=repr(error),traceback=traceback.format_exc()))
        raise


def canonical_phase_score(angle, contacts, *, debounce=3):
    from scripts.g1_stateful_phase import contact_labels
    from src.tasks.velocity.phasor.phase_gate_v4_2 import score_phase_cycles_v4_2
    target, left, right, cycles = contact_labels(contacts, debounce=debounce)
    reference = target.copy()
    reference[left] = 0.
    frames = np.arange(len(angle))
    report = score_phase_cycles_v4_2(
        np.stack((np.sin(angle), np.cos(angle)), axis=-1), reference,
        np.full(len(angle), 'episode'), frames, left,
        reference_valid=np.isfinite(reference), period_cuts_frames=(50.66666666666666, 75.))
    valid = [r for r in report['cycle_records'] if r['evaluable']]
    consistent = [r for r in valid if right[r['start_frame']:r['end_frame']].sum() == 1]
    post3 = [r for r in valid if r['start_frame'] >= 150]
    point = np.isfinite(target) & (frames >= 150)
    error = np.angle(np.exp(1j*(np.asarray(angle)-target)))
    return dict(observed=report['observed_cycle_count'], evaluable=len(valid),
        completed=sum(r['encoder_completed_one_turn'] for r in valid),
        completion=report['encoder_micro_cycle_completion_fraction'],
        consistent_cycles=len(consistent), consistent_completion=float(np.mean([
            r['encoder_completed_one_turn'] for r in consistent])) if consistent else None,
        after_3s_cycles=len(post3), after_3s_completion=float(np.mean([
            r['encoder_completed_one_turn'] for r in post3])) if post3 else None,
        contact_phase_error=circular_stats(error[point]))


def saved_analysis():
    import torch
    from scripts.g1_stateful_phase import PhaseNet, causal_features
    torch.set_num_threads(2)
    torch.manual_seed(20262600)
    assert sha(ENCODER) == ENCODER_SHA
    checkpoint = torch.load(ENCODER, map_location='cpu', weights_only=False)
    model = PhaseNet(64, 'observer').eval().requires_grad_(False)
    model.load_state_dict(checkpoint['model'], strict=True)
    comparison = json.loads((ORIGINAL/'comparison.json').read_text(encoding='utf-8'))
    rows = comparison['episodes']
    results = []
    write('saved_analysis_status.json', dict(status='RUNNING', at=now(), total=len(rows)))
    for start in range(0, len(rows), 10):
        entries = rows[start:start+10]
        raws, features, sources = [], [], []
        for row in entries:
            source = SNAPSHOT/row['raw_path']
            assert sha(source) == row['raw_sha256'], source
            with np.load(source, allow_pickle=False) as saved:
                raw = {k:saved[k] for k in saved.files}
            raws.append(raw)
            sources.append(source)
            features.append((causal_features(raw['joint_pos'], raw['quaternion'])-
                             checkpoint['mean'])/checkpoint['std'])
        if len({len(f) for f in features}) != 1:
            raise ValueError('batch must contain equal-length trajectories')
        with torch.inference_mode():
            output = model(torch.from_numpy(np.stack(features)))
        angles = output['unwrapped'].numpy()
        for i, (row, raw, z, source) in enumerate(zip(entries, raws, features, sources)):
            phase = angles[i]
            replay_error = np.abs(np.angle(np.exp(1j*(phase-raw['angle']))))
            point = raw['time'] >= 3.
            scores = {str(n):canonical_phase_score(phase, raw['contact'], debounce=n)
                      for n in (1, 3, 5)}
            logged = canonical_phase_score(raw['angle'], raw['contact'], debounce=3)
            clock = 2*np.pi*raw['time']/.6
            result = dict(train_seed=row['train_seed'], label=row['label'],
                scenario=row['scenario'], eval_seed=row['seed'],
                raw_path=row['raw_path'], raw_sha256=row['raw_sha256'],
                original_one_turn_fraction=row['phase']['one_turn_fraction'],
                velocity_mae_mps=row['mae_forward_velocity_mps'],
                recorded_vs_replayed_E_max_angle_rad=float(replay_error.max())
                    if row['label'] != 'clock_clock' else None,
                E_scores_by_debounce=scores, recorded_phase_canonical=logged,
                E_minus_recorded_clock=circular_stats(phase[point]-raw['angle'][point])
                    if row['label'] == 'clock_clock' else None,
                E_minus_fixed_clock=circular_stats(phase[point]-clock[point]),
                normalized_features=dict(fraction_abs_gt_3=float((np.abs(z[point])>3).mean()),
                    fraction_abs_gt_5=float((np.abs(z[point])>5).mean()),
                    rms_by_group={name:float(np.sqrt(np.mean(z[point, lo:hi]**2)))
                        for name,lo,hi in [('position',0,29),('joint_velocity',29,58),
                                           ('gravity',58,61),('angular_velocity',61,64)]}))
            results.append(result)
        write('saved_analysis_status.json', dict(status='RUNNING', at=now(), total=len(rows), completed=len(results)))
        print(json.dumps(dict(stage='saved_analysis', completed=len(results), total=len(rows))), flush=True)
    grouped = []
    for seed in (20261901,20261902,20261903):
        for label in ('clock_clock','clock_learned','learned_learned'):
            selected = [r for r in results if r['train_seed']==seed and r['label']==label]
            group = dict(train_seed=seed, label=label, episodes=len(selected),
                velocity_mae_mps=float(np.mean([r['velocity_mae_mps'] for r in selected])),
                E_replay_max_error_rad=max((r['recorded_vs_replayed_E_max_angle_rad'] or 0.) for r in selected),
                E_macro_completion_by_debounce={str(n):float(np.mean([
                    r['E_scores_by_debounce'][str(n)]['completion'] for r in selected])) for n in (1,3,5)},
                recorded_phase_canonical_macro_completion=float(np.mean([
                    r['recorded_phase_canonical']['completion'] for r in selected])),
                E_macro_contact_phase_mae_rad=float(np.mean([
                    r['E_scores_by_debounce']['3']['contact_phase_error']['mae_rad'] for r in selected])),
                E_macro_aligned_contact_phase_mae_rad=float(np.mean([
                    r['E_scores_by_debounce']['3']['contact_phase_error']['aligned_mae_rad'] for r in selected])),
                fraction_features_abs_gt_5=float(np.mean([
                    r['normalized_features']['fraction_abs_gt_5'] for r in selected])))
            grouped.append(group)
            print(json.dumps(group), flush=True)
    write('saved_analysis.json', dict(status='COMPLETED', at=now(), encoder_sha256=ENCODER_SHA,
        comparison_sha256=sha(ORIGINAL/'comparison.json'), all_270_raw_files_verified=True,
        canonical_gate_sha256=sha(ROOT/'src/tasks/velocity/phasor/phase_gate_v4_2.py'),
        causal_attribution=False, rows=results, groups=grouped))
    write('saved_analysis_status.json', dict(status='COMPLETED', at=now(), total=len(rows), completed=len(results)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['saved-analysis','rollout','queue'], required=True)
    parser.add_argument('--name', default='smoke_clock')
    parser.add_argument('--trained', choices=['clock','learned'], default='clock')
    parser.add_argument('--train-seed',type=int,default=20261901)
    parser.add_argument('--checkpoint',type=int,default=500)
    parser.add_argument('--phase', choices=['clock','raw','aligned'], default='clock')
    parser.add_argument('--offset',type=float,default=0.)
    parser.add_argument('--eval-seeds',type=int,nargs='+',default=[20262601])
    parser.add_argument('--scenarios',nargs='+',default=['steady','speed_change','push'])
    parser.add_argument('--horizon',type=float,default=20.)
    args = parser.parse_args()
    if args.stage == 'saved-analysis':
        saved_analysis()
    elif args.stage=='rollout':
        local_rollout(args)
    else:
        queue()
