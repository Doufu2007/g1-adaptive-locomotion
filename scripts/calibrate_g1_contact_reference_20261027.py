"""One predeclared source-policy training-only calibration; zero PPO updates."""
from dataclasses import asdict
import argparse,json,random,traceback
from pathlib import Path
from scripts import g1_phase_lr_factorial_20261027 as a
from scripts.g1_contact_phase_reference import calibrate_trajectory
OUT=a.ROOT/'results/g1_contact_phase_reference_20261027'

def collect():
    lock=a.read(OUT/'calibration_collection_lock.json')
    for row in lock['files']:assert a.sha(row['path'])==row['sha256']
    assert not (OUT/'calibration_collection_manifest.json').exists(),'preserve previous attempt'
    study,torch=a.setup()
    import numpy as np
    import onnxruntime as ort
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.utils.os import dump_yaml
    from src.tasks.velocity.mdp import observations
    seed=20263501;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    cfg=study.make_cfg('flat','clock',seed,play=False,num_envs=64,horizon=20.)
    dump_yaml(OUT/'calibration_collection_env.yaml',asdict(cfg))
    source=a.OUT/'training/clock_low_20261901/policy_0.onnx'
    source_manifest=a.read(a.OUT/'training/clock_low_20261901/manifest.json')
    assert a.sha(source)==source_manifest['checkpoints']['0']['policy_sha256']
    options=ort.SessionOptions();options.intra_op_num_threads=options.inter_op_num_threads=1
    policy=ort.InferenceSession(str(source),sess_options=options,providers=['CPUExecutionProvider'])
    rec=dict(status='RUNNING',at=a.now(),seed=seed,steps=0,transitions=0,planned_steps=500,planned_transitions=32000,
        optimizer_updates=0,policy_sha256=a.sha(source),source_actor_sha256=a.SOURCE_SHA,lock_sha256=a.sha(OUT/'calibration_collection_lock.json'),
        phase_origin_calibration_only=True,not_formal_control_evaluation=True)
    a.write(OUT/'calibration_collection_manifest.json',rec);env=None
    try:
        env=ManagerBasedRlEnv(cfg=cfg,device='cuda:0');env.reset()
        assert abs(float(env.step_dt)-.02)<1e-9
        rec['startup_model_hashes']={k:__import__('hashlib').sha256(np.ascontiguousarray(getattr(env.sim.model,k).cpu().numpy()).tobytes()).hexdigest() for k in ('body_mass','body_ipos','geom_friction')}
        phases=[];contacts=[];done=[]
        for step in range(500):
            obs=env.observation_manager.compute()['actor'].detach().cpu().numpy().astype(np.float32)
            phase=observations.phase(env,.6,'twist').detach().cpu()
            np.testing.assert_allclose(obs[:,9:11],phase.numpy(),atol=1e-7)
            phases.append(phase);contacts.append((env.scene['feet_ground_contact'].data.found>0).detach().cpu())
            action=np.concatenate([policy.run(None,{policy.get_inputs()[0].name:obs[i:i+1]})[0] for i in range(64)])
            result=env.step(torch.as_tensor(action,device=env.device));done.append((result[2]|result[3]).detach().cpu())
            rec.update(steps=step+1,transitions=(step+1)*64)
        phase=torch.stack(phases);contact=torch.stack(contacts);dones=torch.stack(done)
        raw=OUT/'calibration_collection_raw.npz';np.savez_compressed(raw,phase=phase.numpy(),contact=contact.numpy(),dones=dones.numpy())
        value=calibrate_trajectory(phase,contact,dones,150);value.update(raw_path=str(raw),raw_sha256=a.sha(raw),collection_lock_sha256=a.sha(OUT/'calibration_collection_lock.json'))
        a.write(OUT/'calibration_continuous.json',value)
        rec.update(status='COMPLETED',ended_at=a.now(),calibration_passed=value['passed'],raw_sha256=a.sha(raw))
    except BaseException as error:
        rec.update(status='FAILED',ended_at=a.now(),error=repr(error),traceback=traceback.format_exc());raise
    finally:
        a.write(OUT/'calibration_collection_manifest.json',rec)
        if env is not None:env.close()

if __name__=='__main__':collect()
