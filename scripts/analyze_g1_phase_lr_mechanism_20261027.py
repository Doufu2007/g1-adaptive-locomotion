"""Frozen fixed-state action/parameter drift supplement; no policy fitting."""
import numpy as np
import torch
from scripts import g1_phase_lr_factorial_20261027 as study
from scripts.audit_g1_training_mechanism_20261026 import pool,forward


def main():
    torch.set_num_threads(2);torch.set_grad_enabled(False);study.lock()
    seal=study.read(study.OUT/'mechanism_source_seal.json')
    for row in seal['files']:assert study.sha(row['path'])==row['sha256']
    obs,sources=pool();rows=[]
    for job in study.build_jobs():
        folder=study.OUT/'training'/f"{job['arm']}_{job['lr_mode']}_{job['seed']}"
        manifest=study.read(folder/'manifest.json');assert manifest['status']=='COMPLETED'
        baseline=None;previous=None
        for point in study.CHECKPOINTS:
            item=manifest['checkpoints'][str(point)];assert study.sha(item['path'])==item['sha256']
            saved=torch.load(item['path'],map_location='cpu',weights_only=False);state=saved['actor_state_dict']
            action=forward(state,obs);sigma=state['distribution.std_param']
            assert torch.isfinite(action).all() and (sigma>0).all()
            if baseline is None:baseline=(state,action,sigma)
            old,old_action,old_sigma=baseline
            names=[k for k in state if k.startswith('mlp.')]
            parameter_rms=float((sum((state[k]-old[k]).square().sum() for k in names)/sum(state[k].numel() for k in names)).sqrt())
            kl=(torch.log(sigma/old_sigma)+(old_sigma.square()+(old_action-action).square())/(2*sigma.square())-.5).sum(-1)
            shifted=obs.clone();alpha=np.pi/12;s,c=obs[:,9].clone(),obs[:,10].clone()
            shifted[:,9]=s*np.cos(alpha)+c*np.sin(alpha);shifted[:,10]=c*np.cos(alpha)-s*np.sin(alpha)
            phase_change=float((forward(state,shifted)-action).square().mean().sqrt())
            commanded=obs.clone();commanded[:,6]+=.1
            command_change=float((forward(state,commanded)-action).square().mean().sqrt())
            rows.append(dict(arm=job['arm'],lr_mode=job['lr_mode'],seed=job['seed'],completed_updates=point,
                checkpoint_sha256=item['sha256'],fixed_state_action_rms_from_initial=float((action-old_action).square().mean().sqrt()),
                fixed_state_action_rms_from_previous=float((action-previous).square().mean().sqrt()) if previous is not None else 0.,
                mlp_parameter_rms_from_initial=parameter_rms,fixed_state_conditional_KL_from_initial=float(kl.mean()),
                phase_plus15deg_action_rms=phase_change,command_plus_point1_action_rms=command_change,
                normalizer_count=int(state['obs_normalizer.count']),std_mean=float(sigma.mean())))
            previous=action
    study.write(study.OUT/'mechanism_supplement.json',dict(status='COMPLETED',at=study.now(),script_sha256=study.sha(__file__),
        states=len(obs),pool_sources=sources,rows=rows,
        limitation='Conditional actions on a fixed old state pool; not actual training KL or a causal deployment intervention.'))

if __name__=='__main__':main()
