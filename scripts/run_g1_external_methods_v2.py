"""Explicit training-only amendment: paper warmup and raw state loss for TCTST.

The original frozen code/protocol/results stay intact. Compatible completed
TCN/Modern runs are referenced, never relabeled as having the new protocol.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import time

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_g1_external_methods as base
from scripts.g1_external_models import TCTSTPhase, masked_reconstruction_loss, phase_rate_output

OLD_OUT=base.OUT
NEW_OUT=OLD_OUT.parent/'run_v2'
ORIGINAL_FIT=base.fit
ORIGINAL_MODEL=base.make_model
ORIGINAL_RUN=base.run_one


def warmup_scale(epoch):
    """Paper's first 20 epochs: cosine ramp from .2 to 1, then stay at 1."""
    return .2+.8*(1-math.cos(math.pi*min(epoch/19.,1.)))/2


class RawStateTCTST(TCTSTPhase):
    def raw_state(self,x):
        sensor=x.transpose(1,2)
        joint=torch.cat((sensor,sensor.new_zeros(len(x),3,self.window)),dim=1)
        raw=self.head(self.encode_joint(joint).mean(dim=1)).float()
        return torch.cat((raw[:,:2],F.softplus(raw[:,2:3])),dim=-1)

    def forward(self,x):
        raw=self.raw_state(x)
        return torch.cat((F.normalize(raw[:,:2],dim=-1,eps=1e-8),raw[:,2:3]),dim=-1)


def make_model(arm):
    return (RawStateTCTST(),100) if arm.startswith('tctst') else ORIGINAL_MODEL(arm)


def state_loss(raw,target,rate,valid):
    return .5*(raw[:,:2]-target).square().sum(dim=-1).mean()+.02*((raw[:,2]-rate).square()*valid).sum()/valid.sum().clamp_min(1)


def fit(model,bank,ids,run_dir,phase,seed,lr,window,micro,epochs=32,cache=None,pretrainer=None):
    if not isinstance(model,RawStateTCTST):
        return ORIGINAL_FIT(model,bank,ids,run_dir,phase,seed,lr,window,micro,epochs,cache,pretrainer)
    params=[p for p in (pretrainer if pretrainer is not None else model).parameters() if p.requires_grad]
    optimizer=torch.optim.Adam(params,lr=lr)
    history=[]
    for epoch in range(epochs):
        started=time.perf_counter();model.train()
        if pretrainer is not None:pretrainer.train()
        actual_lr=lr*warmup_scale(epoch)
        for group in optimizer.param_groups:group['lr']=actual_lr
        generator=torch.Generator().manual_seed(seed+1009*epoch)
        order=ids[torch.randperm(len(ids),generator=generator).to(base.DEVICE)]
        loss_sum=0.;seen=0
        for start in range(0,len(order),512):
            effective=order[start:start+512];optimizer.zero_grad(set_to_none=True)
            for small in range(0,len(effective),micro):
                ii=effective[small:small+micro]
                with base.amp():
                    if phase=='masked_pretrain':
                        joint,valid=bank.pretraining_joint(ii,window)
                        masked=torch.zeros(len(ii),32,dtype=torch.bool,device=base.DEVICE)
                        masked.scatter_(1,torch.rand(len(ii),32,device=base.DEVICE).topk(10,dim=1).indices,True)
                        reconstructed=pretrainer(joint.masked_fill(masked[:,:,None],0))
                        loss=masked_reconstruction_loss(reconstructed.float(),joint,masked,valid)
                    else:
                        raw=model.raw_state(bank.windows(ii,window))
                        loss=state_loss(raw,bank.target[ii],bank.rate_tensor[ii],bank.rate_mask[ii])
                if not torch.isfinite(loss):raise FloatingPointError(f'nonfinite loss at {phase}/{epoch}')
                (loss*(len(ii)/len(effective))).backward()
                loss_sum+=float(loss.detach())*len(ii);seen+=len(ii)
            norm=nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
            optimizer.step()
        if base.DEVICE.type=='cuda':torch.cuda.synchronize()
        row={'phase':phase,'epoch':epoch+1,'loss':loss_sum/seen,'samples':seen,
             'seconds':time.perf_counter()-started,'grad_norm_last':float(norm),'learning_rate':actual_lr}
        history.append(row)
        with (run_dir/'epochs.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
        if (epoch+1)%8==0 or epoch+1==epochs:
            torch.save({'phase':phase,'epoch':epoch+1,'model':model.state_dict(),
                'pretrainer':pretrainer.state_dict() if pretrainer is not None else None,
                'optimizer':optimizer.state_dict(),'torch_rng':torch.get_rng_state(),
                'cuda_rng':torch.cuda.get_rng_state_all() if base.DEVICE.type=='cuda' else None},run_dir/'partial_checkpoint.pt')
    return history


def freeze_protocol():
    NEW_OUT.mkdir(parents=True,exist_ok=True)
    original=base.json.loads((OLD_OUT/'protocol.json').read_text(encoding='utf-8'))
    # All original code remains byte-for-byte frozen and compatible carried runs
    # retain their own manifest/protocol identity.
    for path,value in original['code_hashes'].items():
        if base.digest(ROOT/path)!=value:raise ValueError('original frozen code changed')
    for path,value in original['source_hashes'].items():
        if base.digest(ROOT/path)!=value:raise ValueError('original frozen data changed')
    hashes={**original['code_hashes'],str(Path(__file__).relative_to(ROOT)):base.digest(__file__)}
    carried={}
    for path in sorted(OLD_OUT.glob('*/completed.json')):
        run=base.json.loads(path.read_text(encoding='utf-8'))
        if run['arm'] not in ('tcn10','modern10','tcn100_rate'):raise ValueError('cannot carry TCTST/FLD across amendment')
        if run['protocol_sha256']!=base.digest(OLD_OUT/'protocol.json'):raise ValueError('carried protocol mismatch')
        attempt=ROOT/run['attempt']
        for artifact,value in run['artifacts'].items():
            if base.digest(attempt/artifact)!=value:raise ValueError('carried artifact changed')
        carried[run['run_id']]={'manifest_sha256':base.digest(path),'original_protocol_sha256':run['protocol_sha256'],'path':str(path.relative_to(ROOT))}
    value={**original,'schema':'g1.external_methods.protocol.v2','frozen_at':base.now(),
        'code_hashes':hashes,'parent_protocol_path':str((OLD_OUT/'protocol.json').relative_to(ROOT)),
        'parent_protocol_sha256':base.digest(OLD_OUT/'protocol.json'),'carried_runs':carried,
        'amendment':{'reason':'TCTST training loss remained approximately 1 through 25 epochs; paper warmup and raw-output MSE were omitted in v1 adaptation',
            'evidence_used':'training logs and paper only; no TCTST validation predictions or scores generated',
            'changes':['TCTST supervised raw sin/cos half-squared-error plus unchanged .02 rate MSE',
                'TCTST both phases cosine LR ramp .2 to 1 over first20 epochs, fixed base LR thereafter'],
            'unchanged':'seeds,32epoch budgets,batches,data,backbone,all evaluation thresholds,Modern/TCN/FLD training',
            'v1_attempt_retained':'tctst_scratch100_seed_20261021/attempt_01; stopped after25 epochs, no completed checkpoint/validation scoring',
            'limits':'still 32epochs, joint inputs and adapted loss weights; not complete original-paper reproduction'}}
    path=NEW_OUT/'protocol.json'
    if path.exists():
        existing=base.json.loads(path.read_text(encoding='utf-8'))
        for key in ('code_hashes','source_hashes','parent_protocol_sha256','carried_runs'):
            if existing[key]!=value[key]:raise ValueError(f'frozen v2 {key} changed')
        value=existing
    else:
        base.write_json(path,value)
    for run_id,entry in value['carried_runs'].items():
        destination=NEW_OUT/run_id/'completed.json';destination.parent.mkdir(exist_ok=True)
        if not destination.exists():shutil.copyfile(ROOT/entry['path'],destination)
        if base.digest(destination)!=entry['manifest_sha256']:raise ValueError('carried reference changed')
    value['protocol_sha256']=base.digest(path)
    return value


def run_one(arm,seed,train,val,protocol,*,role='randomness_estimation'):
    run_id=f'{arm}_seed_{seed}'+('_repro' if role=='fixed_repro' else '')
    if run_id in protocol['carried_runs']:
        path=NEW_OUT/run_id/'completed.json'
        if base.digest(path)!=protocol['carried_runs'][run_id]['manifest_sha256']:raise ValueError('carried manifest changed')
        return base.json.loads(path.read_text(encoding='utf-8'))
    return ORIGINAL_RUN(arm,seed,train,val,protocol,role=role)


def main():
    base.OUT=NEW_OUT
    base.freeze_protocol=freeze_protocol
    base.make_model=make_model
    base.fit=fit
    base.run_one=run_one
    base.main()


if __name__=='__main__':main()
