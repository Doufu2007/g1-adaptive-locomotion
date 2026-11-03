"""Sequential, frozen-budget external-model trials on the existing G1 dev split.

No test arrays are read. Long windows use past features across cycle boundaries.
All endpoints and all complete validation cycles remain in the scoring bundle.
"""
from __future__ import annotations

import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('PYTHONHASHSEED', '20261021')
import argparse
import contextlib
import hashlib
import importlib.util
import json
import platform
import random
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.g1_external_models import (ModernPhase,TCTSTPhase,TCTSTMaskedModel,FLDPhase,
    causal_windows,phase_rates,phase_rate_output,masked_reconstruction_loss,track_phase)
from scripts.run_g1_contact_followup_comparison import TCNEncoder
from scripts.run_g1_long_history_comparison import ContextCausalResidualBlock

DATA=ROOT/'results/g1_phase_dataset_20261019'
SOURCE=DATA/'phase_dataset.npz'
FILTERED=DATA/'contact_consistency_filter_v1/phase_dataset_train_validation.npz'
OUT=ROOT/'results/g1_external_methods_trials_20261022/run_v1'
GATE_PATH=ROOT/'src/tasks/velocity/phasor/phase_gate_v4_2.py'
spec=importlib.util.spec_from_file_location('frozen_phase_gate',GATE_PATH)
gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEEDS=(20261021,20261022,20261023)


def now(): return datetime.now(timezone.utc).isoformat()


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def write_json(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)


def amp():
    return torch.autocast('cuda',dtype=torch.bfloat16) if DEVICE.type=='cuda' else contextlib.nullcontext()


class LongRateTCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks=nn.Sequential(*[ContextCausalResidualBlock(29 if i==0 else 64,d)
                                    for i,d in enumerate((1,2,4,8,16,32))])
        self.head=nn.Linear(64,3)

    def forward(self,x):
        return phase_rate_output(self.head(self.blocks(x.transpose(1,2))[:,:,-1]))


class FloatHeadTCN(TCNEncoder):
    def forward(self,x):
        h=self.blocks(x.transpose(1,2))[:,:,-1]
        return torch.nn.functional.normalize(self.head(h).float(),dim=-1,eps=1e-8)


class FrameBank:
    """Only named split arrays; feature history reconstructed without refitting."""
    def __init__(self,split):
        if split not in ('train','validation'):raise ValueError('test split is forbidden')
        with np.load(SOURCE,allow_pickle=False) as z:
            x=z['x_'+split]
            self.cycles=z['y_'+split+'_cycles']
            self.sequence=z['sequence_'+split]
            self.frame=z['frame_'+split]
            self.left=z['left_touchdown_'+split]
            self.right=z['right_touchdown_'+split]
            self.mean=z['train_mean'];self.std=z['train_std']
        with np.load(FILTERED,allow_pickle=False) as z:
            fs=z['sequence_'+split];ff=z['frame_'+split]
            if not np.array_equal(self.mean,z['train_mean']) or not np.array_equal(self.std,z['train_std']):
                raise ValueError('frozen normalizers differ')
        self.consistent=np.zeros(len(x),dtype=bool)
        names=sorted(set(self.sequence.tolist()))
        lengths={s:int(self.frame[self.sequence==s].max())+1 for s in names}
        dense=np.zeros((sum(lengths.values()),29),dtype=np.float32)
        known=np.zeros(len(dense),dtype=bool)
        ends=np.zeros(len(x),dtype=np.int64)
        offset=0
        for s in names:
            ids=np.flatnonzero(self.sequence==s)
            self.consistent[ids]=np.isin(self.frame[ids],ff[fs==s])
            ends[ids]=offset+self.frame[ids]
            for j in range(10):
                pos=ends[ids]-9+j
                dense[pos]=x[ids,j];known[pos]=True
            offset+=lengths[s]
        # Every overlapping 10-frame value must agree with the source archive.
        for j in range(10):
            if not np.allclose(dense[ends-9+j],x[:,j],rtol=0,atol=1e-6):
                raise ValueError('source overlapping windows disagree')
        del x
        starts_by_frame=np.zeros(len(dense),dtype=np.int64)
        end_by_frame=np.zeros(len(dense),dtype=np.int64)
        offset=0
        for s in names:
            length=lengths[s];k=known[offset:offset+length]
            runs=np.flatnonzero(k & ~np.r_[False,k[:-1]])
            stops=np.flatnonzero(k & ~np.r_[k[1:],False])+1
            for a,b in zip(runs,stops):
                starts_by_frame[offset+a:offset+b]=offset+a
                end_by_frame[offset+a:offset+b]=offset+b-1
            offset+=length
        self.end_np=ends;self.start_np=starts_by_frame[ends]
        self.rate,self.rate_valid=phase_rates(self.cycles,self.sequence,self.frame)
        phase=np.stack((np.sin(self.cycles*2*np.pi),np.cos(self.cycles*2*np.pi)),axis=-1).astype(np.float32)
        self.target=torch.as_tensor(phase,device=DEVICE)
        self.rate_tensor=torch.as_tensor(self.rate,device=DEVICE)
        self.rate_mask=torch.as_tensor(self.rate_valid,device=DEVICE)
        self.features=torch.as_tensor(dense,device=DEVICE)
        self.ends=torch.as_tensor(ends,device=DEVICE)
        self.starts=torch.as_tensor(self.start_np,device=DEVICE)
        self.ids=torch.as_tensor(np.flatnonzero(self.consistent),device=DEVICE)
        self.future_ok=torch.as_tensor(ends+4<=end_by_frame[ends],device=DEVICE)
        self.pretrain_targets=None;self.pretrain_valid=None
        if split=='train':
            pd=np.zeros((len(dense),3),dtype=np.float32)
            pv=np.zeros_like(pd,dtype=bool)
            keep=np.flatnonzero(self.consistent)
            pd[ends[keep],:2]=phase[keep];pv[ends[keep],:2]=True
            rk=keep[self.rate_valid[keep]]
            pd[ends[rk],2]=self.rate[rk]/3.;pv[ends[rk],2]=True
            self.pretrain_targets=torch.as_tensor(pd,device=DEVICE)
            self.pretrain_valid=torch.as_tensor(pv,device=DEVICE)

    def windows(self,ids,window,shift=0):
        return causal_windows(self.features,self.ends[ids]+shift,window,self.starts[ids])

    def pretraining_joint(self,ids,window):
        end=self.ends[ids]
        indices=torch.maximum(end[:,None]+torch.arange(1-window,1,device=DEVICE),self.starts[ids,None])
        sensor=self.features[indices].transpose(1,2)
        ph=self.pretrain_targets[indices].transpose(1,2)
        valid=torch.cat((torch.ones_like(sensor,dtype=torch.bool),self.pretrain_valid[indices].transpose(1,2)),dim=1)
        return torch.cat((sensor,ph),dim=1),valid


def score_subset(records,names):
    rates=[];by={};lengths={}
    for s in names:
        rr=[r for r in records if r['sequence_id']==s and r['evaluable']]
        rate=float(np.mean([r['encoder_completed_one_turn'] for r in rr])) if rr else 0.
        by[s]={'cycles':len(rr),'completed':sum(r['encoder_completed_one_turn'] for r in rr),'macro_component':rate}
        rates.append(rate)
    for stratum in ('short','medium','long'):
        rr=[r for r in records if r['period_length_stratum']==stratum and r['evaluable']]
        lengths[stratum]={'cycles':len(rr),'completion':float(np.mean([r['encoder_completed_one_turn'] for r in rr])) if rr else 0.}
    ref_rates=[np.mean([r['reference_completed_one_turn'] for r in records if r['sequence_id']==s and r['evaluable']]) for s in names]
    ref_lengths=[np.mean([r['reference_completed_one_turn'] for r in records if r['period_length_stratum']==st and r['evaluable']]) for st in ('short','medium','long')]
    eligible=bool(records) and all(v['cycles']>=10 and v['macro_component']>=.8 for v in by.values()) and all(v['cycles']>=10 and v['completion']>=.8 for v in lengths.values()) and np.mean(rates)>=.9 and np.mean(ref_rates)>=.9 and min(ref_rates)>=.8 and min(ref_lengths)>=.8
    return {'observed_cycles':len(records),'evaluable_cycles':sum(r['evaluable'] for r in records),
        'macro_completion':float(np.mean(rates)),
        'micro_completion':float(np.mean([r['encoder_completed_one_turn'] for r in records if r['evaluable']])) if records else 0.,
        'by_sequence':by,'length_strata':lengths,'freeze_eligible':bool(eligible)}


def evaluate(pred,bank,run_dir,tag,window):
    pred=pred.astype(np.float32)
    if not np.isfinite(pred).all():raise FloatingPointError('nonfinite prediction')
    theta=np.arctan2(pred[:,0],pred[:,1]);truth=bank.cycles.astype(np.float64)*2*np.pi
    errors=np.abs(np.arctan2(np.sin(theta-truth),np.cos(theta-truth)))
    names=sorted(set(bank.sequence.tolist()))
    point={}
    for label,mask in [('all',np.ones(len(pred),bool)),('consistency',bank.consistent),('other_full_windows',~bank.consistent)]:
        seq={s:float(errors[mask & (bank.sequence==s)].mean()) for s in names}
        point[label]={'windows':int(mask.sum()),'macro_mae_rad':float(np.mean(list(seq.values()))),
            'micro_mae_rad':float(errors[mask].mean()),'rmse_rad':float(np.sqrt(np.mean(errors[mask]**2))), 'by_sequence':seq}
    full=gate.score_phase_cycles_v4_2(pred[:,:2],truth,bank.sequence,bank.frame,bank.left,
        period_cuts_frames=(50.66666666666666,75.),required_conditions=names)
    consistent=[];mismatch=[]
    for row in full['cycle_records']:
        select=(bank.sequence==row['sequence_id'])&(bank.frame>=row['start_frame'])&(bank.frame<row['end_frame'])
        nright=int(bank.right[select].sum())
        row['right_touchdown_count']=nright
        (consistent if nright==1 else mismatch).append(row)
    if (len(full['cycle_records']),len(consistent),len(mismatch))!=(603,504,99):
        raise ValueError('locked cycle cohort changed')
    result={'point':point,'cycles':{'all':score_subset(full['cycle_records'],names),
        'consistency':score_subset(consistent,names),'mismatch':score_subset(mismatch,names)},
        'official_full_freeze_eligible':full['encoder_freeze_eligible'],
        'padded_startup_windows':int(np.sum(bank.end_np-window+1<bank.start_np)),
        'window':window}
    np.save(run_dir/(tag+'_predictions.npy'),pred)
    write_json(run_dir/(tag+'_cycle_records.json'),full)
    write_json(run_dir/(tag+'_metrics.json'),result)
    return result


def predict(model,bank,window,cache=None,micro=256):
    model.eval();pieces=[]
    with torch.no_grad():
        for a in range(0,len(bank.ends),micro):
            ids=torch.arange(a,min(a+micro,len(bank.ends)),device=DEVICE)
            with amp():
                out=phase_rate_output(model.head(cache[ids])) if cache is not None else model(bank.windows(ids,window))
            pieces.append(out.float().cpu().numpy())
    return np.concatenate(pieces)


def cache_fld(model,bank,micro=256):
    model.eval();parts=[]
    with torch.no_grad():
        for a in range(0,len(bank.ends),micro):
            ids=torch.arange(a,min(a+micro,len(bank.ends)),device=DEVICE)
            with amp():parts.append(model.features(bank.windows(ids,51)).float())
    return torch.cat(parts)


def fit(model,bank,ids,run_dir,phase,seed,lr,window,micro,epochs=32,cache=None,pretrainer=None):
    params=[p for p in (pretrainer if pretrainer is not None else model).parameters() if p.requires_grad]
    optimizer=torch.optim.Adam(params,lr=lr)
    history=[]
    for epoch in range(epochs):
        t=time.perf_counter();model.train()
        if isinstance(model,FLDPhase) and phase=='supervised':model.encoder.eval()
        if pretrainer is not None:pretrainer.train()
        generator=torch.Generator().manual_seed(seed+1009*epoch)
        order=ids[torch.randperm(len(ids),generator=generator).to(DEVICE)]
        loss_sum=0.;seen=0
        for start in range(0,len(order),512):
            effective=order[start:start+512];optimizer.zero_grad(set_to_none=True)
            for small in range(0,len(effective),micro):
                ii=effective[small:small+micro]
                with amp():
                    if phase=='masked_pretrain':
                        joint,valid=bank.pretraining_joint(ii,window)
                        masked=torch.zeros(len(ii),32,dtype=torch.bool,device=DEVICE)
                        masked.scatter_(1,torch.rand(len(ii),32,device=DEVICE).topk(10,dim=1).indices,True)
                        reconstructed=pretrainer(joint.masked_fill(masked[:,:,None],0))
                        loss=masked_reconstruction_loss(reconstructed.float(),joint,masked,valid)
                    elif phase=='fld_pretrain':
                        encoder=model.encoder
                        encoder.device=DEVICE;encoder.args=encoder.args.to(DEVICE);encoder.freqs=encoder.freqs.to(DEVICE)
                        reconstruction,_,_,_=encoder(bank.windows(ii,window).transpose(1,2),k=5)
                        target=torch.stack([bank.windows(ii,window,shift=k).transpose(1,2) for k in range(5)])
                        loss=(reconstruction.float()-target).square().mean()
                    else:
                        out=phase_rate_output(model.head(cache[ii])) if cache is not None else model(bank.windows(ii,window))
                        out=out.float()
                        loss=(1-(out[:,:2]*bank.target[ii]).sum(dim=-1)).mean()
                        if out.shape[1]==3:
                            valid=bank.rate_mask[ii]
                            loss=loss+.02*((out[:,2]-bank.rate_tensor[ii]).square()*valid).sum()/valid.sum().clamp_min(1)
                if not torch.isfinite(loss):raise FloatingPointError(f'nonfinite loss at {phase}/{epoch}')
                (loss*(len(ii)/len(effective))).backward()
                loss_sum+=float(loss.detach())*len(ii);seen+=len(ii)
            norm=nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
            optimizer.step()
        torch.cuda.synchronize() if DEVICE.type=='cuda' else None
        row={'phase':phase,'epoch':epoch+1,'loss':loss_sum/seen,'samples':seen,
             'seconds':time.perf_counter()-t,'grad_norm_last':float(norm)}
        history.append(row)
        with (run_dir/'epochs.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
        if (epoch+1)%8==0 or epoch+1==epochs:
            torch.save({'phase':phase,'epoch':epoch+1,'model':model.state_dict(),
                'pretrainer':pretrainer.state_dict() if pretrainer is not None else None,
                'optimizer':optimizer.state_dict(),'torch_rng':torch.get_rng_state(),
                'cuda_rng':torch.cuda.get_rng_state_all() if DEVICE.type=='cuda' else None},run_dir/'partial_checkpoint.pt')
    del optimizer
    return history


def make_model(arm):
    if arm=='tcn10':return FloatHeadTCN(),10
    if arm=='modern10':return ModernPhase(),10
    if arm=='tcn100_rate':return LongRateTCN(),100
    if arm.startswith('tctst'):return TCTSTPhase(),100
    if arm=='fld51':return FLDPhase(),51
    raise ValueError(arm)


class Tee:
    def __init__(self,*streams):self.streams=streams
    def write(self,s):
        for f in self.streams:f.write(s);f.flush()
    def flush(self):
        for f in self.streams:f.flush()


def run_one(arm,seed,train,val,protocol,*,role='randomness_estimation'):
    run_id=f'{arm}_seed_{seed}'+('_repro' if role=='fixed_repro' else '')
    parent=OUT/run_id;parent.mkdir(exist_ok=True)
    complete=parent/'completed.json'
    if complete.exists():
        saved=json.loads(complete.read_text(encoding='utf-8'))
        if saved['protocol_sha256']!=protocol['protocol_sha256']:raise ValueError('completed run protocol mismatch')
        return saved
    attempts=sorted(parent.glob('attempt_*'));run_dir=parent/f'attempt_{len(attempts)+1:02d}';run_dir.mkdir()
    started=now()
    manifest={'schema':'g1.external_trial.run.v1','run_id':run_id,'arm':arm,'seed':seed,'seed_role':role,
        'started_at':started,'status':'RUNNING','protocol_sha256':protocol['protocol_sha256'],
        'source_hashes':protocol['source_hashes'],'code_hashes':protocol['code_hashes'],
        'environment':protocol['environment'],'argv':sys.argv,'attempt':str(run_dir.relative_to(ROOT))}
    write_json(run_dir/'manifest.json',manifest)
    with (run_dir/'stdout.log').open('w',encoding='utf-8') as log,(run_dir/'stderr.log').open('w',encoding='utf-8') as err:
        with contextlib.redirect_stdout(Tee(sys.stdout,log)),contextlib.redirect_stderr(Tee(sys.stderr,err)):
            try:
                seed_all(seed);model,window=make_model(arm);model=model.to(DEVICE)
                print(json.dumps({'started':run_id,'params':sum(p.numel() for p in model.parameters()),'at':started}),flush=True)
                micro=512
                histories=[]
                if arm=='tctst_masked100':
                    pre=TCTSTMaskedModel(model).to(DEVICE)
                    histories+=fit(model,train,train.ids[::5],run_dir,'masked_pretrain',seed,6e-4,window,128,pretrainer=pre)
                    del pre;torch.cuda.empty_cache()
                if arm=='fld51':
                    future_ids=train.ids[train.future_ok[train.ids]][::5]
                    histories+=fit(model,train,future_ids,run_dir,'fld_pretrain',seed,1e-4,window,128)
                    for p in model.encoder.parameters():p.requires_grad_(False)
                    train_cache=cache_fld(model,train);val_cache=cache_fld(model,val)
                else:train_cache=val_cache=None
                lr=1e-3 if arm in ('tcn10','modern10') else 6e-4
                histories+=fit(model,train,train.ids,run_dir,'supervised',seed,lr,window,micro,cache=train_cache)
                prediction=predict(model,val,window,cache=val_cache,micro=256)
                result=evaluate(prediction,val,run_dir,'direct',window)
                metrics={'direct':result}
                if arm=='fld51':
                    tracked=track_phase(prediction[:,:2],prediction[:,2],val.sequence,val.frame,alpha=.1)
                    metrics['tracked']=evaluate(tracked,val,run_dir,'tracked',window)
                torch.save({'arm':arm,'window':window,'joint_dim':29,'seed':seed,'model':model.state_dict(),
                    'train_mean':train.mean,'train_std':train.std,'protocol_sha256':protocol['protocol_sha256']},run_dir/'final_checkpoint.pt')
                weights=hashlib.sha256()
                for name,v in sorted(model.state_dict().items()):
                    weights.update(name.encode());weights.update(v.detach().cpu().contiguous().numpy().tobytes())
                manifest.update(status='COMPLETED',ended_at=now(),metrics=metrics,
                    parameter_count=sum(p.numel() for p in model.parameters()),
                    weight_content_sha256=weights.hexdigest(),prediction_sha256=digest(run_dir/'direct_predictions.npy'),
                    final_training_loss=histories[-1]['loss'],microbatch=micro,
                    artifacts={p.name:digest(p) for p in run_dir.iterdir() if p.suffix in ('.npy','.json','.pt') and p.name!='manifest.json'})
                write_json(run_dir/'manifest.json',manifest);write_json(complete,manifest)
                print(json.dumps({'completed':run_id,'mae':result['point']['consistency']['macro_mae_rad'],
                    'gate504':result['cycles']['consistency']['macro_completion'],
                    'gate603':result['cycles']['all']['macro_completion']}),flush=True)
                del model,train_cache,val_cache;torch.cuda.empty_cache()
                return manifest
            except BaseException as exc:
                manifest.update(status='FAILED',ended_at=now(),failure_type=type(exc).__name__,failure=str(exc))
                write_json(run_dir/'manifest.json',manifest)
                write_json(run_dir/'failure.json',{'traceback':traceback.format_exc(),'exception':str(exc)})
                raise


def freeze_protocol():
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'protocol.json'
    codes=[Path(__file__),ROOT/'scripts/g1_external_models.py',ROOT/'scripts/g1_external_vendor/modern_blocks.py',
        ROOT/'scripts/g1_external_vendor/fld.py',GATE_PATH,
        ROOT/'scripts/run_g1_contact_followup_comparison.py',ROOT/'scripts/run_g1_long_history_comparison.py']
    hashes={str(p.relative_to(ROOT)):digest(p) for p in codes}
    source_hashes={str(p.relative_to(ROOT)):digest(p) for p in (SOURCE,FILTERED)}
    if path.exists():
        value=json.loads(path.read_text(encoding='utf-8'))
        if value['code_hashes']!=hashes:raise ValueError('frozen code changed; create an explicit amendment/new run root')
        if value['source_hashes']!=source_hashes:raise ValueError('frozen dataset changed; refusing to mix runs')
    else:
        git=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True)
        dirty=subprocess.run(['git','status','--porcelain'],cwd=ROOT,capture_output=True,text=True)
        value={'schema':'g1.external_methods.protocol.v1','frozen_at':now(),
            'scope':'existing train/validation development comparison; no independent confirmation',
            'test_split_loaded':False,'source_hashes':source_hashes,
            'code_hashes':hashes,'plan_sha256':digest(ROOT/'docs/superpowers/plans/2026-10-22-external-phase-models.md'),
            'seeds':list(SEEDS),'supervised_epochs':32,'pretraining_epochs':32,'effective_batch':512,
            'normalization':'frozen original training mean/std, no refit',
            'selection':'final fixed epoch; validation not used for optimization or selection',
            'input':'29 joint positions, no contacts as estimator input',
            'stages':{'1':['tcn10','modern10'],'2':['tcn100_rate','tctst_scratch100','tctst_masked100'],'3':['fld51']},
            'loss':'1-cos phase + .02*MSE cycles_per_second for joint-output models',
            'fld_tracking':{'alpha':.1,'dt':.02,'reference_events_used':False},
            'environment':{'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'os':platform.platform(),
                'device':str(DEVICE),'gpu':torch.cuda.get_device_name(0) if DEVICE.type=='cuda' else None,
                'training_precision':'bf16 autocast; float32 loss and FFT','git_head':git.stdout.strip(),
                'git_dirty_paths':dirty.stdout.splitlines()}}
        write_json(path,value)
    value['protocol_sha256']=digest(path)
    return value


def aggregate(records):
    rows=[]
    for arm in sorted(set(r['arm'] for r in records if r['seed_role']=='randomness_estimation')):
        runs=[r for r in records if r['arm']==arm and r['seed_role']=='randomness_estimation']
        for tag in runs[0]['metrics']:
            rr=[r['metrics'][tag] for r in runs]
            row={'arm':arm,'mode':tag,'n_seeds':len(rr),'seeds':[r['seed'] for r in runs],
                 'window':rr[0]['window'],'parameters':runs[0]['parameter_count']}
            for label,path in [('mae_consistency',('point','consistency','macro_mae_rad')),
                ('mae_all',('point','all','macro_mae_rad')),
                ('gate504',('cycles','consistency','macro_completion')),
                ('gate99',('cycles','mismatch','macro_completion')),
                ('gate603',('cycles','all','macro_completion'))]:
                values=[r[path[0]][path[1]][path[2]] for r in rr]
                row[label]={'mean':float(np.mean(values)),'sd':float(np.std(values,ddof=1)) if len(values)>1 else None,'values':values}
            row['any_seed_freeze_eligible']=any(r['official_full_freeze_eligible'] for r in rr)
            rows.append(row)
    repro=[r for r in records if r['seed_role']=='fixed_repro']
    checks=[]
    for r in repro:
        original=next(v for v in records if v['arm']==r['arm'] and v['seed']==r['seed'] and v['seed_role']=='randomness_estimation')
        checks.append({'arm':r['arm'],'seed':r['seed'],
            'prediction_hash_equal':r['prediction_sha256']==original['prediction_sha256'],
            'weight_hash_equal':r['weight_content_sha256']==original['weight_content_sha256']})
    write_json(OUT/'summary.json',{'updated_at':now(),'rows':rows,'fixed_repro_checks':checks,
        'all_completed_run_ids':[r['run_id'] for r in records]})


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=('1','2','3','all'),default='all')
    args=parser.parse_args();seed_all(SEEDS[0]);protocol=freeze_protocol()
    train=FrameBank('train');val=FrameBank('validation')
    if len(train.ids)!=55820 or int(val.consistent.sum())!=32878:raise ValueError('locked samples changed')
    if set(train.sequence)&set(val.sequence):raise ValueError('sequence leakage')
    # Audit reference cohort before scoring any trained model.
    write_json(OUT/'data_audit.json',{'train_windows':len(train.ends),'train_consistency_windows':len(train.ids),
        'validation_windows':len(val.ends),'validation_consistency_windows':int(val.consistent.sum()),
        'train_sequences':sorted(set(train.sequence.tolist())),'validation_sequences':sorted(set(val.sequence.tolist())),
        'test_arrays_accessed':False,'normalizer_refitted':False})
    np.savez_compressed(OUT/'validation_index.npz',sequence=val.sequence,frame=val.frame,cycles=val.cycles,
        consistency=val.consistent,left_touchdown=val.left,right_touchdown=val.right)
    records=[]
    for f in OUT.glob('*/completed.json'):records.append(json.loads(f.read_text(encoding='utf-8')))
    for stage,arms in protocol['stages'].items():
        if args.stage not in (stage,'all'):continue
        if stage!='1':
            for previous in map(str,range(1,int(stage))):
                for previous_arm in protocol['stages'][previous]:
                    for seed in SEEDS:
                        if not (OUT/f'{previous_arm}_seed_{seed}'/'completed.json').is_file():
                            raise RuntimeError(f'stage {stage} requires completed stage {previous}')
            if not (OUT/f'modern10_seed_{SEEDS[0]}_repro'/'completed.json').is_file():
                raise RuntimeError('fixed-seed reproduction must precede later stages')
        print(json.dumps({'stage':stage,'status':'STARTED','at':now()}),flush=True)
        for seed in SEEDS:
            for arm in arms:
                record=run_one(arm,seed,train,val,protocol)
                records=[r for r in records if r['run_id']!=record['run_id']]+[record];aggregate(records)
        if stage=='1':
            record=run_one('modern10',SEEDS[0],train,val,protocol,role='fixed_repro')
            records=[r for r in records if r['run_id']!=record['run_id']]+[record];aggregate(records)
            check=json.loads((OUT/'summary.json').read_text(encoding='utf-8'))['fixed_repro_checks'][0]
            if not check['prediction_hash_equal'] or not check['weight_hash_equal']:raise RuntimeError('same-seed full-run reproduction failed')
        print(json.dumps({'stage':stage,'status':'COMPLETED','at':now()}),flush=True)


if __name__=='__main__':main()
