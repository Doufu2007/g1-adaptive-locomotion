"""Recompute every factorial rollout and audit the frozen training budget."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
from scipy.stats import t
from statsmodels.stats.multitest import multipletests
from scripts import g1_phase_lr_factorial_20261027 as study
from scripts.summarize_g1_best_phase_control import score

METRICS=('mae_forward_velocity_mps','tilt_rms_rad','abs_heading_change_rad','initial_direction_displacement_m',
 'mean_observed_forward_velocity_mps','after3_velocity_mps','hip_pitch_90pct_range_rad','knee_90pct_range_rad',
 'cycle_forward_travel_m','phase_completion','contact_phase_aligned_mae_rad')

def extra(raw,row):
    from scripts.g1_stateful_phase import contact_labels
    from scripts.diagnose_g1_best_e_closed_loop_20261026 import canonical_phase_score
    q=raw['quaternion'];yaw=np.unwrap(np.arctan2(2*(q[:,0]*q[:,3]+q[:,1]*q[:,2]),1-2*(q[:,2]**2+q[:,3]**2)))
    direction=np.array([np.cos(yaw[0]),np.sin(yaw[0])])
    row['abs_heading_change_rad']=abs(float(yaw[-1]-yaw[0]))
    row['initial_direction_displacement_m']=float((raw['position'][-1,:2]-raw['position'][0,:2])@direction)
    mask=raw['time']>=3
    if mask.any():
        amp=np.percentile(raw['joint_pos'][mask],95,axis=0)-np.percentile(raw['joint_pos'][mask],5,axis=0)
        row.update(after3_velocity_mps=float(raw['velocity'][mask,0].mean()),hip_pitch_90pct_range_rad=float(amp[[0,6]].mean()),knee_90pct_range_rad=float(amp[[3,9]].mean()))
    else:row.update(after3_velocity_mps=None,hip_pitch_90pct_range_rad=None,knee_90pct_range_rad=None)
    _,_,_,cycles=contact_labels(raw['contact'],debounce=3)
    travel=[float(raw['velocity'][s:e,0].sum())*.02 for s,e,_ in cycles if raw['time'][s]>=3 and .15<=raw['time'][e]-raw['time'][s]<=2]
    row['cycle_forward_travel_m']=float(np.mean(travel)) if travel else None
    phase=canonical_phase_score(raw['angle'],raw['contact'],debounce=3)
    row['phase_completion']=phase['completion'];row['canonical_phase']=phase
    row['contact_phase_aligned_mae_rad']=phase['contact_phase_error']['aligned_mae_rad']
    return row

def effect(values):
    x=np.asarray(values);mean=float(x.mean());se=float(x.std(ddof=1)/np.sqrt(len(x)));half=float(t.ppf(.975,len(x)-1)*se)
    return dict(mean=mean,per_seed=x.tolist(),n=len(x),ci95=[mean-half,mean+half],p=float(2*t.sf(abs(mean/se),len(x)-1)) if se else (1. if mean==0 else 0.))

def main(partial=False):
    protocol=study.lock()
    seal=study.read(study.OUT/'extended_source_seal.json')
    for item in seal['files']:assert study.sha(item['path'])==item['sha256'],('Extended frozen source changed',item['path'])
    training=[];timeline=[];episodes=[];pairing=[];starts={};physics={};coverage=[]
    for job in study.build_jobs():
        arm,lr,seed=job['arm'],job['lr_mode'],job['seed'];name=f'{arm}_{lr}_{seed}'
        folder=study.OUT/'training'/name
        if not (folder/'manifest.json').exists():
            if partial:continue
            raise ValueError('Missing training '+name)
        m=study.read(folder/'manifest.json')
        if m['status']!='COMPLETED':
            if partial:continue
            raise ValueError('Incomplete training '+name)
        assert m['updates']==501 and m['transitions']==769536 and m['encoder_frozen']
        assert m['protocol_sha256']==study.sha(study.OUT/'prospective_protocol.json')
        traces=[json.loads(line) for line in (folder/'updates.jsonl').read_text(encoding='utf-8').splitlines()]
        assert len(traces)==501 and [r['completed_update'] for r in traces]==list(range(1,502))
        assert all(len(r['minibatches'])==20 for r in traces)
        training.append(m)
        timeline.append(dict(arm=arm,lr_mode=lr,seed=seed,per_update=[dict(completed_update=r['completed_update'],
            at=r['at'],reward_mean=r['reward_mean'],return_std=r['return_std'],lr_after=r['lr_after'],losses=r['losses'],
            kl_max=max(b['kl_mean'] for b in r['minibatches']),kl_first=r['minibatches'][0]['kl_mean']) for r in traces]))
        for point in study.CHECKPOINTS:
            entry=m['checkpoints'][str(point)]
            assert study.sha(entry['path'])==entry['sha256'] and study.sha(entry['policy'])==entry['policy_sha256']
            evaluation=study.OUT/'evaluation'/f'{name}_u{point}'
            if not (evaluation/'manifest.json').exists():
                if partial:continue
                raise ValueError('Missing evaluation '+str(evaluation))
            em=study.read(evaluation/'manifest.json')
            if em['status']!='COMPLETED':
                if partial:continue
                raise ValueError('Incomplete evaluation '+str(evaluation))
            manifest,rows,raws=score(evaluation)
            expected={(s,e) for s in (['steady','speed_change','push'] if point==501 else ['steady'])
                      for e in (range(20263411,20263421) if point==501 else range(20263411,20263414))}
            assert len(rows)==len(expected) and {(r['scenario'],r['seed']) for r in rows}==expected
            assert manifest['policy_sha256']==entry['policy_sha256'] and manifest['encoder_sha256']==study.ENCODER_SHA
            assert manifest['protocol_sha256']==m['protocol_sha256']
            physics[(seed,arm,lr,point)]=manifest['startup_model_hashes']
            coverage.append(dict(arm=arm,lr_mode=lr,seed=seed,point=point,episodes=len(rows)))
            for r in rows:
                raw=raws[('flat',r['scenario'],r['seed'])]
                r=extra(raw,dict(r,eval_seed=r['seed'],seed=seed,lr_mode=lr,completed_updates=point))
                starts[(seed,arm,lr,point,r['scenario'],r['eval_seed'])]={k:raw[k][0].copy() for k in ('position','quaternion','joint_pos','velocity')}
                episodes.append(r)
    for seed in study.SEEDS:
        matched=[r for r in training if r['seed']==seed]
        if len(matched)!=4:
            if partial:continue
            raise ValueError('Missing training pair')
        for key in ('initial_actor_sha256','initial_critic_sha256','common_cfg_sha256','common_agent_sha256','startup_model_hashes'):
            assert all(r[key]==matched[0][key] for r in matched),key
        for point in study.CHECKPOINTS:
            pk=(seed,'clock','default',point)
            if pk not in physics:
                if partial:continue
                raise ValueError('Missing physics pair')
            for arm in ('clock','learned'):
                for lr in study.LR:
                    pk2=(seed,arm,lr,point)
                    if pk2 not in physics:
                        if partial:continue
                        raise ValueError('Missing physics pair')
                    assert physics[pk]==physics[pk2]
                    for k,values in starts.items():
                        if k[:4]!=(seed,arm,lr,point):continue
                        base=starts[(seed,'clock','default',point,*k[4:])]
                        error={field:float(np.max(np.abs(values[field].astype(float)-base[field].astype(float)))) for field in base}
                        assert max(error.values())<=1e-6
                        pairing.append(dict(key=list(k),initial_errors=error))
    aggregates=[]
    for point in study.CHECKPOINTS:
        for scenario in (['steady','speed_change','push'] if point==501 else ['steady']):
            for seed in (*study.SEEDS,'all'):
                for arm in ('clock','learned'):
                    for lr in study.LR:
                        chosen=[r for r in episodes if r['completed_updates']==point and r['scenario']==scenario and r['arm']==arm and r['lr_mode']==lr and (seed=='all' or r['seed']==seed)]
                        if not chosen:continue
                        aggregates.append(dict(point=point,scenario=scenario,seed=seed,arm=arm,lr_mode=lr,episodes=len(chosen),falls=sum(r['fell'] for r in chosen),
                            **{key:float(np.mean([r[key] for r in chosen if r[key] is not None])) if any(r[key] is not None for r in chosen) else None for key in METRICS}))
    final=[r for r in episodes if r['completed_updates']==501 and r['scenario']=='steady']
    contrasts={};candidate=None
    if len(final)==120:
        raw_contrasts=study.factorial_contrasts(final,'mae_forward_velocity_mps')
        contrasts={name:effect(r['per_seed']) for name,r in raw_contrasts.items()}
        for r,q in zip(contrasts.values(),multipletests([x['p'] for x in contrasts.values()],method='holm')[1]):r['holm_p']=float(q)
        e=[r for r in final if r['arm']=='learned' and r['lr_mode']=='low'];c=[r for r in final if r['arm']=='clock' and r['lr_mode']=='low']
        def avg(rows,key):return float(np.mean([r[key] for r in rows]))
        by_c={(r['seed'],r['eval_seed']):r for r in c}
        guards=dict(falls=sum(r['fell'] and not by_c[(r['seed'],r['eval_seed'])]['fell'] for r in e),
            tilt=avg(e,'tilt_rms_rad')-avg(c,'tilt_rms_rad'),heading=avg(e,'abs_heading_change_rad')-avg(c,'abs_heading_change_rad'),
            displacement=avg(e,'initial_direction_displacement_m')-avg(c,'initial_direction_displacement_m'))
        candidate=dict(study.candidate_gate(raw_contrasts['E_low_minus_clock_low']['per_seed'],guards),guards=guards)
    if not partial:
        assert len(training)==12 and len(episodes)==540 and len(coverage)==72
        assert sum(m['transitions'] for m in training)==9234432
    result=dict(status='PARTIAL_EXPLORATORY' if partial else 'COMPLETED',at=study.now(),analysis_script_sha256=study.sha(__file__),
        protocol_sha256=study.sha(study.OUT/'prospective_protocol.json'),training=training,training_timeline=timeline,
        coverage=coverage,pairing=pairing,episodes=episodes,aggregates=aggregates,contrasts=contrasts,candidate=candidate,
        scope='Three paired algorithm seeds; fixed simulation cohorts; intermediate statistics descriptive. No stable superiority claim without independent confirmation.')
    study.write(study.OUT/('analysis_partial.json' if partial else 'analysis.json'),result)
    lines=['# 最佳E与clock：初始学习率完整对照','',
        '部分结果，仅展示已完成任务，不能提前判定最终差异。' if partial else '12项501更新、540回合已完成并逐条复算；三个训练种子属于探索性对照。','',
        '|完成更新|场景|相位|初始LR组|MAE(m/s)|倾斜RMS|航向变化|初始方向位移|跌倒/回合|',
        '|---:|---|---|---|---:|---:|---:|---:|---:|']
    for r in aggregates:
        if r['seed']!='all':continue
        lines.append(f"|{r['point']}|{r['scenario']}|{r['arm']}|{r['lr_mode']}|{r['mae_forward_velocity_mps']:.5f}|{r['tilt_rms_rad']:.4f}|{r['abs_heading_change_rad']:.3f}|{r['initial_direction_displacement_m']:.3f}|{r['falls']}/{r['episodes']}|")
    lines += ['','## 预定四项比较','']
    for name,r in contrasts.items():lines.append(f"- {name}: 三seed差{r['per_seed']}，均值{r['mean']:.5f}，95%t区间{r['ci95']}，Holm p={r['holm_p']:.5f}。")
    lines += ['','候选分支：'+json.dumps(candidate,ensure_ascii=False),'',
        '小初始学习率组仍使用自适应调度；不得称固定1e-5。中间检查点不是择优终点。0至251使用三世界，501使用十世界，跨批大小的步态变化不能直接称完全相同物理个体的纵向因果。']
    (study.OUT/('diagnosis_partial.md' if partial else 'diagnosis.md')).write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=result['status'],training=len(training),episodes=len(episodes),candidate=candidate)),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--partial',action='store_true');args=parser.parse_args();main(args.partial)
