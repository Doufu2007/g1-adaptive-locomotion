"""Analyze all locked first-update cells and the two predeclared MAE contrasts."""
from __future__ import annotations
import json
import numpy as np
from scipy.stats import t
from statsmodels.stats.multitest import multipletests
from scripts import diagnose_g1_initial_update_20261026 as exp
from scripts.summarize_g1_best_phase_control import score


def effect(values):
    x=np.asarray(values,dtype=float)
    mean,sd=float(x.mean()),float(x.std(ddof=1))
    se=sd/np.sqrt(len(x));half=float(t.ppf(.975,len(x)-1)*se)
    p=float(2*t.sf(abs(mean/se),len(x)-1)) if se else (1. if mean==0 else 0.)
    return dict(per_training_seed=dict(zip(map(str,exp.SEEDS),x.tolist())),
        mean_difference_mps=mean,paired_t_95pct_ci_mps=[mean-half,mean+half],
        two_sided_p=p,seeds_with_lower_mae=int((x<0).sum()),
        standardized_paired_effect_dz=mean/sd if sd else None)


def extra(raw,row):
    from scripts.g1_stateful_phase import contact_labels
    q=raw['quaternion']
    yaw=np.unwrap(np.arctan2(2*(q[:,0]*q[:,3]+q[:,1]*q[:,2]),1-2*(q[:,2]**2+q[:,3]**2)))
    direction=np.array([np.cos(yaw[0]),np.sin(yaw[0])])
    row.update(abs_heading_change_rad=abs(float(yaw[-1]-yaw[0])),
        initial_direction_displacement_m=float((raw['position'][-1,:2]-raw['position'][0,:2])@direction))
    padded=np.zeros(1000);padded[:len(raw['time'])]=raw['velocity'][:,0]
    assert abs(float(abs(padded-.5).mean())-row['mae_forward_velocity_mps'])<1e-10
    mask=raw['time']>=3
    if mask.any():
        amplitude=np.percentile(raw['joint_pos'][mask],95,axis=0)-np.percentile(raw['joint_pos'][mask],5,axis=0)
        row['after3_velocity_mps']=float(raw['velocity'][mask,0].mean())
        row['hip_pitch_90pct_range_rad']=float(amplitude[[0,6]].mean())
        row['knee_90pct_range_rad']=float(amplitude[[3,9]].mean())
    else:
        for key in ('after3_velocity_mps','hip_pitch_90pct_range_rad','knee_90pct_range_rad'):row[key]=None
    _,_,_,cycles=contact_labels(raw['contact'],debounce=3)
    travel=[float(raw['velocity'][s:e,0].sum())*.02 for s,e,_ in cycles
            if raw['time'][s]>=3 and .15<=raw['time'][e]-raw['time'][s]<=2]
    row['cycle_forward_travel_m']=float(np.mean(travel)) if travel else None
    return row


METRICS=('mae_forward_velocity_mps','tilt_rms_rad','abs_heading_change_rad',
    'initial_direction_displacement_m','mean_observed_forward_velocity_mps',
    'after3_velocity_mps','hip_pitch_90pct_range_rad','knee_90pct_range_rad','cycle_forward_travel_m')


def main():
    import torch
    protocol=exp.lock()
    assert exp.read(exp.OUT/'queue_status.json')['status']=='COMPLETED'
    integrity=[];mechanisms=[];traces=[]
    for arm,seed in protocol['capture_order']:
        directory=exp.OUT/'captures'/f'{arm}_{seed}'
        capture=exp.read(directory/'manifest.json')
        assert capture['status']=='COMPLETED' and capture['transitions']==1536
        assert capture['original_initial_actor_matches'] and capture['encoder_frozen']
        assert capture['observation_compatibility']['status']=='PASS'
        for name,key in [('batch.pt','batch_sha256'),('actual_post.pt','actual_post_sha256')]:
            assert exp.sha(directory/name)==capture[key]
        integrity.append(capture)
        original=exp.read(directory/'actual_trace.json')['minibatches']
        for variant in exp.VARIANTS:
            folder=exp.OUT/'replays'/f'{arm}_{seed}_{variant}'
            manifest=exp.read(folder/'manifest.json');trace=exp.read(folder/'trace.json')
            assert manifest['status']=='COMPLETED' and manifest['batch_sha256']==capture['batch_sha256']
            assert exp.sha(folder/'model_0.pt')==manifest['checkpoint_sha256']
            assert exp.sha(folder/'policy.onnx')==manifest['policy_sha256']
            assert len(trace['minibatches'])==20
            assert [r['order_sha256'] for r in trace['minibatches']]==[r['order_sha256'] for r in original]
            if variant=='default_fresh':
                assert manifest['rng_exactly_matched']
                assert max(manifest['instrumented_vs_native_max_error'].values())<=1e-7
            targets=torch.load(folder/'targets.pt',map_location='cpu',weights_only=False)
            minibatches=trace['minibatches']
            lr=[r['effective_learning_rate'] for r in minibatches]
            kl=[r['measured_training_minibatch_KL_mean'] for r in minibatches]
            summary=dict(manifest,final_lr=trace['final_lr'],
                first_minibatch_lr=lr[0],first_minibatch_KL=kl[0],second_minibatch_KL=kl[1],
                peak_training_minibatch_KL=max(kl),
                first_floor_minibatch_1based=next((i+1 for i,v in enumerate(lr) if v<=1e-5),None),
                value_mean=float(targets['values'].mean()),value_std=float(targets['values'].std()),
                return_mean=float(targets['returns'].mean()),return_std=float(targets['returns'].std()),
                advantage_mean=float(targets['advantages'].mean()),advantage_std=float(targets['advantages'].std()))
            mechanisms.append(summary)
            traces.append(dict(arm=arm,train_seed=seed,variant=variant,**trace))
    for seed in exp.SEEDS:
        pair=[r for r in integrity if r['train_seed']==seed]
        assert len(pair)==2 and pair[0]['initial_critic_sha256']==pair[1]['initial_critic_sha256']
    episodes=[];starts={};manifests={};pairing=[]
    for seed in exp.SEEDS:
        for condition in exp.EVALUATED:
            directory=exp.OUT/'rollouts'/f'{condition}_{seed}'
            manifest,rows,raws=score(directory)
            assert manifest['episode_count']==3 and manifest['horizon']==20.
            assert manifest['seeds']==list(exp.EVAL) and manifest['encoder_sha256']==protocol['encoder_sha256']
            assert manifest['initial_update_probe_sha256']==exp.sha(exp.__file__)
            assert exp.sha(manifest['policy'])==manifest['policy_sha256']
            manifests[(seed,condition)]=manifest
            for row in rows:
                raw=raws[('flat','steady',row['seed'])]
                row=extra(raw,dict(row,train_seed=seed,condition=condition))
                starts[(seed,condition,row['seed'])]={k:raw[k][0].copy() for k in ('position','quaternion','joint_pos','velocity')}
                episodes.append(row)
    for seed in exp.SEEDS:
        base=manifests[(seed,'before_update')]
        yaml_path=exp.OUT/'rollouts'/f'before_update_{seed}'/'env.yaml'
        for condition in exp.EVALUATED:
            manifest=manifests[(seed,condition)]
            assert manifest['startup_model_hashes']==base['startup_model_hashes']
            assert exp.sha(exp.OUT/'rollouts'/f'{condition}_{seed}'/'env.yaml')==exp.sha(yaml_path)
            for es in exp.EVAL:
                a,b=starts[(seed,'before_update',es)],starts[(seed,condition,es)]
                errors={k:float(np.max(np.abs(a[k].astype(float)-b[k].astype(float)))) for k in a}
                assert max(errors.values())<=1e-6
                pairing.append(dict(train_seed=seed,condition=condition,eval_seed=es,initial_max_errors=errors))
    aggregates=[]
    for seed in (*exp.SEEDS,'all'):
        for condition in exp.EVALUATED:
            chosen=[r for r in episodes if r['condition']==condition and (seed=='all' or r['train_seed']==seed)]
            assert len(chosen)==(9 if seed=='all' else 3)
            aggregates.append(dict(train_seed=seed,condition=condition,episodes=len(chosen),falls=sum(r['fell'] for r in chosen),
                **{k:float(np.mean([r[k] for r in chosen if r[k] is not None]))
                   if any(r[k] is not None for r in chosen) else None for k in METRICS}))
    def cell(seed,condition):return next(r for r in aggregates if r['train_seed']==seed and r['condition']==condition)
    contrasts=[]
    for condition in ('low_lr_fresh','default_source_critic'):
        values=[cell(seed,condition)['mae_forward_velocity_mps']-cell(seed,'default_fresh')['mae_forward_velocity_mps'] for seed in exp.SEEDS]
        a,b=cell('all',condition),cell('all','default_fresh')
        guard=dict(no_additional_fall_in_any_pair=all(
            not r['fell'] or next(x for x in episodes if (x['train_seed'],x['condition'],x['seed'])
                ==(r['train_seed'],'default_fresh',r['seed']))['fell'] for r in episodes if r['condition']==condition),
            tilt_delta_rad=a['tilt_rms_rad']-b['tilt_rms_rad'],
            heading_delta_rad=a['abs_heading_change_rad']-b['abs_heading_change_rad'],
            direction_displacement_delta_m=a['initial_direction_displacement_m']-b['initial_direction_displacement_m'])
        guard['pass']=guard['no_additional_fall_in_any_pair'] and guard['tilt_delta_rad']<=.02 and guard['heading_delta_rad']<=.2 and guard['direction_displacement_delta_m']>=-.5
        contrasts.append(dict(name=condition+'_minus_default_fresh',**effect(values),guardrails=guard))
    for row,q in zip(contrasts,multipletests([r['two_sided_p'] for r in contrasts],method='fdr_bh')[1]):
        row['q_bh_two_contrasts']=float(q)
        row['predeclared_support']=row['seeds_with_lower_mae']==3 and row['paired_t_95pct_ci_mps'][1]<0 and q<.05 and row['guardrails']['pass']
        exp.write(exp.OUT/'statistical_cards'/(row['name']+'.json'),dict(status='REVISION_REQUIRED',
            reason='Exploratory n=3 training seeds; normality untestable; fixed evaluation cohort; no confirmatory long-training conclusion',
            estimand='Treatment minus default first-update policy, seed-level full20s body-forward MAE',
            multiplicity='Complete predeclared two-contrast BH family',result=row))
    result=dict(status='COMPLETED',protocol_sha256=exp.sha(exp.OUT/'prospective_protocol.json'),
        script_sha256=exp.sha(__file__),verified_training_transitions=9216,actual_updates=6,replay_updates=24,
        failed_collection_overhead_transitions=1536,total_environment_training_transitions=10752,
        minibatches=600,formal_evaluation_episodes=len(episodes),all_source_hashes_verified=True,
        captures=integrity,mechanisms=mechanisms,actual_training_traces=traces,episodes=episodes,
        pairing_checks=pairing,aggregates=aggregates,contrasts=contrasts,
        scope='First-update diagnosis, exploratory n=3; not a full501-update superiority claim; transferred critic may be mismatched.')
    assert len(episodes)==36 and len(mechanisms)==24
    exp.write(exp.OUT/'analysis.json',result)
    lines=['# PPO 首次更新诊断','',
        '固定首批数据，对初始学习率与价值网络初始化做2×2回放。6次真实更新、24次冻结回放、600个优化小批次；E四条件共36个正式20秒回合。',
        '默认回放不装记录器，与实际装记录器更新的actor/critic及RNG一致；fresh价值、GAE、观测、动作和小批次次序核验通过。',
        '','## 第一轮行走（目标0.5m/s，MAE越低越好）','',
        '|训练seed|条件|MAE(m/s)|倾斜(rad)|航向变化(rad)|初始方向位移(m)|跌倒|','|---|---|---:|---:|---:|---:|---:|']
    for r in aggregates:
        lines.append(f"|{r['train_seed']}|{r['condition']}|{r['mae_forward_velocity_mps']:.5f}|{r['tilt_rms_rad']:.4f}|{r['abs_heading_change_rad']:.4f}|{r['initial_direction_displacement_m']:.3f}|{r['falls']}/{r['episodes']}|")
    lines+=['','## 同批数据的实际更新过程','',
        '|phase|seed|处理|首步LR|第二批KL|最大KL|首次触底批次|动作RMS变化|','|---|---|---|---:|---:|---:|---:|---:|']
    for r in mechanisms:
        lines.append(f"|{r['arm']}|{r['train_seed']}|{r['variant']}|{r['first_minibatch_lr']:.6g}|{r['second_minibatch_KL']:.5f}|{r['peak_training_minibatch_KL']:.5f}|{r['first_floor_minibatch_1based']}|{r['action_rms_change']:.5f}|")
    lines+=['','## 两项预定速度对照','']
    for r in contrasts:
        lines.append(f"- {r['name']}：差值{r['mean_difference_mps']:.5f}m/s，95% t区间{r['paired_t_95pct_ci_mps']}，BH q={r['q_bh_two_contrasts']:.5g}；{r['seeds_with_lower_mae']}/3种子降低MAE；保护指标通过={r['guardrails']['pass']}；预定支持条件={r['predeclared_support']}。")
    lines+=['','## 解释边界','',
        '- 所有条件均报告，无按表现加预算或重选。配对初态、启动物理和评测YAML一致；物理仿真不宣称逐位重放。',
        '- before_update是共享rollout更新normalizer后、未执行optimizer的actor。',
        '- source_critic同时继承旧价值网络及其normalizer；效果不能细分成两者各自贡献，也不意味着旧V对新奖励准确。',
        '- 小批次KL是实际训练测量；固定状态池条件KL及动作RMS只用于描述策略漂移。完整20个小批次数据在analysis.json中。',
        '- 本机torch2.6/Windows与原远端torch2.8/Linux的随机critic初态哈希不同，具体来源未锁定；原actor相同，本机clock/E配对critic相同。旧冻结版本和两次技术失败保留在父目录；第二次因TensorDict迭代接口误用损失1536采集步、0次optimizer更新。科学预算9216步，含失败总计10752步。本实验是本机匹配诊断，不能冒充原远端首轮的逐位复现。',
        '- 本实验只定位第一轮更新。不能据此认定501轮的慢走由同一原因导致，不能认定E优于clock。n=3无法检验分布，统计卡保留REVISION_REQUIRED。','']
    (exp.OUT/'diagnosis.md').write_text('\n'.join(lines),encoding='utf-8')
    exp.write(exp.OUT/'completion_verification.json',dict(status='PASS',at=exp.now(),
        training_transitions=9216,failed_collection_overhead_transitions=1536,total_training_transitions=10752,
        minibatches=600,formal_episodes=36,primary_family_size=2,
        reports=[dict(path=str(exp.OUT/name),sha256=exp.sha(exp.OUT/name)) for name in ('analysis.json','diagnosis.md')],
        statistical_cards='REVISION_REQUIRED: exploratory n=3',no_remote_launched=True,no_shutdown=True))
    print(json.dumps(dict(contrasts=contrasts,aggregates=aggregates),ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
