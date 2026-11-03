"""Bounded diagnostic result cards; no publication-ready causal claims."""
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys

from scripts.audit_g1_training_mechanism_20261026 import OUT,read,sha,write


def now():
    return datetime.now(timezone.utc).isoformat()


def run():
    engine=Path.home()/'.codex/skills/light-result-analysis/scripts/result_card_gate.py'
    (OUT/'statistical_cards').mkdir(parents=True,exist_ok=True)
    sample=read(OUT/'sampling_analysis.json')
    sources=[dict(path=str(OUT/'sampling_analysis.json'),sha256=sha(OUT/'sampling_analysis.json'))]
    cases=[dict(effect=r,family='locked_action_sampling_three_contrasts',size=3,
        protocol=OUT/'policy_sampling/prospective_protocol.json',analysis=OUT/'sampling_analysis.json',
        code=Path(__file__).with_name('analyze_g1_training_mechanism_20261026.py'),
        roots=OUT/'policy_sampling/rollouts',guard='Every tilt, yaw, displacement and fall reported; no overall superiority claim')
        for r in sample['contrasts']]
    speedpath=OUT/'speed_priority/analysis.json'
    if speedpath.exists():
        speed=read(speedpath);row=dict(speed['contrast'],name='speed_priority601_minus_original_reward601')
        row['q_bh_three_contrasts']=row['two_sided_p']
        cases.append(dict(effect=row,family='speed_priority_one_new_exploratory_contrast',size=1,
            protocol=OUT/'speed_priority/prospective_protocol.json',analysis=speedpath,
            code=Path(__file__).with_name('analyze_g1_speed_priority_20261026.py'),
            roots=OUT/'speed_priority/rollouts',guard=str(speed['guardrails']),
            control_manifests=[OUT.parent/'g1_best_e_cause_audit_20261026/rollouts'/
                f'training_holdout_learned_original_reward_{seed}'/'manifest.json' for seed in (20261901,20261902,20261903)]))
        sources.append(dict(path=str(speedpath),sha256=sha(speedpath)))
    write('statistical_cards/provenance_index.json',dict(at=now(),sources=sources,
        limitations='Families fixed at distinct stages. Later speed hypothesis selected using earlier results; all remain exploratory.'))
    index=OUT/'statistical_cards/provenance_index.json'
    outputs=[]
    for case in cases:
        row=case['effect'];protocol=read(case['protocol'])
        manifests=list(case['roots'].glob('*/manifest.json'))+case.get('control_manifests',[])
        ends=[read(p)['ended_at'] for p in manifests if read(p).get('status')=='COMPLETED']
        claim=row['name'];card=dict(schema='light.result_card.v1',project='g1-training-mechanism',as_of=now(),
            plan_lock=dict(status='EXPLORATORY',locked_at=protocol['at'],results_available_at=min(ends),
                locator=str(case['protocol']),sha256=sha(case['protocol'])),
            result_card=dict(claim_id=claim,target_claim='Exploratory conditional difference; no single-root-cause assertion',
                target_type='EXPLORATORY',estimand_or_hypothesis=claim,metric='full20_body_forward_velocity_MAE_mps',
                comparison_family=case['family'],family_size=case['size'],correction='BH' if case['size']>1 else 'none',
                analysis_set=dict(name='All three training seeds and every planned evaluation slot',n=3,
                    unit='PPO training seed; episodes nested',missingness=dict(rate=0,mechanism='NONE',handling='Retain all outcomes')),
                provenance=dict(source_run_ids=[p.parent.name for p in manifests],
                    run_manifest_locator=str(index),run_manifest_sha256=sha(index),
                    raw_result_locator=str(case['analysis']),raw_result_sha256=sha(case['analysis']),
                    analysis_code_locator=str(case['code']),analysis_code_sha256=sha(case['code']),
                    computed_at=now(),owner_skill='RESULT-ANALYSIS'),
                practical_threshold=dict(metric='MAE difference m/s',value=.005,direction='LOWER',
                    justification_locator='Descriptive scale only, introduced for reporting; not a success or equivalence criterion'),
                assumption_checks=[dict(assumption='Training seed is independent replication unit',status='PASS',evidence_locator=str(case['analysis'])),
                    dict(assumption='n=3 t interval normality',status='UNKNOWN',evidence_locator='Not testable with three policy-level differences')],
                guardrail_analysis=dict(required=True,evidence_locator=str(case['analysis']),evidence_sha256=sha(case['analysis']),checks=[
                    dict(id='all_outcomes',metric='Raw SHA, initial physics and states',status='PASS',observed='verified',threshold='every planned episode',
                        claim_impact='Complete raw evidence retained',evidence_locator=str(case['analysis'])),
                    dict(id='control_quality',metric='tilt,yaw,displacement,falls',status='WARN',observed=case['guard'],threshold='No unqualified superiority',
                        claim_impact='Conditional speed metric cannot establish overall control benefit',evidence_locator=str(case['analysis']))]),
                effect=dict(estimate=row['mean_difference_mps'],effect_size=row['standardized_paired_effect_dz'],
                    ci95=row['paired_t_95pct_ci_mps'],p=row['two_sided_p'],q=row['q_bh_three_contrasts'],
                    direction='SIGNED_CONDITIONAL_DIFFERENCE',practical_significance='UNKNOWN'),
                language=dict(conclusion='Conditional exploratory results only; no sole causal explanation or broad E superiority established.',
                    strength='DESCRIPTIVE_ONLY',near_threshold_wording='Report uncertainty consistently around any threshold.')),
            decision_ledger=[dict(decision_id='exploratory_sequential_diagnosis',timing='POST_RESULTS',kind='SUBGROUP',planned=False,
                classification='EXPLORATORY',affected_claim_ids=[claim],user_approved=False,
                authorization_locator='User authorized continuing diagnosis; bounded local technical choices within that scope',
                rationale='Fixed policies and later reward weight were studied after original PPO results; no confirmatory rebranding',
                evidence_locator=str(case['protocol']))],sensitivity=[],decision='REVISION_REQUIRED')
        folder=OUT/'statistical_cards';path=folder/(claim+'.json');write(path.relative_to(OUT),card)
        result=subprocess.run([sys.executable,str(engine),'--spec',str(path),'--report',str(folder/(claim+'_findings.json')),
            '--json-out',str(folder/(claim+'_gate.json'))],capture_output=True,text=True,encoding='utf-8')
        (folder/(claim+'_gate.log')).write_text(result.stdout+result.stderr,encoding='utf-8')
        outputs.append(dict(claim=claim,result_card_exit=result.returncode))
    for family in sorted({c['family'] for c in cases}):
        selected=[c for c in cases if c['family']==family]
        spec=dict(project='g1-training-mechanism',correction='bh' if len(selected)>1 else 'none',
            comparisons_run=len(selected),comparisons_reported=len(selected),
            claims=[dict(claim_id=c['effect']['name'],text='Exploratory conditional diagnostic difference',
                p=c['effect']['two_sided_p'],q_fdr=c['effect']['q_bh_three_contrasts'],
                effect_size=c['effect']['standardized_paired_effect_dz'],effect_kind='paired_dz',
                ci95=c['effect']['paired_t_95pct_ci_mps'],n=3,asserted_grade='none',is_hypothesis=False,
                seeds=list(c['effect']['per_training_seed'].values())) for c in selected])
        p=OUT/'statistical_cards'/(family+'_rigor_spec.json');write(p.relative_to(OUT),spec)
        result=subprocess.run([sys.executable,str(engine.with_name('stat_rigor_gate.py')),'--spec',str(p),
            '--report',str(p.with_name(family+'_rigor_findings.json')),'--evidence-out',str(p.with_name(family+'_evidence_strength.json')),
            '--json-out',str(p.with_name(family+'_rigor_gate.json'))],capture_output=True,text=True,encoding='utf-8')
        p.with_name(family+'_rigor.log').write_text(result.stdout+result.stderr,encoding='utf-8')
        outputs.append(dict(family=family,stat_rigor_exit=result.returncode))
    write('statistical_cards/status.json',dict(status='COMPLETED',at=now(),outputs=outputs,
        scope='Publication claims remain REVISION_REQUIRED; failed evidence gates do not authorize more experiments.'))
    print(json.dumps(outputs,indent=2),flush=True)


if __name__=='__main__':
    run()
