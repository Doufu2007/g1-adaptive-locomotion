"""Machine-readable bounds on all nine exploratory diagnostic comparisons."""
from pathlib import Path
import subprocess
import sys

from scripts import g1_phase_feedback_probe_20261026 as exp


def run():
    analysis=exp.read(exp.OUT/'analysis.json')
    protocol=exp.read(exp.OUT/'prospective_protocol.json')
    assert analysis['status']=='COMPLETED'
    root=exp.OUT/'statistical_cards'
    root.mkdir(exist_ok=True)
    engine=Path.home()/'.codex/skills/light-result-analysis/scripts/result_card_gate.py'
    index=dict(at=exp.prior.now(),manifests=analysis['sources'],
               raw_results=[dict(path=r['raw_path'],sha256=r['raw_sha256']) for r in analysis['episodes']])
    exp.write('statistical_cards/provenance_index.json',index)
    earliest=min(exp.read(exp.OUT/'rollouts'/s['name']/'manifest.json')['ended_at'] for s in analysis['sources'])
    results=[]
    for row in analysis['contrasts']:
        source_ids=[f'{condition}_{seed}' for condition in (row['treatment'],row['control']) for seed in exp.SEEDS]
        spec=dict(schema='light.result_card.v1',project='g1-phase-feedback-diagnosis',as_of=exp.prior.now(),
            plan_lock=dict(status='EXPLORATORY',locked_at=protocol['at'],results_available_at=earliest,
                locator=str(exp.OUT/'prospective_protocol.json'),sha256=exp.prior.sha(exp.OUT/'prospective_protocol.json')),
            result_card=dict(claim_id=row['name'],target_claim='Conditional fixed-policy MAE difference; mechanism remains bounded',
                target_type='EXPLORATORY',estimand_or_hypothesis=row['treatment']+' minus '+row['control']+' in '+row['scenario'],
                metric='fixed20s_body_forward_velocity_mae_mps',comparison_family=protocol['family_id'],family_size=9,correction='BH',
                analysis_set=dict(name='All three original PPO training seeds; every paired evaluation retained',n=3,
                    unit='PPO training seed; evaluation episodes nested',missingness=dict(rate=0,mechanism='NONE',
                        handling='All outcomes retained; falls scored over fixed horizon with remaining velocity zero')),
                provenance=dict(source_run_ids=source_ids,run_manifest_locator=str(root/'provenance_index.json'),
                    run_manifest_sha256=exp.prior.sha(root/'provenance_index.json'),raw_result_locator=str(exp.OUT/'analysis.json'),
                    raw_result_sha256=exp.prior.sha(exp.OUT/'analysis.json'),analysis_code_locator=str(exp.ROOT/'scripts/analyze_g1_phase_feedback_probe_20261026.py'),
                    analysis_code_sha256=analysis['analysis_script_sha256'],computed_at=analysis['at'],owner_skill='RESULT-ANALYSIS'),
                practical_threshold=dict(metric='MAE difference m/s',value=.005,direction='LOWER',
                    justification_locator='Exploratory reporting scale only:1% of the0.5m/s command; added after partial results, not a success criterion'),
                assumption_checks=[dict(assumption='Seed is replication unit; episodes not independent training runs',status='PASS',evidence_locator='analysis.json'),
                    dict(assumption='Approximately normal seed-level effects required by n=3 t intervals',status='UNKNOWN',evidence_locator='analysis.json:inference')],
                guardrail_analysis=dict(required=True,evidence_locator=str(exp.OUT/'analysis.json'),
                    evidence_sha256=exp.prior.sha(exp.OUT/'analysis.json'),checks=[
                        dict(id='raw_integrity',metric='raw_SHA_and_initial_physics_pairing',status='PASS',observed=180,threshold=180,
                            claim_impact='Every predeclared outcome verified',evidence_locator='analysis.json:pairing_checks'),
                        dict(id='multimetric_scope',metric='tilt_heading_lateral_falls_displacement',status='WARN',observed='All reported',threshold='No overall superiority claim',
                            claim_impact='MAE alone cannot establish improved overall control',evidence_locator='analysis.json:aggregates'),
                        dict(id='replay_identity',metric='original_strict_max_abs_trajectory_error_1e-5',status='WARN',
                            observed=analysis['strict_identity_passes'],threshold=18,
                            claim_impact='Exact phase replay verified; exact physical counterfactual not established. Repeat controls retained.',
                            evidence_locator='analysis.json:identity_checks')]),
                effect=dict(estimate=row['mean_difference_mps'],effect_size=row['standardized_paired_effect_dz'],
                    ci95=row['paired_t_95pct_ci_mps'],p=row['two_sided_p'],q=row['q_bh_nine_contrasts'],
                    direction='SIGNED_CONDITIONAL_DIFFERENCE',practical_significance='UNKNOWN'),
                language=dict(conclusion='Exploratory conditional differences only; mechanism and broad direction remain uncertain.',
                    strength='DESCRIPTIVE_ONLY',near_threshold_wording='Small-n uncertainty and complete-family correction apply; no threshold-dependent promotion.')),
            decision_ledger=[dict(decision_id='nine_conditions_locked',timing='PRE_RESULTS',kind='ANALYSIS_SET',planned=True,
                classification='EXPLORATORY',affected_claim_ids=[row['name']],rationale='All9 contrasts locked before new simulations',evidence_locator='prospective_protocol.json'),
                dict(decision_id='numerical_repeat_controls',timing='POST_RESULTS',kind='GUARDRAIL',planned=False,
                classification='EXPLORATORY',affected_claim_ids=[row['name']],user_approved=False,
                authorization_locator='User authorized the diagnostic task; the agent selected this bounded technical repeat check without separate approval.',
                rationale='Preserve strict identity failure; add all-seed technical repeat controls',evidence_locator='numerical_control_amendment.json'),
                dict(decision_id='reporting_scale',timing='POST_RESULTS',kind='THRESHOLD',planned=False,classification='EXPLORATORY',
                affected_claim_ids=[row['name']],user_approved=False,
                authorization_locator='Agent-selected descriptive scale within the diagnostic report; no change to the primary metric, family, or success rule.',
                rationale='0.005m/s is a descriptive reference, not a decision or equivalence margin',evidence_locator='statistical_cards')],
            sensitivity=[],decision='REVISION_REQUIRED')
        path=root/(row['name']+'.json')
        exp.write(path.relative_to(exp.OUT),spec)
        completed=subprocess.run([sys.executable,str(engine),'--spec',str(path),'--report',str(root/(row['name']+'_findings.json')),
            '--json-out',str(root/(row['name']+'_gate.json'))],capture_output=True,text=True,encoding='utf-8')
        (root/(row['name']+'_gate.log')).write_text(completed.stdout+completed.stderr,encoding='utf-8')
        results.append(dict(contrast=row['name'],exit_code=completed.returncode,
            status='Diagnostic only; not a publication claim handoff',spec_sha256=exp.prior.sha(path)))
    rigor_spec=dict(project='g1-phase-feedback-diagnosis',correction='bh',comparisons_run=9,comparisons_reported=9,
        claims=[dict(claim_id=r['name'],text='Exploratory conditional fixed-policy MAE contrast',
            p=r['two_sided_p'],q_fdr=r['q_bh_nine_contrasts'],effect_size=r['standardized_paired_effect_dz'],
            effect_kind='paired_dz',ci95=r['paired_t_95pct_ci_mps'],n=3,asserted_grade='none',
            is_hypothesis=False,seeds=list(r['per_training_seed'].values())) for r in analysis['contrasts']])
    rigor_path=root/'stat_rigor_spec.json'
    exp.write(rigor_path.relative_to(exp.OUT),rigor_spec)
    rigor_engine=engine.with_name('stat_rigor_gate.py')
    completed=subprocess.run([sys.executable,str(rigor_engine),'--spec',str(rigor_path),
        '--report',str(root/'stat_rigor_findings.json'),'--evidence-out',str(root/'evidence_strength.json'),
        '--json-out',str(root/'stat_rigor_gate.json')],capture_output=True,text=True,encoding='utf-8')
    (root/'stat_rigor_gate.log').write_text(completed.stdout+completed.stderr,encoding='utf-8')
    exp.write('statistical_cards/status.json',dict(status='COMPLETED',at=exp.prior.now(),results=results,
        complete_family_rigor_exit_code=completed.returncode,
        scope='Explicit REVISION_REQUIRED is expected for broad claims; does not block the authorized bounded diagnostic report.'))
    print(results)


if __name__=='__main__':
    run()
