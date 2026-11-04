def test_long_protocol_uses_three_paired_seeds_and_3000_updates():
    from scripts import run_g1_extero_long as long

    protocol = long.long_protocol()
    assert protocol['training']['formal_iterations'] == 3000
    assert protocol['training']['num_envs'] == 512
    assert protocol['training']['seeds'] == [20261111, 20261112, 20261113]
    assert long.LONG_ARMS == ('matched', 'semantic')


def test_long_jobs_pair_matched_and_semantic_within_each_seed():
    from scripts import run_g1_extero_long as long

    jobs = long.selected_jobs(long.long_protocol())
    assert [(j['seed'], j['arm']) for j in jobs] == [
        (20261111, 'matched'), (20261111, 'semantic'),
        (20261112, 'matched'), (20261112, 'semantic'),
        (20261113, 'matched'), (20261113, 'semantic'),
    ]


def test_long_policy_path_is_separated_from_short_pilot():
    from scripts import run_g1_extero_long as long

    path = long.policy_directory(long.LONG_OUTPUT, 'semantic', 20261111)
    assert path.parts[-2:] == ('formal', 'semantic_seed20261111')


def test_decision_requires_paired_rmse_gain_and_no_survival_regression():
    from scripts import run_g1_extero_long as long

    good = {
        'paired': [
            {'semantic_rmse': .30, 'matched_rmse': .34,
             'semantic_survival': .85, 'matched_survival': .82},
            {'semantic_rmse': .31, 'matched_rmse': .34,
             'semantic_survival': .84, 'matched_survival': .83},
            {'semantic_rmse': .30, 'matched_rmse': .33,
             'semantic_survival': .86, 'matched_survival': .84},
        ]
    }
    assert long.judge_result(good)['action'] == 'SHUTDOWN'

    bad = {'paired': [{'semantic_rmse': .35, 'matched_rmse': .34,
                       'semantic_survival': .70, 'matched_survival': .82}]}
    assert long.judge_result(bad)['action'] == 'NEXT_EXPERIMENT'


def test_full_escalation_changes_only_the_training_budget():
    from scripts import run_g1_extero_full as full

    protocol = full.full_protocol()
    assert protocol['training']['formal_iterations'] == 10001
    assert protocol['training']['seeds'] == [20261111, 20261112, 20261113]
    assert protocol['arms'] == ['matched', 'semantic']
