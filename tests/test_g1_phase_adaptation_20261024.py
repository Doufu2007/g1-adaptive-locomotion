"""Scientific invariants for the matched policy-adaptation experiment."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json

import pytest


def subject():
    from scripts import g1_phase_adaptation_20261024
    return g1_phase_adaptation_20261024


def test_equal_reward_and_only_phase_source_differs():
    m = subject()
    m.bootstrap()
    for terrain in ("flat", "rough"):
        clock = m.make_cfg(terrain, "clock", 20261401, play=False, num_envs=64, horizon=20.)
        learned = m.make_cfg(terrain, "learned", 20261401, play=False, num_envs=64, horizon=20.)
        assert "foot_gait" not in clock.rewards and "foot_gait" not in learned.rewards
        assert asdict(clock)["rewards"] == asdict(learned)["rewards"]
        for group in ("actor", "critic"):
            assert learned.observations[group].terms["phase"].func.__name__ == "learned_phase"
            assert clock.observations[group].terms["phase"].func.__name__ == "phase"
            assert learned.observations[group].terms["phase"].params["encoder_path"] == str(m.RUNTIME)
        assert m.common_cfg_digest(clock) == m.common_cfg_digest(learned)


def test_evaluation_terrain_is_separate_and_phase_order_preserved():
    m = subject()
    m.bootstrap()
    train = m.make_cfg("rough", "learned", 20261401, play=False, num_envs=64, horizon=20.)
    test = m.make_cfg("rough", "learned", 20261501, play=True, num_envs=10, horizon=20.)
    assert train.scene.terrain.terrain_generator.seed != test.scene.terrain.terrain_generator.seed
    assert list(test.observations["actor"].terms).index("phase") == 3
    assert test.observations["actor"].history_length == 0


def test_dead_process_or_partial_episodes_never_unlocks_queue(tmp_path):
    m = subject()
    assert m.dependency_ready(tmp_path) is False
    for terrain in ("flat", "rough"):
        for arm in ("clock", "learned"):
            for batch in range(1, 4):
                p = tmp_path / "rollouts" / f"{terrain}_{arm}_batch{batch:02d}"
                p.mkdir(parents=True)
                seeds = list(range(20261201 + (batch-1)*10, 20261201 + batch*10))
                (p / "manifest.json").write_text(json.dumps({"status":"COMPLETED","episode_count":30,
                    "seeds":seeds,"scenarios":["steady","speed_change","push"]}))
                rows = [{"seed":s,"scenario":c} for s in seeds for c in ("steady","speed_change","push")]
                (p / "episodes.json").write_text(json.dumps(rows))
    assert m.dependency_ready(tmp_path)
    target = tmp_path / "rollouts" / "rough_learned_batch03" / "episodes.json"
    target.write_text("[]")
    assert m.dependency_ready(tmp_path) is False


def test_matrix_uses_equal_budget_and_same_clock_policy_for_direct_swap():
    m = subject()
    assert m.TRANSITIONS == 64 * 24 * 501
    jobs = m.training_jobs()
    assert len(jobs) == 12
    assert {(j["terrain"],j["seed"]) for j in jobs} == {
        (t,s) for t in ("flat","rough") for s in m.TRAIN_SEEDS}
    for terrain in ("flat", "rough"):
        combos = m.evaluation_jobs(terrain, m.TRAIN_SEEDS[0])
        assert [c["label"] for c in combos] == ["clock_clock","clock_learned","learned_learned"]
        assert combos[0]["policy"] == combos[1]["policy"]
        assert combos[1]["policy"] != combos[2]["policy"]


def test_inference_uses_training_runs_as_replicates():
    m = subject()
    # Repeating evaluation episodes must not create additional independent training runs.
    rows = []
    for seed in m.TRAIN_SEEDS:
        for terrain in ("flat","rough"):
            for label, value in (("clock_clock",.2),("clock_learned",.3),("learned_learned",.1)):
                for env_seed in (1,2,3,4):
                    rows.append(dict(train_seed=seed,terrain=terrain,label=label,seed=env_seed,
                        scenario="steady",fell=False,mae_forward_velocity_mps=value))
    result = m.paired_training_effect(rows,"flat","learned_learned","clock_clock","mae_forward_velocity_mps")
    assert result["training_seed_count"] == 3
    assert result["mean_difference"] == pytest.approx(-.1)
    assert len(result["per_training_seed"]) == 3
    with pytest.raises(ValueError, match="cohort|paired|missing"):
        m.paired_training_effect(rows[:-1],"rough","learned_learned","clock_learned","mae_forward_velocity_mps")


def test_previous_summary_accepts_ten_seed_shards_and_uses_terrain_keys():
    from scripts import summarize_g1_selected_phase_control_v3 as summary
    manifests=[dict(seeds=list(range(20261201+i*10,20261211+i*10)),
                    scenarios=["steady","speed_change","push"],horizon=20.,parallel_worlds=10)
               for i in range(3)]
    summary.validate_batch_manifests(manifests)
    good={"rough":{"fall_rate":{"bootstrap_95pct_ci":[-.2,-.1]},
                   "speed_mae":{"bootstrap_95pct_ci":[-.1,-.01]}},
          "flat":{"speed_mae":{"bootstrap_95pct_ci":[-.02,0.]}}}
    assert summary.claim_rule_passed(good)
    good["flat"]["speed_mae"]["bootstrap_95pct_ci"][1]=.001
    assert not summary.claim_rule_passed(good)
    manifests[1]["seeds"]=manifests[0]["seeds"]
    with pytest.raises(ValueError,match="batch"):
        summary.validate_batch_manifests(manifests)


def test_resume_recovers_intact_checkpoint_and_preserves_broken_files(tmp_path):
    import torch
    m=subject()
    good=tmp_path/"model_50.pt"
    state=dict(completed_updates=51,ppo_seed=m.TRAIN_SEEDS[0],terrain="flat",arm="learned",
               encoder_sha256=m.RUNTIME_SHA)
    torch.save(dict(iter=50,infos={"adaptation_state":state}),good)
    wrong=tmp_path/"model_100.pt"
    torch.save(dict(iter=100,infos={"adaptation_state":dict(state,completed_updates=101,arm="clock")}),wrong)
    partial=tmp_path/"model_150.pt"
    partial.write_bytes(b"interrupted write")
    selected,rejected=m.choose_resume_checkpoint(tmp_path,m.TRAIN_SEEDS[0],"flat","learned")
    assert selected==good
    assert len(rejected)==2
    assert partial.read_bytes()==b"interrupted write"
    assert wrong.exists()
