"""Wait for the frozen direct-swap study, then run matched policy adaptation."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from scripts import g1_phase_adaptation_20261024 as study


def train(terrain, arm, seed, *, smoke=False, resume=False):
    if not smoke and seed not in study.TRAIN_SEEDS:
        raise ValueError("PPO seed is outside the frozen training matrix")
    if not study.dependency_ready():
        raise RuntimeError("prior 360-episode study is incomplete; no new GPU training is allowed")
    if not smoke:
        study.verify_assets()
    study.verify_runtime()
    np,torch = study.bootstrap()
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_rl_cfg,load_runner_cls
    from mjlab.utils.os import dump_yaml
    from src.tasks.velocity.mdp import observations

    iterations = 2 if smoke else study.ITERATIONS
    directory = (study.OUT/"smoke"/f"{terrain}_{arm}") if smoke else study.train_dir(terrain,arm,seed)
    manifest_path = directory/"manifest.json"
    previous = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
    if previous and previous["status"]=="COMPLETED":
        if (study.sha(directory/"policy.onnx") != previous["policy_sha256"]
                or previous["iterations"]!=iterations or previous["num_envs"]!=study.NUM_ENVS
                or previous["encoder_sha256"]!=study.RUNTIME_SHA
                or study.sha(previous["final_checkpoint"])!=previous["final_checkpoint_sha256"]):
            raise ValueError("completed policy changed")
        return
    if directory.exists() and (not resume or smoke):
        raise FileExistsError(f"preserve existing attempt; explicit --resume required: {directory}")
    directory.mkdir(parents=True,exist_ok=resume)
    if previous:
        archive = directory/"attempts"
        archive.mkdir(exist_ok=True)
        study.write(archive/f"manifest_{time.time_ns()}.json",previous)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    cfg = study.make_cfg(terrain,arm,seed,play=False,num_envs=study.NUM_ENVS,horizon=20.)
    other = study.make_cfg(terrain,"clock" if arm=="learned" else "learned",seed,
                          play=False,num_envs=study.NUM_ENVS,horizon=20.)
    if study.common_cfg_digest(cfg)!=study.common_cfg_digest(other):
        raise ValueError("training task configurations differ beyond phase source")
    task = "Unitree-G1-"+terrain.title()
    agent=load_rl_cfg(task)
    agent.seed=seed
    agent.logger="tensorboard"
    agent.save_interval=50
    agent.upload_model=False
    agent.algorithm.learning_rate=1e-3
    if agent.num_steps_per_env!=study.STEPS:
        raise ValueError("PPO rollout length differs from the frozen transition budget")
    source=study.base.POLICIES[terrain]/"model_9900.pt"
    record=dict(schema="g1.phase_policy_adaptation.run.v1",status="RUNNING",started_at=study.now(),
        terrain=terrain,arm=arm,ppo_seed=seed,scope="finite-budget actor warm-start adaptation",
        source_actor_checkpoint=str(source),source_actor_sha256=study.sha(source),
        encoder_sha256=study.sha(study.RUNTIME),iterations=iterations,num_envs=study.NUM_ENVS,
        num_steps_per_env=study.STEPS,environment_transition_budget=iterations*study.NUM_ENVS*study.STEPS,
        common_task_config_sha256=study.common_cfg_digest(cfg),fresh_critic_and_optimizer=not resume,
        ppo_config_sha256=__import__('hashlib').sha256(json.dumps(study.canonical(asdict(agent)),sort_keys=True).encode()).hexdigest(),
        resumed=resume,simulation_state_continuously_restored=False if resume else None,
        argv=sys.argv,python=sys.version,seed_controls="PYTHONHASHSEED/random/numpy/torch/CUDA; no cross-platform bitwise claim")
    study.write(manifest_path,record)
    env,runner = None,None
    collected=0
    carried_updates=0
    try:
        raw=ManagerBasedRlEnv(cfg=cfg,device="cuda:0")
        env=RslRlVecEnvWrapper(raw,clip_actions=agent.clip_actions)
        bundle=observations._load_phasor_encoder(raw,str(study.RUNTIME))
        bundle.encoder.eval().requires_grad_(False)
        encoder_initial=study.base.state_sha(bundle.encoder)
        record["initial_encoder_state_sha256"]=encoder_initial
        record["packages"]={p:__import__('importlib.metadata',fromlist=['version']).version(p)
                             for p in ("torch","mjlab","rsl-rl-lib","mujoco","warp-lang")}
        record["device_name"]=torch.cuda.get_device_name()
        record["startup_model_hashes"]={k:__import__('hashlib').sha256(
            np.ascontiguousarray(getattr(raw.sim.model,k).detach().cpu().numpy()).tobytes()).hexdigest()
            for k in ("body_mass","body_ipos","geom_friction")}
        original_step=env.step
        def checked_step(action):
            nonlocal collected
            if not torch.isfinite(action).all():
                raise ValueError("nonfinite PPO action")
            command=raw.command_manager.get_command("twist")
            if not bool(((command[:,0]>=.3-1e-6)&(command[:,0]<=.8+1e-6)
                         &(command[:,1:].abs().amax(dim=1)<1e-6)).all().item()):
                raise ValueError("command outside frozen forward-walking domain")
            result=original_step(action)
            collected+=study.NUM_ENVS
            return result
        env.step=checked_step
        torch.manual_seed(seed+1000);torch.cuda.manual_seed_all(seed+1000)
        cls=load_runner_cls(task) or MjlabOnPolicyRunner
        runner=cls(env,asdict(agent),str(directory),device="cuda:0")
        runner.load(str(source),load_cfg={"actor":True},strict=True,map_location="cuda:0")
        runner.current_learning_iteration=0
        raw.common_step_counter=0
        if runner.alg.optimizer.state:
            raise ValueError("actor warm-start must leave critic optimizer fresh")
        record["initial_actor_state_sha256"]=study.base.state_sha(runner.alg.actor)
        record["initial_critic_state_sha256"]=study.base.state_sha(runner.alg.critic)
        encoder_ids={id(p) for p in bundle.encoder.parameters()}
        optimizer_ids={id(p) for g in runner.alg.optimizer.param_groups for p in g["params"]}
        if encoder_ids & optimizer_ids or any(p.requires_grad for p in bundle.encoder.parameters()):
            raise ValueError("frozen encoder must not be trainable by PPO")
        if resume:
            checkpoint,rejected=study.choose_resume_checkpoint(directory,seed,terrain,arm)
            infos=runner.load(str(checkpoint),strict=True,map_location="cuda:0")
            state=infos["adaptation_state"]
            carried_updates=int(state["completed_updates"])
            runner.current_learning_iteration=carried_updates
            runner.alg.learning_rate=float(state["learning_rate"])
            random.setstate(state["python_rng"]);np.random.set_state(state["numpy_rng"])
            torch.set_rng_state(state["torch_rng"].cpu())
            torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda_rng"]])
            record.update(resume_checkpoint=str(checkpoint),resume_checkpoint_sha256=study.sha(checkpoint),
                recovered_updates=carried_updates,rejected_checkpoints=rejected,
                recovery_overhead_previous_attempt=previous.get("collected_environment_transitions"))
        original_save=runner.save
        def checked_save(path,infos=None):
            completed=runner.current_learning_iteration+1
            if study.base.state_sha(bundle.encoder)!=encoder_initial:
                raise ValueError("encoder weights changed during policy training")
            state=dict(completed_updates=completed,learning_rate=runner.alg.learning_rate,
                ppo_seed=seed,terrain=terrain,arm=arm,encoder_sha256=study.RUNTIME_SHA,
                python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all())
            original_save(path,dict(infos or {},adaptation_state=state))
            record.update(completed_learning_iterations=completed,
                completed_environment_transitions=completed*study.NUM_ENVS*study.STEPS,
                collected_environment_transitions=collected,
                latest_checkpoint=str(path),latest_checkpoint_sha256=study.sha(path),updated_at=study.now())
            study.write(manifest_path,record)
        runner.save=checked_save
        dump_yaml(directory/"env.yaml",asdict(cfg));dump_yaml(directory/"agent.yaml",asdict(agent))
        study.write(manifest_path,record)
        remaining=iterations-carried_updates
        if remaining<0:
            raise ValueError("resume checkpoint exceeds frozen training budget")
        if remaining:
            runner.learn(num_learning_iterations=remaining,init_at_random_ep_len=True)
        else:
            runner.current_learning_iteration=iterations-1
            runner.save(str(directory/f"model_{iterations-1}.pt"))
        if collected != remaining*study.NUM_ENVS*study.STEPS:
            raise ValueError("collected transition count differs from planned remaining budget")
        final=directory/f"model_{iterations-1}.pt"
        if not final.is_file() or study.base.state_sha(bundle.encoder)!=encoder_initial:
            raise ValueError("final checkpoint missing or encoder changed")
        import onnxruntime as ort
        # Match the CPU float32 backend used by the exported evaluation policy.
        # Training may use TF32; comparing it to ONNX CPU confounds export parity.
        policy=runner.get_inference_policy(device="cpu")
        obs=env.get_observations()
        with torch.inference_mode():
            expected=policy(obs.to("cpu"),stochastic_output=False).detach().numpy()
        options=ort.SessionOptions()
        options.intra_op_num_threads=options.inter_op_num_threads=1
        exported=ort.InferenceSession(str(directory/"policy.onnx"),sess_options=options,
                                      providers=["CPUExecutionProvider"])
        inputs=obs["actor"].detach().cpu().numpy().astype(np.float32)
        actual=np.concatenate([exported.run(None,{exported.get_inputs()[0].name:x[None]})[0]
                               for x in inputs])
        np.testing.assert_allclose(actual,expected,rtol=2e-4,atol=2e-5)
        record["torch_onnx_parity_reference"]="PyTorch CPU float32 vs ONNX Runtime CPU float32"
        record["torch_onnx_parity_max_abs_error"]=float(np.max(np.abs(actual-expected)))
        study.verify_runtime()
        record.update(status="COMPLETED",ended_at=study.now(),policy_sha256=study.sha(directory/"policy.onnx"),
            final_checkpoint=str(final),final_checkpoint_sha256=study.sha(final),encoder_frozen_verified=True,
            completed_learning_iterations=iterations,completed_environment_transitions=record["environment_transition_budget"])
    except BaseException as error:
        record.update(status="FAILED",ended_at=study.now(),error=repr(error),traceback=traceback.format_exc(),
            resume_command=[sys.executable,"-u",str(Path(__file__)),"--stage","train","--terrain",terrain,
                            "--arm",arm,"--seed",str(seed),"--resume"])
        raise
    finally:
        record["collected_environment_transitions"]=collected
        study.write(manifest_path,record)
        if env is not None:
            env.close()


def evaluate(terrain,seed,label):
    if seed not in study.TRAIN_SEEDS:
        raise ValueError("PPO seed is outside the frozen evaluation matrix")
    study.verify_assets()
    from scripts import g1_best_phase_batched as batch
    study.base.ENCODER=study.RUNTIME
    batch.ENCODER=study.RUNTIME
    batch.make_cfg=study.make_cfg
    job=next(j for j in study.evaluation_jobs(terrain,seed) if j["label"]==label)
    training=json.loads((Path(job["policy"]).parent/"manifest.json").read_text(encoding="utf-8"))
    if training["status"]!="COMPLETED" or study.sha(job["policy"])!=training["policy_sha256"]:
        raise ValueError("evaluation policy is incomplete or changed")
    target=Path(job["output"])
    if (target/"manifest.json").exists():
        m=json.loads((target/"manifest.json").read_text(encoding="utf-8"))
        if m["status"]=="COMPLETED" and m["episode_count"]==30 and m["policy_sha256"]==training["policy_sha256"]:
            return
        raise FileExistsError("incomplete evaluation attempt preserved; archive explicitly before rerun")
    batch.rollout_batched(SimpleNamespace(**job,terrain=terrain,scenarios=list(study.SCENARIOS),
                                         seeds=list(study.EVAL_SEEDS),horizon=20.))


def summarize():
    study.verify_assets()
    import numpy as np
    from statsmodels.stats.multitest import multipletests
    from scripts.summarize_g1_best_phase_control import score,aggregate
    rows,aggregates,pairing=[],[],[]
    for terrain in ("flat","rough"):
        for seed in study.TRAIN_SEEDS:
            training={arm:json.loads((study.train_dir(terrain,arm,seed)/"manifest.json").read_text(encoding="utf-8"))
                      for arm in ("clock","learned")}
            fields=("source_actor_sha256","initial_actor_state_sha256","initial_critic_state_sha256",
                "iterations","num_envs","num_steps_per_env","completed_environment_transitions",
                "completed_learning_iterations","common_task_config_sha256","ppo_config_sha256","startup_model_hashes")
            for key in fields:
                if training["clock"][key]!=training["learned"][key]:
                    raise ValueError(f"training pair mismatch: {terrain}/{seed}/{key}")
            for record in training.values():
                if (record["status"]!="COMPLETED" or record["iterations"]!=study.ITERATIONS
                        or record["completed_learning_iterations"]!=study.ITERATIONS
                        or record["completed_environment_transitions"]!=study.TRANSITIONS
                        or record["num_envs"]!=study.NUM_ENVS or not record["encoder_frozen_verified"]):
                    raise ValueError("training completion does not satisfy the frozen budget")
            raw_groups={}
            manifests={}
            for job in study.evaluation_jobs(terrain,seed):
                manifest,current,raw=score(Path(job["output"]))
                expected={(terrain,c,s) for s in study.EVAL_SEEDS for c in study.SCENARIOS}
                if set(raw)!=expected or len(current)!=30:
                    raise ValueError("missing evaluation cohort")
                if manifest["encoder_sha256"]!=study.RUNTIME_SHA:
                    raise ValueError("evaluation encoder mismatch")
                expected_training=training["learned" if job["label"]=="learned_learned" else "clock"]
                if manifest["policy_sha256"]!=expected_training["policy_sha256"]:
                    raise ValueError("wrong trained policy in evaluation")
                manifests[job["label"]]=manifest
                raw_groups[job["label"]]=raw
                for r in current:
                    r.update(train_seed=seed,label=job["label"])
                rows.extend(current)
                aggregates.append(dict(terrain=terrain,train_seed=seed,label=job["label"],**aggregate(current)))
            for label in ("clock_learned","learned_learned"):
                if manifests[label]["startup_model_hashes"]!=manifests["clock_clock"]["startup_model_hashes"]:
                    raise ValueError("evaluation startup physics differ")
                for key,a in raw_groups["clock_clock"].items():
                    b=raw_groups[label][key]
                    errors={k:float(np.max(np.abs(a[k][0].astype(float)-b[k][0].astype(float))))
                            for k in ("position","quaternion","joint_pos","velocity")}
                    if max(errors.values())>1e-6:
                        raise ValueError("evaluation initial states are not paired")
                    n=min(len(a["command"]),len(b["command"]))
                    np.testing.assert_array_equal(a["command"][:n],b["command"][:n])
                    pairing.append(dict(train_seed=seed,terrain=terrain,label=label,key=key,errors=errors))
    families=[]
    for control in ("clock_clock","clock_learned"):
        effects=[study.paired_training_effect(rows,t,"learned_learned",control,metric)
                 for t,metric in (("rough","fall_rate"),("rough","mae_forward_velocity_mps"),
                                  ("flat","mae_forward_velocity_mps"))]
        adjusted=multipletests([e["two_sided_p"] for e in effects],method="holm")[1]
        for effect,p in zip(effects,adjusted):
            effect["holm_adjusted_p"]=float(p)
        passed=all(e["paired_t_95pct_ci"][1]<0 and e["holm_adjusted_p"]<.05 for e in effects[:2])
        passed=passed and effects[2]["paired_t_95pct_ci"][1]<=0
        families.append(dict(treatment="learned_learned",control=control,effects=effects,
                             fixed_budget_improvement_established=bool(passed)))
    result=dict(schema="g1.phase_policy_adaptation.comparison.v1",computed_at=study.now(),
        scope="finite-budget warm-start simulation; three PPO seeds; fixed small terrain cohort",
        asset_seal_sha256=study.sha(study.OUT/"asset_seal.json"),inference=families,
        aggregates=aggregates,pairing_checks=pairing,episodes=rows)
    study.write(study.OUT/"comparison.json",result)
    lines=["# 冻结 TCTST 的控制策略适配：三组结果","",
        "每臂 501 更新、64 环境、769536 transition；3 配对 PPO seed。报告按训练 seed 分层，评测回合不当作独立训练。","",
        "| 地形 | PPO seed | 训练→测试 | 跌倒/n | 速度 MAE(m/s) | 存活(s) |",
        "|---|---:|---|---:|---:|---:|"]
    for r in aggregates:
        lines.append(f"| {r['terrain']} | {r['train_seed']} | {r['label']} | {r['falls']}/{r['episodes']} | {r['mae_forward_velocity_mps']:.4f} | {r['survival_seconds']:.2f} |")
    for family in families:
        lines += ["",f"learned→learned 对 {family['control']}：固定预算改善{'已建立' if family['fixed_budget_improvement_established'] else '尚未建立'}。"]
        for e in family["effects"]:
            lines.append(f"- {e['terrain']} / {e['metric']}：差 {e['mean_difference']:.5f}，训练 seed 的 95% t 区间 {e['paired_t_95pct_ci']}；Holm p={e['holm_adjusted_p']:.5g}。")
    lines += ["","这是有限预算微调尝试，不代表从零充分训练或真机效果；未建立改善不能解释为等效或原理被否定。",""]
    (study.OUT/"comparison.md").write_text("\n".join(lines),encoding="utf-8")


def child(stage, log_name, **kwargs):
    logs=study.OUT/"logs"
    logs.mkdir(parents=True,exist_ok=True)
    cmd=[sys.executable,"-u",str(Path(__file__)),"--stage",stage]
    for key,value in kwargs.items():
        if value is True:
            cmd.append("--"+key.replace('_','-'))
        elif value is not False:
            cmd += ["--"+key.replace('_','-'),str(value)]
    env=dict(os.environ,PYTHONHASHSEED=str(kwargs.get("seed",20261400)),PYTHONIOENCODING="utf-8",WANDB_MODE="disabled")
    log=logs/f"{log_name}_{time.time_ns()}.log"
    with log.open('w',encoding='utf-8') as stream:
        process=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
        state=dict(status="RUNNING",stage=stage,task=log_name,child_pid=process.pid,argv=cmd,
                   log_path=str(log),updated_at=study.now())
        study.write(study.OUT/"queue_status.json",state)
        code=process.wait()
    if code:
        raise RuntimeError(f"child failed ({code}): {log}")


def queue(prior_pid):
    import psutil
    study.OUT.mkdir(parents=True,exist_ok=True)
    lock_path=study.OUT/"queue.lock"
    if lock_path.exists():
        previous=json.loads(lock_path.read_text())
        if psutil.pid_exists(previous["pid"]):
            raise RuntimeError("another adaptation queue is already active")
    status_path=study.OUT/"queue_status.json"
    if status_path.exists():
        previous=json.loads(status_path.read_text(encoding="utf-8"))
        latest=previous.get("last_stage",previous)
        pid=latest.get("child_pid")
        if pid and psutil.pid_exists(pid):
            try:
                command=" ".join(psutil.Process(pid).cmdline())
            except psutil.NoSuchProcess:
                command=""
            if Path(__file__).name in command:
                raise RuntimeError(f"previous queue child is still active: PID {pid}; do not duplicate GPU work")
    study.write(lock_path,dict(pid=os.getpid(),started_at=study.now()))
    try:
        while True:
            alive=psutil.pid_exists(prior_pid) if prior_pid else False
            ready=study.dependency_ready()
            if ready and not alive:
                break
            study.write(study.OUT/"queue_status.json",dict(status="WAITING_FOR_PREVIOUS_STUDY",
                prior_pid=prior_pid,prior_process_alive=alive,prior_episode_matrix_complete=ready,
                updated_at=study.now()))
            if not alive and not ready:
                raise RuntimeError("prior study stopped before the required episode matrix completed")
            time.sleep(30)
        # All old data stay unchanged. This separate repair only corrects aggregation bugs.
        with (study.OUT/"previous_study_summary.log").open('w',encoding='utf-8') as log:
            subprocess.run([sys.executable,"-u","-m","scripts.summarize_g1_selected_phase_control_v3"],
                           cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        for terrain in ("flat","rough"):
            for arm in ("clock","learned"):
                child("smoke",f"smoke_{terrain}_{arm}",terrain=terrain,arm=arm,seed=20261400)
        study.freeze_assets()
        for job in study.training_jobs():
            child("train",f"train_{job['terrain']}_{job['arm']}_{job['seed']}",**job)
        for terrain in ("flat","rough"):
            for seed in study.TRAIN_SEEDS:
                for job in study.evaluation_jobs(terrain,seed):
                    child("evaluate",f"eval_{terrain}_{seed}_{job['label']}",terrain=terrain,seed=seed,label=job["label"])
        child("summarize","final_analysis")
        study.write(study.OUT/"queue_status.json",dict(status="COMPLETED",ended_at=study.now(),
            comparison=str(study.OUT/"comparison.json")))
    except BaseException as error:
        previous=json.loads((study.OUT/"queue_status.json").read_text(encoding="utf-8")) if (study.OUT/"queue_status.json").exists() else {}
        study.write(study.OUT/"queue_status.json",dict(status="FAILED",ended_at=study.now(),
            error=repr(error),traceback=traceback.format_exc(),last_stage=previous))
        raise
    finally:
        lock_path.unlink(missing_ok=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage",choices=("queue","smoke","train","evaluate","summarize"),required=True)
    p.add_argument("--terrain",choices=("flat","rough"),default="flat")
    p.add_argument("--arm",choices=("clock","learned"),default="clock")
    p.add_argument("--seed",type=int,default=study.TRAIN_SEEDS[0])
    p.add_argument("--prior-pid",type=int,default=37904)
    p.add_argument("--label",choices=("clock_clock","clock_learned","learned_learned"))
    p.add_argument("--resume",action="store_true")
    a=p.parse_args()
    if a.stage=="queue": queue(a.prior_pid)
    elif a.stage in ("train","smoke"): train(a.terrain,a.arm,a.seed,smoke=a.stage=="smoke",resume=a.resume)
    elif a.stage=="evaluate": evaluate(a.terrain,a.seed,a.label)
    else: summarize()


if __name__=="__main__":
    main()
