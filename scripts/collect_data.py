"""Collect trajectory data (q, qd, qdd, tau) from G1 simulation for PINN training."""
import torch
import numpy as np
import time
import os
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

# Load G1 flat terrain config
env_cfg = load_env_cfg("Mjlab-Velocity-Flat-Unitree-G1")
env_cfg.scene.num_envs = 128

# Determine device
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Creating env on {device} with {env_cfg.scene.num_envs} environments...")
env = ManagerBasedRlEnv(cfg=env_cfg, device=device)

num_envs = env.num_envs
total_steps = 500
print(f"Collecting {total_steps} steps × {num_envs} envs = {total_steps * num_envs} samples")

# Reset environment first
obs_dict, extras = env.reset()

all_q, all_qd, all_qdd, all_tau = [], [], [], []

t_start = time.time()
for step in range(total_steps):
    # Random actions around default pose for diverse dynamics data
    action_dim = env.action_manager.total_action_dim
    actions = torch.randn(num_envs, action_dim, device=device) * 0.1

    obs_dict, reward, terminated, truncated, extras = env.step(actions)

    # Extract joint dynamics from asset
    asset = env.scene["robot"]
    q = asset.data.joint_pos.clone()
    qd = asset.data.joint_vel.clone()
    qdd = asset.data.joint_acc.clone()
    tau = asset.data.actuator_force.clone()

    all_q.append(q.cpu().numpy())
    all_qd.append(qd.cpu().numpy())
    all_qdd.append(qdd.cpu().numpy())
    all_tau.append(tau.cpu().numpy())

    if (step + 1) % 100 == 0:
        elapsed = time.time() - t_start
        rate = (step + 1) * num_envs / elapsed
        print(f"  Step {step + 1}/{total_steps} | {rate:.0f} samples/s")

# Stack
all_q = np.stack(all_q)
all_qd = np.stack(all_qd)
all_qdd = np.stack(all_qdd)
all_tau = np.stack(all_tau)

os.makedirs("/root/unitree_rl_mjlab/data", exist_ok=True)
np.savez("/root/unitree_rl_mjlab/data/g1_trajectory.npz",
         q=all_q, qd=all_qd, qdd=all_qdd, tau=all_tau)

elapsed = time.time() - t_start
print(f"\nDone! {all_q.shape[0] * all_q.shape[1]} samples in {elapsed:.1f}s")
print(f"Shape: {all_q.shape} (steps, envs, dofs={all_q.shape[2]})")
print(f"Saved to /root/unitree_rl_mjlab/data/g1_trajectory.npz")

env.close()
