"""Frozen E model online inference; exact causal layer caches and per-world reset."""
from pathlib import Path
import torch
from torch.nn import functional as F

from scripts.g1_stateful_phase import DT, PhaseNet, sha


def motion_features(q, quat, *, previous_q=None, previous_quat=None):
    if q.ndim != 2 or q.shape[1] != 29 or quat.shape != (len(q), 4):
        raise ValueError('expected aligned [env,29] joints and WXYZ quaternion')
    quat = F.normalize(quat, dim=-1)
    w, x, y, z = quat.unbind(-1)
    gravity = torch.stack((2*(w*y-x*z), -2*(y*z+w*x), 2*(x*x+y*y)-1), -1)
    velocity = torch.zeros_like(q) if previous_q is None else (q-previous_q)/DT
    if previous_quat is None:
        gyro = torch.zeros_like(gravity)
    else:
        old = F.normalize(previous_quat, dim=-1)
        a, b = old[:, :1], old[:, 1:]
        c, d = quat[:, :1], quat[:, 1:]
        scalar = a*c + (b*d).sum(-1, keepdim=True)
        vector = a*d-c*b-torch.linalg.cross(b, d)
        sign = torch.where(scalar < 0, -1., 1.)
        scalar, vector = scalar*sign, vector*sign
        size = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
        angle = 2*torch.atan2(size, scalar)
        scale = torch.where(size > 1e-7, angle/size.clamp_min(1e-12), 2*torch.ones_like(size))/DT
        gyro = vector*scale
    return torch.cat((q, velocity, gravity, gyro), -1)


class OnlinePhase:
    def __init__(self, model, mean, std, *, num_envs, device):
        self.encoder = model.to(device).eval().requires_grad_(False)
        self.mean = torch.as_tensor(mean, dtype=torch.float32, device=device)
        self.std = torch.as_tensor(std, dtype=torch.float32, device=device)
        if (self.mean.shape != (64,) or self.std.shape != (64,)
                or not torch.isfinite(self.mean).all() or not torch.isfinite(self.std).all()
                or not (self.std > 0).all() or model.channels != 64 or model.mode != 'observer'):
            raise ValueError('E model requires finite 64-channel train normalization and observer')
        self.num_envs, self.device = num_envs, device
        self.history = [torch.zeros(num_envs, b.conv.in_channels, b.padding, device=device)
                        for b in self.encoder.blocks]
        self.previous_q = torch.zeros(num_envs, 29, device=device)
        self.previous_quat = torch.zeros(num_envs, 4, device=device)
        self.previous_quat[:, 0] = 1
        self.phase = torch.zeros(num_envs, device=device)
        self.output = torch.zeros(num_envs, 2, device=device)
        self.last_stamp = torch.full((num_envs,), -1, dtype=torch.long, device=device)
        self.update_counts = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.initialized = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def reset(self, ids=None):
        if ids is None:
            ids = torch.arange(self.num_envs, device=self.device)
        for history in self.history:
            history[ids] = 0
        self.last_stamp[ids] = -1
        self.initialized[ids] = False
        self.update_counts[ids] = 0
        self.phase[ids] = 0
        self.output[ids] = 0

    @torch.inference_mode()
    def observe(self, q, quat, stamp):
        if q.shape != (self.num_envs, 29) or quat.shape != (self.num_envs, 4):
            raise ValueError('online phase environment dimensions changed')
        ids = torch.nonzero(self.last_stamp != int(stamp), as_tuple=False).flatten()
        if not len(ids):
            return self.output
        position, rotation = q[ids], quat[ids]
        if (not torch.isfinite(position).all() or not torch.isfinite(rotation).all()
                or bool((torch.linalg.vector_norm(rotation, dim=-1) < 1e-6).any())):
            raise ValueError('nonfinite joints or invalid quaternion in phase inference')
        initial = ~self.initialized[ids]
        old_q = torch.where(initial[:, None], position, self.previous_q[ids])
        old_rot = torch.where(initial[:, None], rotation, self.previous_quat[ids])
        value = (motion_features(position, rotation, previous_q=old_q,
                                 previous_quat=old_rot)-self.mean)/self.std
        for block, history in zip(self.encoder.blocks, self.history):
            saved = history[ids]
            weights = block.conv.weight
            dilation = block.conv.dilation[0]
            hidden = (F.linear(saved[:, :, 0], weights[:, :, 0])
                      + F.linear(saved[:, :, dilation], weights[:, :, 1])
                      + F.linear(value, weights[:, :, 2], block.conv.bias))
            skip = value if isinstance(block.skip, torch.nn.Identity) else F.linear(
                value, block.skip.weight[:, :, 0], block.skip.bias)
            history[ids] = torch.cat((saved[:, :, 1:], value[:, :, None]), dim=-1)
            value = F.silu(hidden+skip)
        raw = self.encoder.head(value)
        anchor = torch.atan2(raw[:, 0], raw[:, 1])
        rate, gain = 4*torch.sigmoid(raw[:, 2]), .25*torch.sigmoid(raw[:, 3])
        step = rate*(2*torch.pi*DT)
        residual = anchor-self.phase[ids]-step
        innovation = torch.atan2(torch.sin(residual), torch.cos(residual))
        phase = torch.where(initial, anchor, self.phase[ids]+(step+gain*innovation).clamp(0, torch.pi/3))
        self.phase[ids] = phase
        self.output[ids] = torch.stack((torch.sin(phase), torch.cos(phase)), -1)
        self.previous_q[ids], self.previous_quat[ids] = position, rotation
        self.last_stamp[ids] = int(stamp)
        self.initialized[ids] = True
        self.update_counts[ids] += 1
        if not torch.isfinite(self.output).all():
            raise ValueError('nonfinite frozen phase output')
        return self.output


def load_online(path, *, num_envs, device):
    from scripts import g1_stateful_flat_ppo as study
    path = Path(path).resolve()
    if path != study.RUNTIME.resolve() or sha(path) != study.RUNTIME_SHA:
        raise ValueError('online adapter only accepts the sealed E checkpoint')
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if (checkpoint['row'] != 'observer_sensors_mixed' or checkpoint['seed'] != study.ENCODER_SEED
            or checkpoint['completed_updates'] != 2000 or checkpoint['channels'] != 64
            or checkpoint['mode'] != 'observer'):
        raise ValueError('unexpected E model identity')
    with torch.random.fork_rng(devices=[]):
        model = PhaseNet(64, 'observer')
        model.load_state_dict(checkpoint['model'], strict=True)
    return OnlinePhase(model, checkpoint['mean'], checkpoint['std'], num_envs=num_envs, device=device)


def bundle(env, encoder_path):
    if abs(float(env.step_dt)-DT) > 1e-9:
        raise ValueError('online E phase requires 50 Hz control')
    from scripts import g1_stateful_flat_ppo as study
    if Path(encoder_path).resolve() != study.RUNTIME.resolve():
        raise ValueError('wrong E checkpoint path')
    runtime = getattr(env, '_new_e_online_phase', None)
    if runtime is None:
        runtime = load_online(encoder_path, num_envs=env.num_envs, device=env.device)
        env._new_e_online_phase = runtime
    return runtime


def learned_phase(env, encoder_path):
    from src.tasks.velocity.mdp.observations import _joint_positions_in_order
    from src.tasks.velocity.phasor.data import G1_JOINT_ORDER
    runtime = bundle(env, encoder_path)
    robot = env.scene['robot']
    q = _joint_positions_in_order(robot, robot.data.joint_pos, G1_JOINT_ORDER)
    return runtime.observe(q, robot.data.root_link_quat_w, env.common_step_counter)


def reset_phase(env, env_ids, encoder_path):
    bundle(env, encoder_path).reset(env_ids)


_installed = False


def install():
    """Process-local adapter; legacy checkpoint paths retain their original behavior."""
    global _installed
    if _installed:
        return
    from scripts import g1_stateful_flat_ppo as study
    from src.tasks.velocity.mdp import observations
    previous_load = observations._load_phasor_encoder
    previous_phase = observations.learned_phase
    previous_reset = observations.reset_learned_phase_history
    def is_e(path):
        return Path(path).resolve() == study.RUNTIME.resolve()
    def load(env, encoder_path):
        return bundle(env, encoder_path) if is_e(encoder_path) else previous_load(env, encoder_path)
    def phase(env, encoder_path):
        return learned_phase(env, encoder_path) if is_e(encoder_path) else previous_phase(env, encoder_path)
    def reset(env, env_ids, encoder_path):
        return reset_phase(env, env_ids, encoder_path) if is_e(encoder_path) else previous_reset(env, env_ids, encoder_path)
    phase.__name__ = 'learned_phase'
    observations._load_phasor_encoder = load
    observations.learned_phase = phase
    observations.reset_learned_phase_history = reset
    _installed = True
