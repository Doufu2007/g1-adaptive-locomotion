"""Explicit G1 adaptations of pinned ModernTCN/FLD and paper-based TCTST.

Inputs always end at the current frame. The paper implementations are adapted
to 29 joint positions; these are not pretrained models or original benchmarks.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from scripts.g1_external_vendor.modern_blocks import Block
from scripts.g1_external_vendor.fld import FLD


def unit_phase(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, dim=-1, eps=1e-8)


def causal_windows(frames: torch.Tensor, ends: torch.Tensor, window: int,
                   starts: torch.Tensor) -> torch.Tensor:
    """Gather past-only windows, repeating the first observed frame on startup."""
    offsets = torch.arange(1-window, 1, device=ends.device)
    indices = torch.maximum(ends[:, None]+offsets, starts[:, None])
    return frames[indices]


def phase_rates(cycles: np.ndarray, sequences: np.ndarray, frames: np.ndarray,
                fps: float = 50.) -> tuple[np.ndarray, np.ndarray]:
    rate = np.zeros(len(cycles), dtype=np.float32)
    valid = np.zeros(len(cycles), dtype=bool)
    delta = (np.diff(cycles.astype(np.float64))+.5) % 1. - .5
    adjacent = (sequences[1:] == sequences[:-1]) & (np.diff(frames) == 1)
    valid[1:] = adjacent & (delta > 0) & (delta*fps < 10.)
    rate[1:] = np.where(valid[1:], delta*fps, 0.)
    return rate, valid


def phase_rate_output(raw: torch.Tensor) -> torch.Tensor:
    raw = raw.float()
    return torch.cat((unit_phase(raw[:, :2]), F.softplus(raw[:, 2:3])), dim=-1)


class ModernPhase(nn.Module):
    """Official ModernTCN blocks with a 10-frame stem and phase regression head."""
    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(1, dim, 1), nn.BatchNorm1d(dim))
        self.blocks = nn.Sequential(*[
            Block(7, 3, dim, dim*2, 29, drop=.1) for _ in range(2)
        ])
        self.head = nn.Linear(29*dim*10, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1:] != (10, 29):
            raise ValueError('ModernPhase requires [batch,10,29]')
        b, t, m = x.shape
        h = self.stem(x.transpose(1, 2).reshape(b*m, 1, t))
        h = self.blocks(h.reshape(b, m, -1, t))
        return unit_phase(self.head(F.gelu(h).flatten(1)).float())


class TCTSTPhase(nn.Module):
    """Paper Fig.3 shared channel embedding + across-channel Transformer."""
    def __init__(self, window: int = 100, dim: int = 384, heads: int = 8,
                 layers: int = 8) -> None:
        super().__init__()
        self.window, self.dim = window, dim
        # Total downsampling is 3*3*2. Left-only padding right-aligns every
        # window so the current frame cannot be discarded by floor pooling.
        padded_window = ((window+17)//18)*18
        self.conv = nn.Sequential(nn.ReplicationPad1d((padded_window-window,0)),
                                  nn.Conv1d(1,4,3,stride=3),nn.ReLU(),
                                  nn.Conv1d(4,4,3,stride=3),nn.ReLU(),nn.MaxPool1d(2),nn.Flatten())
        self.embedding = nn.Linear(4*(padded_window//18), dim)
        self.channel_position = nn.Parameter(torch.randn(1,32,1)*.02)
        layer = nn.TransformerEncoderLayer(dim, heads, 4*dim, dropout=.1,
                                           activation='relu', batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.Linear(dim,128),nn.ReLU(),nn.Linear(128,3))

    def encode_joint(self, joint: torch.Tensor) -> torch.Tensor:
        b,c,t = joint.shape
        h = self.embedding(self.conv(joint.reshape(b*c,1,t))).reshape(b,c,self.dim)
        return self.transformer(h+self.channel_position)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fine-tuning and inference NEVER supply phase targets as observations.
        sensor = x.transpose(1,2)
        joint = torch.cat((sensor, sensor.new_zeros(len(x),3,self.window)),dim=1)
        return phase_rate_output(self.head(self.encode_joint(joint).mean(dim=1)))


class TCTSTMaskedModel(nn.Module):
    def __init__(self, phase_model: TCTSTPhase) -> None:
        super().__init__()
        self.phase_model=phase_model
        layer=nn.TransformerEncoderLayer(phase_model.dim,8,4*phase_model.dim,
                                         dropout=.1,activation='relu',batch_first=True)
        self.decoder=nn.TransformerEncoder(layer,8,enable_nested_tensor=False)
        self.reconstruct=nn.Linear(phase_model.dim,phase_model.window)

    def forward(self, joint: torch.Tensor) -> torch.Tensor:
        return self.reconstruct(self.decoder(self.phase_model.encode_joint(joint)))


def masked_reconstruction_loss(prediction: torch.Tensor, target: torch.Tensor,
                               masked_channels: torch.Tensor,
                               valid: torch.Tensor) -> torch.Tensor:
    selected = masked_channels[:,:,None] & valid
    return ((prediction-target).square()*selected).sum()/selected.sum().clamp_min(1)


class StableFLD(FLD):
    """Numerical/Hz safeguards around the otherwise pinned official model."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for encoder in self.phase_encoder:
            encoder.add_module('nonzero_pair', NonzeroPhasePair())

    def fft(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        x=x.float()  # FFT remains float32 under mixed precision.
        rfft=torch.fft.rfft(x,dim=2)
        power=rfft[:,:,1:].abs().square()
        total=power.sum(dim=2)
        freqs=torch.fft.rfftfreq(self.history_horizon,d=self.dt,device=x.device)[1:]
        weighted=(freqs*power).sum(dim=2)/total.clamp_min(1e-10)
        frequency=torch.where(total>1e-10,weighted,torch.ones_like(weighted))
        amplitude=2*total.clamp_min(1e-10).sqrt()/self.history_horizon
        offset=rfft.real[:,:,0]/self.history_horizon
        return frequency,amplitude,offset


class NonzeroPhasePair(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x=x.float()
        fallback=torch.zeros_like(x)
        fallback[:,0]=1e-4
        return torch.where(x.square().sum(dim=-1,keepdim=True)>1e-12,x,fallback)


def endpoint_parameters(params: list[torch.Tensor], horizon: int,
                        dt: float) -> torch.Tensor:
    phase,freq,amp,offset=params
    theta=2*torch.pi*(phase+freq*((horizon-1)*dt/2))
    return torch.cat((theta.sin(),theta.cos(),freq,amp,offset),dim=-1)


class FLDPhase(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder=StableFLD(29,51,8,torch.device('cpu'),dt=.02,
                               encoder_shape=[64,64],decoder_shape=[64,64])
        self.head=nn.Sequential(nn.Linear(40,64),nn.ReLU(),nn.Linear(64,3))

    def features(self,x:torch.Tensor) -> torch.Tensor:
        # Upstream stores tensors outside register_buffer; follow model device.
        self.encoder.device=x.device
        self.encoder.args=self.encoder.args.to(x.device)
        self.encoder.freqs=self.encoder.freqs.to(x.device)
        _,_,_,params=self.encoder(x.transpose(1,2))
        return endpoint_parameters(params,51,.02)

    def forward(self,x:torch.Tensor) -> torch.Tensor:
        return phase_rate_output(self.head(self.features(x)))


def track_phase(anchors: np.ndarray, rates: np.ndarray, sequences: np.ndarray,
                frames: np.ndarray, *, alpha: float = .1, dt: float = .02) -> np.ndarray:
    """Causal observer: integrate predicted rate and correct towards phase anchor.

    Reset only at a new recording or real frame gap. No reference phase or
    touchdown event is accepted by this interface.
    """
    result=np.empty_like(anchors,dtype=np.float32)
    state=0.
    for i in range(len(anchors)):
        anchor=float(np.arctan2(anchors[i,0],anchors[i,1]))
        if i==0 or sequences[i]!=sequences[i-1] or frames[i]!=frames[i-1]+1:
            state=anchor
        else:
            propagated=state+float(rates[i])*dt*2*np.pi
            innovation=float(np.arctan2(np.sin(anchor-propagated),np.cos(anchor-propagated)))
            state=propagated+alpha*innovation
        result[i]=[np.sin(state),np.cos(state)]
    return result
