"""Boundary tests for the external-model experiment, independent of real scores."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.g1_external_models import (  # noqa: E402
    causal_windows, phase_rates, endpoint_parameters, track_phase,
    ModernPhase, TCTSTPhase, FLDPhase, masked_reconstruction_loss,
)


def test_history_is_causal_left_padded_and_keeps_cycle_start():
    frames = torch.arange(8, dtype=torch.float32)[:, None]
    x = causal_windows(frames, torch.tensor([0, 2, 6]), 4, torch.tensor([0, 0, 5]))
    torch.testing.assert_close(x[:, :, 0], torch.tensor([[0., 0., 0., 0.], [0., 0., 1., 2.], [5., 5., 5., 6.]]))
    changed = frames.clone(); changed[3:5] = 999
    torch.testing.assert_close(causal_windows(changed, torch.tensor([2]), 4, torch.tensor([0])), x[1:2])


def test_phase_rate_wrap_and_sequence_boundary():
    rate, valid = phase_rates(np.array([.94, .98, .02, .7]), np.array(['a','a','a','b']), np.array([10,11,12,0]))
    np.testing.assert_allclose(rate[1:3], [2.,2.], atol=1e-5)
    assert valid.tolist() == [False, True, True, False]


def test_fld_center_phase_is_advanced_to_endpoint():
    params = [torch.zeros(2, 1), torch.ones(2, 1), torch.ones(2, 1), torch.zeros(2, 1)]
    features = endpoint_parameters(params, horizon=51, dt=.02)
    torch.testing.assert_close(features[:, :2], torch.tensor([[0.,-1.],[0.,-1.]]), atol=1e-6, rtol=0)


def test_tracking_progresses_without_reference_and_resets_only_at_boundaries():
    anchors = np.tile([0.,1.], (54,1))
    phase = track_phase(anchors, np.ones(54), np.array(['a']*51+['b']*3), np.r_[np.arange(51),np.arange(3)], alpha=0.)
    np.testing.assert_allclose(np.unwrap(np.arctan2(phase[:51,0],phase[:51,1]))[-1], 2*np.pi, atol=1e-6)
    np.testing.assert_allclose(phase[51], [0.,1.], atol=1e-6)
    prefix = track_phase(anchors[:25], np.ones(25), np.array(['a']*25), np.arange(25), alpha=.1)
    whole = track_phase(anchors, np.ones(54), np.array(['a']*51+['b']*3), np.r_[np.arange(51),np.arange(3)], alpha=.1)
    np.testing.assert_allclose(prefix,whole[:25],atol=1e-7)


def test_reconstruction_ignores_unmasked_and_missing_targets():
    pred=torch.tensor([[[1.,100.],[100.,100.]]],requires_grad=True)
    target=torch.zeros_like(pred)
    valid=torch.tensor([[[True,False],[True,True]]])
    loss=masked_reconstruction_loss(pred,target,torch.tensor([[True,False]]),valid)
    torch.testing.assert_close(loss,torch.tensor(1.))
    loss.backward()
    torch.testing.assert_close(pred.grad,torch.tensor([[[2.,0.],[0.,0.]]]))


def test_data_reader_rejects_test_before_opening_archives():
    from scripts.run_g1_external_methods import FrameBank
    with pytest.raises(ValueError,match='test split is forbidden'):
        FrameBank('test')


def test_tctst_downsampling_uses_current_frame():
    model = TCTSTPhase(dim=48, heads=4, layers=1)
    for layer in model.conv:
        if isinstance(layer, torch.nn.Conv1d):
            torch.nn.init.ones_(layer.weight)
            torch.nn.init.ones_(layer.bias)
    baseline = torch.zeros(1, 1, 100)
    current_changed = baseline.clone()
    current_changed[0, 0, -1] = 1.
    current_changed.requires_grad_()
    embedded = model.conv(current_changed)
    assert not torch.equal(embedded, model.conv(baseline))
    embedded.sum().backward()
    assert current_changed.grad[0, 0, -1] > 0


@pytest.mark.parametrize('kind,window', [('modern',10),('tctst',100),('fld',51)])
def test_constant_inputs_have_finite_phase_rate_and_gradients(kind,window):
    torch.manual_seed(41)
    if kind=='modern': model=ModernPhase()
    elif kind=='tctst': model=TCTSTPhase(dim=48,heads=4,layers=1)
    else: model=FLDPhase()
    x=torch.zeros(3,window,29)
    out=model(x)
    assert out.shape==(3,2 if kind=='modern' else 3)
    assert torch.isfinite(out).all()
    out.square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    torch.testing.assert_close(out[:,:2].norm(dim=-1),torch.ones(3),atol=1e-5,rtol=1e-5)
    if kind!='modern': assert (out[:,2]>=0).all()
