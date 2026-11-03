import sys
from pathlib import Path

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_g1_external_methods_v2 import warmup_scale,state_loss,RawStateTCTST


def test_paper_warmup_endpoints_and_fixed_after20():
    assert warmup_scale(0)==.2
    assert warmup_scale(19)==warmup_scale(31)==1.
    assert all(warmup_scale(i)<warmup_scale(i+1) for i in range(19))


def test_raw_state_loss_has_useful_gradient_at_zero_phase():
    raw=torch.tensor([[0.,0.,1.]],requires_grad=True)
    loss=state_loss(raw,torch.tensor([[1.,0.]]),torch.ones(1),torch.tensor([True]))
    loss.backward()
    torch.testing.assert_close(raw.grad,torch.tensor([[-1.,0.,0.]]))


def test_raw_state_and_normalized_inference_match_direction():
    model=RawStateTCTST(dim=48,heads=4,layers=1).eval()
    x=torch.randn(2,100,29)
    raw=model.raw_state(x);normalized=model(x)
    torch.testing.assert_close(normalized[:,:2],torch.nn.functional.normalize(raw[:,:2],dim=-1))
    torch.testing.assert_close(raw[:,2],normalized[:,2])
    state_loss(raw,torch.tensor([[1.,0.],[0.,1.]]),torch.ones(2),torch.ones(2,dtype=torch.bool)).backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
