import torch
import pytest
from torch import nn

from scripts import recover_g1_tctst_v4_run as recovery


def test_cuda_oom_cleanup_continues_after_deferred_oom(monkeypatch):
    def deferred_oom(*_args, **_kwargs):
        raise RuntimeError('CUDA error: out of memory')

    monkeypatch.setattr(torch.cuda, 'synchronize', deferred_oom)
    monkeypatch.setattr(torch.cuda, 'empty_cache', deferred_oom)
    cleanup = getattr(recovery, 'clear_cuda_oom_state', None)
    assert callable(cleanup), 'recovery must provide CUDA OOM cleanup'
    errors = cleanup()
    assert len(errors) == 2
    assert all('out of memory' in error.lower() for error in errors)


def test_cuda_oom_cleanup_does_not_swallow_unrelated_errors(monkeypatch):
    def illegal_access(*_args, **_kwargs):
        raise RuntimeError('CUDA error: illegal memory access')

    monkeypatch.setattr(torch.cuda, 'synchronize', illegal_access)
    cleanup = getattr(recovery, 'clear_cuda_oom_state', None)
    assert callable(cleanup), 'recovery must provide CUDA OOM cleanup'
    with pytest.raises(RuntimeError, match='illegal memory access'):
        cleanup()


def test_partial_checkpoint_restores_model_optimizer_and_rng(tmp_path, monkeypatch):
    monkeypatch.setattr(recovery.base, 'DEVICE', torch.device('cpu'))
    torch.manual_seed(71)
    model = nn.Linear(3, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=6e-4)
    generator = torch.Generator().manual_seed(101)

    x = torch.randn(8, 3)
    loss = model(x).square().mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    checkpoint_path = tmp_path / 'partial.pt'
    recovery.save_partial(model, optimizer, generator, epoch=10, path=checkpoint_path)
    expected_model = {key: value.detach().clone() for key, value in model.state_dict().items()}
    expected_optimizer = {
        parameter: {
            key: value.detach().clone() if torch.is_tensor(value) else value
            for key, value in state.items()
        }
        for parameter, state in optimizer.state.items()
    }
    expected_torch_rng = torch.get_rng_state().clone()
    expected_generator_rng = generator.get_state().clone()
    expected_torch_draw = torch.rand(5)
    expected_order = torch.randperm(20, generator=generator)

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(3.0)
    extra_loss = model(x).square().mean()
    extra_loss.backward()
    optimizer.step()
    _ = torch.rand(7)
    _ = torch.randperm(20, generator=generator)

    assert recovery.restore_partial(model, optimizer, generator, checkpoint_path) == 10
    for key, expected in expected_model.items():
        torch.testing.assert_close(model.state_dict()[key], expected, rtol=0, atol=0)
    assert optimizer.param_groups[0]['lr'] == 6e-4
    assert set(optimizer.state) == set(expected_optimizer)
    for parameter, expected_state in expected_optimizer.items():
        for key, expected in expected_state.items():
            observed = optimizer.state[parameter][key]
            if torch.is_tensor(expected):
                torch.testing.assert_close(observed, expected, rtol=0, atol=0)
            else:
                assert observed == expected
    assert torch.equal(torch.get_rng_state(), expected_torch_rng)
    assert torch.equal(generator.get_state(), expected_generator_rng)
    torch.testing.assert_close(torch.rand(5), expected_torch_draw, rtol=0, atol=0)
    assert torch.equal(torch.randperm(20, generator=generator), expected_order)
