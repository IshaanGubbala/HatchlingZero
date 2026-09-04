"""PAPER-4 structural tests for HZCQReasoningWorkspaceFastSlow, same
discipline as the single-state workspace's Phase 1A checklist: promotion
gate, do not train seriously until these pass."""
from __future__ import annotations

import torch

from reference.hz0h_bdh_hzcq_v1_paper4_fast_slow_torch import (
    HZCQFastSlowConfig,
    HZCQReasoningWorkspaceFastSlow,
)

D = 32
M_S = 8
M_H = 8
T_QUERY = 10
B = 2


def _workspace(seed: int = 0) -> HZCQReasoningWorkspaceFastSlow:
    torch.manual_seed(seed)
    return HZCQReasoningWorkspaceFastSlow(HZCQFastSlowConfig(n_embd=D, workspace_slots=M_H, gate_hidden=8))


def _fake_s(batch: int = B) -> torch.Tensor:
    g = torch.Generator().manual_seed(2)
    return torch.randn(batch, M_S, D, generator=g)


def _fake_query(batch: int = B, t: int = T_QUERY) -> torch.Tensor:
    g = torch.Generator().manual_seed(3)
    return torch.randn(batch, t, D, generator=g)


def test_h_slow_shape_fixed_for_every_r():
    ws = _workspace()
    S, x = _fake_s(), _fake_query()
    for r in (1, 2, 4, 8, 16, 32):
        H = ws.run(B, S, x, n_rounds=r)
        assert H.shape == (B, M_H, D)
        assert torch.isfinite(H).all()


def test_r2_and_r16_differ_only_in_content_not_shape():
    ws = _workspace()
    S, x = _fake_s(), _fake_query()
    H_r2 = ws.run(B, S, x, n_rounds=2)
    H_r16 = ws.run(B, S, x, n_rounds=16)
    assert H_r2.shape == H_r16.shape == (B, M_H, D)
    assert not torch.allclose(H_r2, H_r16, atol=1e-6)


def test_query_sequence_never_grows_with_r():
    ws = _workspace()
    S, x = _fake_s(), _fake_query()
    x_before = x.clone()
    for r in (1, 4, 16, 32):
        ws.run(B, S, x, n_rounds=r)
        assert x.shape[1] == T_QUERY
        assert torch.equal(x, x_before)


def test_gradients_flow_to_every_parameter():
    ws = _workspace()
    S = _fake_s()
    x = _fake_query()
    x.requires_grad_(True)
    H_final = ws.run(B, S, x, n_rounds=8)
    loss = H_final.pow(2).sum()
    loss.backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
    for name, p in ws.named_parameters():
        assert p.grad is not None, f"{name} got no gradient"
        assert torch.isfinite(p.grad).all(), f"{name} has non-finite gradient"


def test_numerically_stable_through_deep_recurrence():
    ws = _workspace()
    S = _fake_s()
    x = _fake_query()
    x.requires_grad_(True)
    H_final = ws.run(B, S, x, n_rounds=32)
    assert torch.isfinite(H_final).all()
    H_final.sum().backward()
    assert torch.isfinite(x.grad).all()
    for p in ws.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()


def test_h_slow_depends_on_s_not_just_x():
    ws = _workspace()
    x = _fake_query()
    S_a = _fake_s()
    g = torch.Generator().manual_seed(99)
    S_b = torch.randn(B, M_S, D, generator=g)
    H_a = ws.run(B, S_a, x, n_rounds=8)
    H_b = ws.run(B, S_b, x, n_rounds=8)
    assert not torch.allclose(H_a, H_b, atol=1e-6)


def test_h_fast_and_h_slow_are_independent_states_with_different_content():
    """Real, PAPER-4-specific check: H_fast and H_slow must actually be
    different tensors carrying different content, not accidentally
    aliased or degenerate copies of each other."""
    ws = _workspace()
    S, x = _fake_s(), _fake_query()
    H_fast, H_slow = ws.init_state(B)
    for _ in range(8):
        H_fast, H_slow = ws.step(H_fast, H_slow, S, x)
    assert not torch.allclose(H_fast, H_slow, atol=1e-6)
    assert H_fast.shape == H_slow.shape == (B, M_H, D)


def test_config_rejects_disallowed_workspace_slots():
    import pytest
    with pytest.raises(ValueError):
        HZCQFastSlowConfig(n_embd=D, workspace_slots=24, allow_ablation_slots=True)
