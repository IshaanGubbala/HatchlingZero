"""Phase 1A structural tests for HZCQReasoningWorkspace, exactly
matching plans/HatchlingZero — Mainline Research Plan.md section 7's
"Workspace tests" checklist (promotion gate: do not train seriously
until all of these pass)."""
from __future__ import annotations

import pytest
import torch

from reference.hz0h_bdh_hzcq_v1_persistent_memory_torch import (
    HZCQPersistentMemory,
    HZCQPersistentMemoryConfig,
)
from reference.hz0h_bdh_hzcq_v1_reasoning_workspace_torch import (
    HZCQReasoningWorkspace,
    HZCQReasoningWorkspaceConfig,
)

D = 32
M_S = 8
M_H = 8
T_QUERY = 10
B = 2


def _workspace(seed: int = 0) -> HZCQReasoningWorkspace:
    torch.manual_seed(seed)
    return HZCQReasoningWorkspace(HZCQReasoningWorkspaceConfig(n_embd=D, workspace_slots=M_H, gate_hidden=8))


def _memory(seed: int = 1) -> HZCQPersistentMemory:
    torch.manual_seed(seed)
    return HZCQPersistentMemory(HZCQPersistentMemoryConfig(n_embd=D, memory_slots=M_S, gate_hidden=8))


def _fake_s(batch: int = B) -> torch.Tensor:
    g = torch.Generator().manual_seed(2)
    return torch.randn(batch, M_S, D, generator=g)


def _fake_query(batch: int = B, t: int = T_QUERY) -> torch.Tensor:
    g = torch.Generator().manual_seed(3)
    return torch.randn(batch, t, D, generator=g)


# 1. H has fixed shape for every R.
def test_h_shape_fixed_for_every_r():
    ws = _workspace()
    S, x = _fake_s(), _fake_query()
    for r in (1, 2, 4, 8, 16, 32):
        H = ws.run(B, S, x, n_rounds=r)
        assert H.shape == (B, M_H, D)


# 2. R=2 and R=16 differ only by recurrent computation count -- i.e.
# they produce DIFFERENT content (real computation happened) but the
# SAME shape (no structural difference beyond depth).
def test_r2_and_r16_differ_only_in_content_not_shape():
    ws = _workspace()
    S, x = _fake_s(), _fake_query()
    H_r2 = ws.run(B, S, x, n_rounds=2)
    H_r16 = ws.run(B, S, x, n_rounds=16)
    assert H_r2.shape == H_r16.shape == (B, M_H, D)
    assert not torch.allclose(H_r2, H_r16, atol=1e-6)


# 3 & 4. Sequence length is identical across R, and increasing R does
# not allocate additional token positions -- checked directly against
# the QUERY tensor x, which is what would grow under HZ-CQ-v0's real
# failure mode (each "reasoning step" appending a new sequence
# position). Here x is passed in unchanged and untouched by `run` at
# every R; there is no code path in HZCQReasoningWorkspace that could
# append to x, S, or H even in principle -- verified by asserting the
# exact tensor identity (not just shape) of x survives untouched.
def test_query_sequence_never_grows_with_r():
    ws = _workspace()
    S, x = _fake_s(), _fake_query()
    x_before = x.clone()
    for r in (1, 4, 16, 32):
        ws.run(B, S, x, n_rounds=r)
        assert x.shape[1] == T_QUERY
        assert torch.equal(x, x_before)  # run() must never mutate its inputs


# 5. Gradients flow through the recurrent computation -- across
# MULTIPLE rounds, not just the last one: perturbing round-1's inputs
# must still influence the R-round-later output (real recurrent
# dependency, not e.g. an accidental detach mid-loop).
def test_gradients_flow_through_multiple_rounds():
    ws = _workspace()
    S = _fake_s()
    x = _fake_query()
    x.requires_grad_(True)
    H_final = ws.run(B, S, x, n_rounds=8)
    loss = H_final.pow(2).sum()
    loss.backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
    assert ws.read_x.q_proj.weight.grad is not None and ws.read_x.q_proj.weight.grad.abs().sum() > 0
    assert ws.gate_w2.grad is not None and ws.gate_w2.grad.abs().sum() > 0


# 6. Adaptive gates remain numerically stable through R=16+ -- no NaN/
# Inf in H's content OR in gradients, at real depth well past the
# plan's R=16 target (tested to 32 for margin).
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


# PAPER-2 structural check: the identity_biased (LayerScale) update
# path is a real, deliberate alternative to the default LN-residual
# update -- must satisfy the exact same structural contract (fixed
# shape across R, gradients including into the new `alpha` param,
# numerically stable through deep R) without being byte-identical to
# the default path's output.
def test_identity_biased_layerscale_path_is_structurally_sound():
    torch.manual_seed(0)
    ws = HZCQReasoningWorkspace(HZCQReasoningWorkspaceConfig(
        n_embd=D, workspace_slots=M_H, gate_hidden=8, identity_biased=True, layerscale_init=0.1))
    assert ws.alpha is not None and ws.alpha.requires_grad
    S = _fake_s()
    x = _fake_query()
    x.requires_grad_(True)
    for r in (1, 2, 4, 8, 16, 32):
        H = ws.run(B, S, x, n_rounds=r)
        assert H.shape == (B, M_H, D)
        assert torch.isfinite(H).all()
    H2 = ws.run(B, S, x, n_rounds=2)
    H16 = ws.run(B, S, x, n_rounds=16)
    assert not torch.allclose(H2, H16, atol=1e-6)
    H16.sum().backward()
    assert torch.isfinite(x.grad).all()
    assert ws.alpha.grad is not None and torch.isfinite(ws.alpha.grad).all()
    for p in ws.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()


# PAPER-3 structural check: the bounded_residual (re-anchored, tanh-
# capped) update path must satisfy the same structural contract as
# every other path, PLUS its own real invariant -- run()'s cached-
# H_base path must be bit-identical to a manual step()-loop (which
# recomputes H_base fresh every call), and it must add ZERO new
# parameters (the whole point of reusing read_s for the anchor read
# instead of adding new weights).
def test_bounded_residual_path_is_structurally_sound():
    torch.manual_seed(0)
    ws = HZCQReasoningWorkspace(HZCQReasoningWorkspaceConfig(
        n_embd=D, workspace_slots=M_H, gate_hidden=8, bounded_residual=True, bound_scale=1.0))
    ws_default = HZCQReasoningWorkspace(HZCQReasoningWorkspaceConfig(n_embd=D, workspace_slots=M_H, gate_hidden=8))
    assert sum(p.numel() for p in ws.parameters()) == sum(p.numel() for p in ws_default.parameters())

    S = _fake_s()
    x = _fake_query()
    x.requires_grad_(True)
    for r in (1, 2, 4, 8, 16, 32):
        H = ws.run(B, S, x, n_rounds=r)
        assert H.shape == (B, M_H, D)
        assert torch.isfinite(H).all()
    H2 = ws.run(B, S, x, n_rounds=2)
    H16 = ws.run(B, S, x, n_rounds=16)
    assert not torch.allclose(H2, H16, atol=1e-6)

    H_manual = ws.init_state(B)
    for _ in range(12):
        H_manual = ws.step(H_manual, S, x.detach())
    H_run = ws.run(B, S, x.detach(), n_rounds=12)
    assert torch.equal(H_manual, H_run)

    H16.sum().backward()
    assert torch.isfinite(x.grad).all()
    for p in ws.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()


def test_bounded_residual_and_identity_biased_are_mutually_exclusive():
    with pytest.raises(ValueError):
        HZCQReasoningWorkspaceConfig(n_embd=D, workspace_slots=M_H, identity_biased=True, bounded_residual=True)


# Real, additional end-to-end check: H genuinely depends on S (not
# just x) -- swapping S for a different persistent memory must change
# H's output at a fixed R, otherwise the "S answers what the rule is"
# design intent (section 6.1's whole point) isn't actually wired.
def test_h_depends_on_s_not_just_x():
    ws = _workspace()
    x = _fake_query()
    S_a = _fake_s()
    g = torch.Generator().manual_seed(99)
    S_b = torch.randn(B, M_S, D, generator=g)
    H_a = ws.run(B, S_a, x, n_rounds=8)
    H_b = ws.run(B, S_b, x, n_rounds=8)
    assert not torch.allclose(H_a, H_b, atol=1e-6)


# Real integration smoke test: S built by HZCQPersistentMemory (not a
# random fake tensor) feeds cleanly into HZCQReasoningWorkspace end to
# end, gradients flow all the way back into the ORIGINAL demo hidden
# states through both modules chained together -- the real, intended
# S -> H pipeline, not each module tested only in isolation.
def test_real_s_to_h_pipeline_end_to_end():
    torch.manual_seed(0)
    mem = _memory()
    ws = _workspace(seed=0)
    demo = _fake_query(t=6)
    demo.requires_grad_(True)
    S = mem.update_sequence(B, [demo])
    x = _fake_query()
    H = ws.run(B, S, x, n_rounds=4)
    assert H.shape == (B, M_H, D)
    H.sum().backward()
    assert demo.grad is not None and demo.grad.abs().sum() > 0
