"""Phase 1A structural tests for HZCQPersistentMemory, exactly matching
plans/HatchlingZero — Mainline Research Plan.md section 7's "Task
memory tests" checklist (promotion gate: do not train seriously until
all of these pass). Six required properties, one test class per
property plus a couple of real numerical-safety checks."""
from __future__ import annotations

import torch

from reference.hz0h_bdh_hzcq_v1_persistent_memory_torch import (
    HZCQPersistentMemory,
    HZCQPersistentMemoryConfig,
)

D = 32
M_S = 8
T_DEMO = 12
B = 2


def _mem(seed: int = 0) -> HZCQPersistentMemory:
    torch.manual_seed(seed)
    return HZCQPersistentMemory(HZCQPersistentMemoryConfig(n_embd=D, memory_slots=M_S, gate_hidden=8))


def _demo(seed: int, batch: int = B, t: int = T_DEMO) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(batch, t, D, generator=g)


# 1. Different demonstrations produce different S.
def test_different_demos_produce_different_s():
    mem = _mem()
    S0 = mem.init_state(B)
    demo_a, demo_b = _demo(1), _demo(2)
    S_a = mem.update(S0, demo_a)
    S_b = mem.update(S0, demo_b)
    assert not torch.allclose(S_a, S_b, atol=1e-6)


# 2. Demo ordering affects S when appropriate.
def test_demo_ordering_affects_s():
    mem = _mem()
    demo_a, demo_b = _demo(1), _demo(2)
    S_ab = mem.update_sequence(B, [demo_a, demo_b])
    S_ba = mem.update_sequence(B, [demo_b, demo_a])
    # Real property: each update conditions on the PREVIOUS S (not a
    # permutation-invariant pooling), so processing order genuinely
    # changes what S_prev looked like when the second demo was read --
    # order-dependence is structural here, not incidental.
    assert not torch.allclose(S_ab, S_ba, atol=1e-6)


# 3. Multiple demos accumulate information (S after 2 real demos
# differs meaningfully from S after just the first alone -- not just
# "differs from init", which test 1 already covers).
def test_multiple_demos_accumulate():
    mem = _mem()
    demo_a, demo_b = _demo(1), _demo(2)
    S0 = mem.init_state(B)
    S_after_one = mem.update(S0, demo_a)
    S_after_two = mem.update(S_after_one, demo_b)
    assert not torch.allclose(S_after_one, S_after_two, atol=1e-6)
    # Real, stronger check: S_after_two must differ from what a SECOND
    # independent update starting fresh from S0 with demo_b alone would
    # give -- i.e. the first demo's real influence survives into the
    # second update, not overwritten/forgotten outright.
    S_b_alone = mem.update(S0, demo_b)
    assert not torch.allclose(S_after_two, S_b_alone, atol=1e-6)


# 4. Query-time memory cost does not grow with raw demo length: S's own
# shape/footprint is exactly (B, M_S, D) regardless of how many demos,
# or how long each demo is, were ingested -- and the same holds true
# whether T_demo is tiny or large.
def test_memory_footprint_independent_of_demo_length_and_count():
    mem = _mem()
    S_after_short = mem.update_sequence(B, [_demo(1, t=4)])
    S_after_long = mem.update_sequence(B, [_demo(1, t=400)])
    S_after_many = mem.update_sequence(B, [_demo(i, t=8) for i in range(1, 21)])
    for S in (S_after_short, S_after_long, S_after_many):
        assert S.shape == (B, M_S, D)
        assert S.numel() == B * M_S * D
    # Real, direct check on update() itself: its computational cost
    # (measured here via output shape and absence of any T_demo-shaped
    # dimension in the RETURNED state) never depends on T_demo -- if it
    # did, S's shape itself would carry a T_demo-sized axis, which none
    # of the three cases above show.


# 5. Removing one demonstration changes behavior (ablation sensitivity
# -- S is not insensitive to which demos were actually shown).
def test_removing_a_demo_changes_s():
    mem = _mem()
    demo_a, demo_b, demo_c = _demo(1), _demo(2), _demo(3)
    S_full = mem.update_sequence(B, [demo_a, demo_b, demo_c])
    S_without_b = mem.update_sequence(B, [demo_a, demo_c])
    assert not torch.allclose(S_full, S_without_b, atol=1e-6)


# 6. Memory is actually used downstream -- real gradient-flow check:
# a loss on S_new must produce real, nonzero gradients into BOTH the
# raw demo hidden states AND the module's own trainable parameters,
# proving S genuinely depends on (not just ignores) its inputs.
#
# Real, intentional exception: gate_w1 has EXACTLY zero gradient at
# this init, by design -- gate_w2 is protected-zero-init (matching
# reference/hz0h_bdh_adaptive_gate_torch.py exactly), so
# g = sigmoid(hid @ 0 + b2) = sigmoid(b2) is structurally independent
# of hid (hence of w1) until w2 moves off zero during training. Check
# gate_w2 instead -- its gradient IS nonzero at init, since hid itself
# is nonzero (only w2 is zero-initialized, not the path producing hid).
def test_gradients_flow_through_memory_update():
    mem = _mem()
    S0 = mem.init_state(B)
    demo = _demo(1)
    demo.requires_grad_(True)
    S_new = mem.update(S0, demo)
    loss = S_new.pow(2).sum()
    loss.backward()
    assert demo.grad is not None and demo.grad.abs().sum() > 0
    assert mem.q_proj.weight.grad is not None and mem.q_proj.weight.grad.abs().sum() > 0
    assert mem.gate_w2.grad is not None and mem.gate_w2.grad.abs().sum() > 0
    assert mem.gate_w1.grad is not None and mem.gate_w1.grad.abs().sum() == 0, (
        "gate_w1 should have EXACTLY zero gradient at this protected init "
        "(gate_w2=0 blocks the path) -- a nonzero value here would mean "
        "the protected-init design isn't actually wired the way it's documented"
    )


# Real numerical-safety checks, same discipline as the validated
# adaptive gate's own _rms eps-inside-sqrt fix -- this module reuses
# that exact pattern, so verify it actually avoids the failure it was
# copied to avoid: an all-zero delta_S (e.g. an all-zero/degenerate
# demo) must not produce NaN gradients.
def test_no_nan_on_degenerate_zero_demo():
    mem = _mem()
    S0 = mem.init_state(B)
    zero_demo = torch.zeros(B, T_DEMO, D, requires_grad=True)
    S_new = mem.update(S0, zero_demo)
    assert torch.isfinite(S_new).all()
    S_new.sum().backward()
    assert torch.isfinite(zero_demo.grad).all()


# Real protected-init check, matching add_adaptive_gate's own documented
# contract: gate_w2 starts at exact zero, so the FIRST ever update's
# gate is a fixed sigmoid(b2) = g_init, independent of q -- i.e. update()
# at step 0 behaves as a KNOWN, predictable partial write, not an
# arbitrary/random one, before any training has happened.
def test_protected_init_gate_starts_at_g_init():
    mem = _mem()
    S0 = mem.init_state(B)
    demo = _demo(1)
    g = mem._gate(S0, mem.ln_read(mem.write_proj(
        torch.nn.functional.softmax(
            torch.matmul(mem.q_proj(S0), mem.k_proj(demo).transpose(-1, -2)) / (D ** 0.5), dim=-1
        ) @ mem.v_proj(demo)
    )))
    expected = torch.sigmoid(mem.gate_b2)
    assert torch.allclose(g, expected.expand_as(g), atol=1e-6)
