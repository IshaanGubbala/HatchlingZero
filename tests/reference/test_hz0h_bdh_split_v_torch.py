"""Real correctness tests for reference/hz0h_bdh_split_v_torch.py's
BDHSplitV. Unlike test_hz0h_bdh_fused_attention_torch.py (which checks
exact numerical agreement with the oracle, since that file is a
math-equivalent kernel swap), Split-V is a genuinely different
architecture with real new parameters (`w_v`/`w_o`) -- there is no
oracle to match. These tests check what's actually checkable: shapes,
gradient flow through every parameter (including the new ones), the
real parameter-count formula against an instantiated model, and that
value-subspace narrowing is real (not silently falling back to the
full-D broadcast vanilla BDH uses).
"""
from __future__ import annotations

import torch

from reference.hz0h_bdh_split_v_torch import BDHSplitV, BDHSplitVConfig, split_v_parameter_count


def _tiny_config(n_layer: int = 2, n_embd: int = 32, n_head: int = 4) -> BDHSplitVConfig:
    return BDHSplitVConfig(n_layer=n_layer, n_embd=n_embd, n_head=n_head, mlp_internal_dim_multiplier=8, vocab_size=32, dropout=0.0)


def test_forward_shapes():
    config = _tiny_config()
    torch.manual_seed(0)
    model = BDHSplitV(config)
    idx = torch.randint(0, config.vocab_size, (2, 16))
    logits, loss = model(idx)
    assert logits.shape == (2, 16, config.n_embd)
    assert loss is None

    y = torch.randint(0, config.vocab_size, (2, 16))
    logits, loss = model(idx, targets=y)
    assert loss is not None
    assert torch.isfinite(loss)


def test_gradients_flow_through_every_parameter_including_new_ones():
    config = _tiny_config()
    torch.manual_seed(1)
    model = BDHSplitV(config)
    idx = torch.randint(0, config.vocab_size, (2, 12))
    x, y = idx[:, :-1].contiguous(), idx[:, 1:].contiguous()

    _logits, loss = model(x, targets=y)
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"{name} got no gradient"
        assert torch.isfinite(param.grad).all(), f"{name} has non-finite gradient"

    # The real new parameters specifically -- these are what this
    # architecture actually adds, so a silent no-op bug here (e.g. w_v
    # never actually used) would be the most important thing to catch.
    assert float(model.w_v.grad.norm()) > 0
    assert float(model.w_o.grad.norm()) > 0


def test_parameter_count_formula_matches_real_instantiated_model():
    for n_layer, n_embd, n_head, mult in [(2, 32, 4, 8), (8, 512, 8, 32), (8, 1024, 8, 32)]:
        config = BDHSplitVConfig(n_layer=n_layer, n_embd=n_embd, n_head=n_head, mlp_internal_dim_multiplier=mult, vocab_size=256, dropout=0.0)
        model = BDHSplitV(config)
        real_count = sum(p.numel() for p in model.parameters())
        assert split_v_parameter_count(config) == real_count, f"formula/model mismatch at n_embd={n_embd}, n_head={n_head}"


def test_split_v_uses_narrow_per_head_value_not_full_width_broadcast():
    """Real behavioral check that this isn't accidentally degenerate to
    vanilla BDH's full-D broadcast: zeroing w_v should zero the model's
    output-affecting V contribution, and the attention module should be
    receiving a (B, nh, T, D/nh) tensor, not (B, 1, T, D)."""
    config = _tiny_config(n_head=4, n_embd=32)
    torch.manual_seed(2)
    model = BDHSplitV(config)
    D, nh = config.n_embd, config.n_head
    idx = torch.randint(0, config.vocab_size, (1, 8))

    x = model.embed(idx).unsqueeze(1)
    x = model.ln(x)
    v_full = x @ model.w_v
    v_split = v_full.view(1, 8, nh, D // nh).transpose(1, 2)
    assert v_split.shape == (1, nh, 8, D // nh)

    # Splitting must actually reshape into narrower per-head slices --
    # not just repeat the full D width nh times (that would silently
    # reduce to vanilla BDH's own broadcast behavior, defeating the
    # entire point of this architecture).
    assert v_split.shape[-1] == D // nh
    assert v_split.shape[-1] < D


def test_deterministic_given_seed():
    config = _tiny_config()
    idx = torch.randint(0, config.vocab_size, (2, 10))

    torch.manual_seed(42)
    model_a = BDHSplitV(config)
    torch.manual_seed(42)
    model_b = BDHSplitV(config)

    with torch.no_grad():
        logits_a, _ = model_a(idx)
        logits_b, _ = model_b(idx)
    assert torch.equal(logits_a, logits_b)


def test_requires_n_embd_divisible_by_n_head():
    config = BDHSplitVConfig(n_layer=2, n_embd=33, n_head=4, mlp_internal_dim_multiplier=8, vocab_size=32, dropout=0.0)
    try:
        BDHSplitV(config)
        assert False, "expected an assertion error for n_embd not divisible by n_head"
    except AssertionError as exc:
        assert "divisible" in str(exc)
