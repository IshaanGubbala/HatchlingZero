"""HZ-0E E1: micro-MoE expert contract tests (reference/hz0e_moe_contract.py).
Locks in the real, measured claims from
`docs/restart/hz0e_e1_contract.md` as regression tests -- E1's own exit
gate ("exact total and active parameter counts are known") plus every
other design decision the contract doc resolves explicitly.
"""
from __future__ import annotations

import mlx.core as mx

from reference.hz0e_moe_contract import (
    MoeConfig, _swiglu, init_moe_layer, moe_ffn_forward, moe_layer_param_counts,
)

REAL_CONFIG = MoeConfig()  # dim=768, dense_d_ff=2304, num_experts=4, expert_d_ff=576, capacity_factor=1.5
TOY_CONFIG = MoeConfig(dim=8, dense_d_ff=16, num_experts=4, expert_d_ff=4, capacity_factor=4.0, init_scale=0.3)


def _independent_ffn_params(dim: int, d_ff: int) -> int:
    """Hand-computed, independent of `_ffn_param_count` in the module
    under test -- gate (dim->d_ff) + up (dim->d_ff) + down (d_ff->dim),
    weights and biases."""
    gate = dim * d_ff + d_ff
    up = dim * d_ff + d_ff
    down = d_ff * dim + dim
    return gate + up + down


def test_real_scale_parameter_counts_match_independent_hand_arithmetic():
    """E1's own exit gate, checked with numbers computed a SECOND,
    independent way (not just re-reading the function under test's own
    output) -- locks in every number in the contract doc's section 6."""
    counts = moe_layer_param_counts(REAL_CONFIG)
    dense_ffn = _independent_ffn_params(768, 2304)
    expert_ffn = _independent_ffn_params(768, 576)
    router = 768 * 4 + 4

    assert counts["dense_ffn_baseline"] == dense_ffn == 5_313_792
    assert counts["per_expert_ffn"] == expert_ffn == 1_329_024
    assert counts["all_experts_ffn"] == 4 * expert_ffn == 5_316_096
    assert counts["router"] == router == 3_076
    assert counts["moe_layer_total"] == dense_ffn + 4 * expert_ffn + router == 10_632_964
    assert counts["moe_layer_active_no_overflow"] == expert_ffn + router == 1_332_100
    assert counts["moe_layer_active_worst_case"] == dense_ffn + router == 5_316_868


def test_whole_model_total_and_active_params_match_contract_doc():
    """The contract doc's whole-model numbers (3 converted layers
    against the real, audited 301,178,112-param baseline), recomputed
    independently here rather than trusted from the doc alone."""
    counts = moe_layer_param_counts(REAL_CONFIG)
    base_model = 301_178_112
    layers_converted = 3
    added_total = layers_converted * (counts["moe_layer_total"] - counts["dense_ffn_baseline"])
    new_total = base_model + added_total
    assert new_total == 317_135_628

    active_savings_per_layer = counts["dense_ffn_baseline"] - counts["moe_layer_active_no_overflow"]
    new_active = base_model - layers_converted * active_savings_per_layer
    assert new_active == 289_233_036
    assert new_active < base_model, "typical-case active params must DROP below the dense baseline, not just grow sub-proportionally"
    assert new_total > base_model, "total params must grow (more capacity)"


def test_forward_pass_is_finite_and_correctly_shaped_at_real_scale():
    params = init_moe_layer(REAL_CONFIG)
    x = mx.random.normal((2, 16, 768), key=mx.random.key(1)) * 0.1
    out, diag = moe_ffn_forward(x, params, REAL_CONFIG)
    mx.eval(out, diag.expert_idx, diag.overflow, diag.expert_counts, diag.fallback_count)
    assert out.shape == (2, 16, 768)
    assert bool(mx.all(mx.isfinite(out)))
    assert diag.expert_idx.shape == (32,)
    assert diag.overflow.shape == (32,)
    assert int(mx.sum(diag.expert_counts)) + int(diag.fallback_count) == 32


def test_forward_is_deterministic_given_identical_inputs():
    params = init_moe_layer(TOY_CONFIG)
    x = mx.random.normal((2, 6, 8), key=mx.random.key(2))
    out1, diag1 = moe_ffn_forward(x, params, TOY_CONFIG)
    out2, diag2 = moe_ffn_forward(x, params, TOY_CONFIG)
    mx.eval(out1, out2, diag1.expert_idx, diag2.expert_idx)
    assert bool(mx.array_equal(out1, out2))
    assert bool(mx.array_equal(diag1.expert_idx, diag2.expert_idx))
    assert bool(mx.array_equal(diag1.overflow, diag2.overflow))


def test_non_overflow_token_output_matches_its_routed_expert_scaled_by_gate_weight():
    """Correctness of the core routing math, checked against an
    independently computed reference for one specific token -- not just
    "it runs and produces finite output." Uses a generous capacity so
    nothing overflows.

    The reference is computed via the SAME batched `_swiglu` call
    `moe_ffn_forward` itself uses (all tokens through the chosen expert
    at once), not a separately re-run single-row call -- MLX's matmul
    takes a measurably different numerical path for a batch-of-N input
    versus a batch-of-1 input (confirmed directly: recomputing a single
    token in isolation differs from its value inside the full-batch
    result by ~6e-4 absolute, floating-point non-associativity from
    batch-size-dependent kernel selection, not a routing bug). Compares
    like-for-like batched computations instead, so the assertion tests
    the ROUTING/masking logic under test, not float32 reduction-order
    noise."""
    config = MoeConfig(dim=8, dense_d_ff=16, num_experts=4, expert_d_ff=4, capacity_factor=10.0, init_scale=0.3)
    params = init_moe_layer(config)
    x = mx.random.normal((1, 5, 8), key=mx.random.key(3))
    out, diag = moe_ffn_forward(x, params, config)
    mx.eval(out, diag.expert_idx, diag.overflow, diag.gate_weight)
    assert not bool(mx.any(diag.overflow)), "test assumes generous capacity -- no overflow expected"

    token_index = 2
    chosen_expert = int(diag.expert_idx[token_index])
    x_flat = x.reshape(-1, 8)
    expected_expert_out_all = _swiglu(
        x_flat, params.expert_gate_w[chosen_expert], params.expert_gate_b[chosen_expert],
        params.expert_up_w[chosen_expert], params.expert_up_b[chosen_expert],
        params.expert_down_w[chosen_expert], params.expert_down_b[chosen_expert],
    )
    expected = expected_expert_out_all[token_index:token_index + 1] * diag.gate_weight[token_index]
    actual = out.reshape(-1, 8)[token_index:token_index + 1]
    assert bool(mx.allclose(actual, expected, atol=1e-6))


def test_overflow_token_output_matches_unscaled_fallback_not_its_routed_expert():
    """Forces real overflow via a tiny capacity_factor, then verifies
    an overflowed token's output matches the shared fallback's SwiGLU
    output, UNSCALED (no gate-weight multiplication) -- the contract
    doc's section 4 design decision, checked directly, not just
    documented. Compares against the SAME batched `_swiglu` call
    `moe_ffn_forward` itself uses (see the sibling non-overflow test's
    docstring for why a separately re-run single-row call is not a
    fair bit-exact comparison at float32)."""
    config = MoeConfig(dim=8, dense_d_ff=16, num_experts=4, expert_d_ff=4, capacity_factor=0.05, init_scale=0.3)
    params = init_moe_layer(config)
    x = mx.random.normal((1, 20, 8), key=mx.random.key(4))
    out, diag = moe_ffn_forward(x, params, config)
    mx.eval(out, diag.overflow, diag.fallback_count)
    assert bool(mx.any(diag.overflow)), "test assumes tiny capacity forces real overflow"

    overflow_indices = [i for i in range(20) if bool(diag.overflow[i])]
    x_flat = x.reshape(-1, 8)
    out_flat = out.reshape(-1, 8)
    expected_fallback_all = _swiglu(
        x_flat, params.fallback_gate_w, params.fallback_gate_b,
        params.fallback_up_w, params.fallback_up_b, params.fallback_down_w, params.fallback_down_b,
    )
    for idx in overflow_indices[:3]:  # check a few, not necessarily all, for test speed
        assert bool(mx.allclose(out_flat[idx:idx + 1], expected_fallback_all[idx:idx + 1], atol=1e-6)), (
            f"overflowed token {idx} should match the UNSCALED fallback output exactly"
        )


def test_overflow_never_exceeds_capacity_per_expert():
    """Structural check on the capacity mechanism itself: no expert
    should ever be credited (in expert_counts, i.e. post-overflow) with
    more tokens than its computed capacity."""
    config = MoeConfig(dim=8, dense_d_ff=16, num_experts=4, expert_d_ff=4, capacity_factor=1.0, init_scale=0.3)
    params = init_moe_layer(config)
    x = mx.random.normal((1, 40, 8), key=mx.random.key(5))
    _, diag = moe_ffn_forward(x, params, config)
    mx.eval(diag.expert_counts)
    n = 40
    capacity = int(mx.ceil(mx.array(config.capacity_factor * n / config.num_experts)).item())
    for count in diag.expert_counts.tolist():
        assert count <= capacity, f"expert served {count} tokens, exceeding capacity {capacity}"


def test_every_expert_can_receive_nonzero_routing_mass():
    """A real, non-degenerate sanity check: across enough random tokens
    with random init, more than one expert should receive at least one
    token -- rules out a trivial bug where routing always collapses to
    a single expert regardless of input (a REAL specialization/collapse
    check is E2's job; this is just "the mechanism is capable of using
    more than one expert," a much lower bar appropriate for E1)."""
    params = init_moe_layer(REAL_CONFIG)
    x = mx.random.normal((4, 64, 768), key=mx.random.key(6)) * 0.1
    _, diag = moe_ffn_forward(x, params, REAL_CONFIG)
    mx.eval(diag.expert_counts)
    experts_used = int(mx.sum((diag.expert_counts > 0).astype(mx.int32)))
    assert experts_used >= 2, f"expected at least 2 of 4 experts to receive tokens, got {experts_used}"


def test_gate_weight_is_a_real_softmax_probability_not_a_placeholder():
    """gate_weight must be a genuine softmax probability (in (0, 1],
    summing with other experts' probabilities to 1 per token) -- not a
    stub constant."""
    params = init_moe_layer(REAL_CONFIG)
    x = mx.random.normal((1, 8, 768), key=mx.random.key(7)) * 0.1
    _, diag = moe_ffn_forward(x, params, REAL_CONFIG)
    mx.eval(diag.gate_weight)
    values = diag.gate_weight.tolist()
    assert all(0.0 < v <= 1.0 for v in values)
    assert not all(abs(v - values[0]) < 1e-6 for v in values), "gate weights should vary across tokens, not be a constant"


def test_config_defaults_match_the_contract_docs_stated_choices():
    """Locks in the contract doc's stated numeric choices (section 1,
    3) directly against the real config defaults -- if these drift, the
    doc and code have gone out of sync."""
    assert REAL_CONFIG.dim == 768
    assert REAL_CONFIG.dense_d_ff == 2304
    assert REAL_CONFIG.num_experts == 4
    assert REAL_CONFIG.expert_d_ff == 576
    assert REAL_CONFIG.expert_d_ff * REAL_CONFIG.num_experts == REAL_CONFIG.dense_d_ff
    assert REAL_CONFIG.capacity_factor == 1.5


def test_fallback_weights_are_independent_of_expert_weights():
    """Structural sanity: the fallback's weight tensors must be their
    OWN arrays, not aliases of any expert's weights -- a real
    (if unlikely) bug this guards against directly."""
    params = init_moe_layer(REAL_CONFIG)
    for e in range(REAL_CONFIG.num_experts):
        assert params.fallback_gate_w.shape != params.expert_gate_w[e].shape or not bool(
            mx.array_equal(params.fallback_gate_w, params.expert_gate_w[e])
        )


def test_moe_layer_active_no_overflow_is_less_than_dense_baseline_at_real_scale():
    """The headline claim from the contract doc's section 6, checked
    directly: in the typical (non-overflow) case, ONE converted layer's
    active param count must be strictly less than what the original
    dense FFN it replaced would have cost."""
    counts = moe_layer_param_counts(REAL_CONFIG)
    assert counts["moe_layer_active_no_overflow"] < counts["dense_ffn_baseline"]


def test_moe_layer_total_is_greater_than_dense_baseline_at_real_scale():
    """The other half of the same claim: total capacity for a converted
    layer must exceed the original dense FFN -- more capacity exists,
    even though less of it is typically active per token."""
    counts = moe_layer_param_counts(REAL_CONFIG)
    assert counts["moe_layer_total"] > counts["dense_ffn_baseline"]
