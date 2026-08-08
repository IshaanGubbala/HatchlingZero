"""HZ-0G G5: correctness tests for reference/hz0g_g5_full_integration.py,
the new A+B+C+D+E composed forward pass.

Checked against the ACTUAL frozen checkpoint, matching this project's
established convention (D's own D7 tests use the same pattern) -- MoE's
target layers (27, 28, 30) only exist in the real 31-layer model, a
synthetic tiny model can't exercise this. Skips if the checkpoint isn't
present locally.
"""
from __future__ import annotations

import mlx.core as mx
import pytest

from reference.hz0b_b8_latent_write import init_latent_write_controller
from reference.hz0d_d6_integration import ATTENTION_INDICES, conditional_hidden_with_fast_weights, d6_fast_weight_config
from reference.hz0d_fast_weights import FastWeightConfig, init_fast_weights
from reference.hz0e_e6_integration import TARGET_LAYERS, forward_e6, init_e6_layers
from reference.hz0g_g5_full_integration import full_integration_forward, full_integration_hidden
from scripts.hz0b_b11_baseline_comparison import CHECKPOINT, load_frozen_model
from scripts.hz0c_c6_conditional_attention_eval import fixed_matched_trigger

pytestmark = pytest.mark.skipif(
    not (CHECKPOINT / "state.json").exists(),
    reason="frozen HZ-0A checkpoint not present locally (gitignored under outputs/)",
)


def _setup():
    model, _payload = load_frozen_model()
    config = d6_fast_weight_config()
    fast_state = init_fast_weights(config)
    latent_params = init_latent_write_controller(d_model=768, key_dim=64, value_dim=64, seed=0)
    tokens = mx.array([[1, 45, 982, 12, 7, 300, 44, 1023, 55, 66, 77, 88]])
    trigger = fixed_matched_trigger(1, tokens.shape[1], 0.15)
    return model, config, fast_state, latent_params, tokens, trigger


def test_moe_target_layers_disjoint_from_attention_indices():
    assert not (set(ATTENTION_INDICES) & set(TARGET_LAYERS))


def test_moe_disabled_is_bit_identical_to_existing_a_c_d_path():
    """HZ-Dense's own backbone (A+C+D, no E): must exactly reproduce
    conditional_hidden_with_fast_weights -- the already-tested path
    every G2-G4 result was measured against. Not "close", bit-exact."""
    model, config, fast_state, _latent_params, tokens, trigger = _setup()
    existing = conditional_hidden_with_fast_weights(model, tokens, trigger, fast_state, config)
    new_disabled = full_integration_hidden(model, tokens, trigger, fast_state, config, moe_layers=None, moe_enabled=False)
    new_enabled_but_no_layers = full_integration_hidden(model, tokens, trigger, fast_state, config, moe_layers=None, moe_enabled=True)
    mx.eval(existing, new_disabled, new_enabled_but_no_layers)
    assert bool(mx.array_equal(existing, new_disabled))
    assert bool(mx.array_equal(existing, new_enabled_but_no_layers))  # moe_layers=None disables E regardless of the flag


def test_moe_enabled_matches_forward_e6_at_the_isolated_moe_layers():
    """full_integration_hidden's MoE branch must be bit-identical to
    forward_e6's own MoE computation -- proves the ported MoE
    substitution logic wasn't transcribed wrong, not just that it runs
    without error. Uses a small, freshly (not checkpoint-)initialized
    model -- this test checks the MoE plumbing's correctness, not real
    trained quality.

    full_integration_hidden branches on the module-level ATTENTION_INDICES
    constant (4, 9, 14, 19, 24, 29), not on whatever `model` was actually
    built with -- a real, load-bearing assumption (it always operates on
    the real 301M checkpoint's fixed architecture in practice), which
    forward_e6 does NOT share (it only ever checks `index in
    target_layers`). To isolate the MoE branch cleanly without that
    mismatch confounding the comparison, this toy model has FEWER layers
    (4) than the real ATTENTION_INDICES' smallest value (4, i.e. indices
    0-3 only) -- so full_integration_hidden's attention branch is
    structurally unreachable here, and both functions reduce to the same
    two branches (MoE vs. plain block(x, state)) for every layer."""
    from reference.hz0a_mlx_model import HZ0AMlxModel
    from reference.hz0d_d6_integration import logits_from_hidden

    dim, layers, heads, d_ff, vocab = 32, 4, 2, 64, 64
    toy_model = HZ0AMlxModel(vocab, dim, layers, heads, d_ff, attention_indices=(), native_metal=False, mixer="gdn2_fix")
    mx.eval(toy_model.parameters())

    small_config = FastWeightConfig(dim=dim, rank=4, num_layers=0, decay_rate=1.0, max_delta_norm=1.0)
    small_fast_state = init_fast_weights(small_config)
    small_target_layers = (1, 2)  # both < 4, so full_integration_hidden's attention branch never triggers
    moe_layers = init_e6_layers(toy_model, seed=1, target_layers=small_target_layers, warm_start_experts=False)
    tokens = mx.array([[1, 5, 20, 3]])
    never_trigger = mx.zeros((1, tokens.shape[1]))

    e6_result = forward_e6(toy_model, tokens, moe_layers=moe_layers, enabled=True, target_layers=small_target_layers)
    full_hidden = full_integration_hidden(
        toy_model, tokens, never_trigger, small_fast_state, small_config,
        moe_layers=moe_layers, moe_enabled=True, moe_target_layers=small_target_layers,
    )
    full_logits = logits_from_hidden(toy_model, full_hidden)
    mx.eval(e6_result.logits, full_logits)
    assert bool(mx.allclose(e6_result.logits, full_logits, atol=1e-5, rtol=1e-5))


def test_overlapping_moe_and_attention_layers_raises():
    model, config, fast_state, _latent_params, tokens, trigger = _setup()
    bad_moe_layers = {4: object()}  # 4 is in ATTENTION_INDICES -- must be rejected, not silently mishandled
    with pytest.raises(ValueError, match="overlap"):
        full_integration_hidden(model, tokens, trigger, fast_state, config, moe_layers=bad_moe_layers, moe_enabled=True, moe_target_layers=(4,))


def test_full_integration_forward_finite_and_correct_shapes():
    model, config, fast_state, latent_params, tokens, trigger = _setup()
    moe_layers = init_e6_layers(model, seed=1, target_layers=TARGET_LAYERS)
    result = full_integration_forward(model, tokens, trigger, latent_params, fast_state, config, moe_layers=moe_layers, moe_enabled=True)
    mx.eval(result.logits, result.write_gates)
    assert result.logits.shape == (1, tokens.shape[1], 24576)
    assert result.write_gates.shape == (1, tokens.shape[1])
    assert bool(mx.all(mx.isfinite(result.logits)))
    assert bool(mx.all((result.write_gates >= 0) & (result.write_gates <= 1)))


def test_full_integration_forward_moe_disabled_matches_dense_hz_dense_path():
    """HZ-Dense (moe_enabled=False) must match B+full_integration_hidden's
    own A+C+D output composed with B's read/write -- confirms the E flag
    genuinely gates the whole pipeline, not just the hidden-state stage."""
    from reference.hz0b_b8_latent_write import sequential_latent_write_and_read
    from reference.hz0d_d6_integration import logits_from_hidden

    model, config, fast_state, latent_params, tokens, trigger = _setup()
    result = full_integration_forward(model, tokens, trigger, latent_params, fast_state, config, moe_layers=None, moe_enabled=False)
    hidden = conditional_hidden_with_fast_weights(model, tokens, trigger, fast_state, config)
    hidden2, _memory_state, gates2 = sequential_latent_write_and_read(latent_params, hidden)
    logits2 = logits_from_hidden(model, hidden2)
    mx.eval(result.logits, logits2, result.write_gates, gates2)
    assert bool(mx.array_equal(result.logits, logits2))
    assert bool(mx.array_equal(result.write_gates, gates2))
