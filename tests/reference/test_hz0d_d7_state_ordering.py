"""HZ-0D D7: state ordering tests (reference/hz0d_d7_state_ordering.py).

Checked against the ACTUAL frozen checkpoint, matching this project's
established convention. Skips if the checkpoint isn't present locally.
Locks in D7's own exit gate: "state transitions are deterministic and
unambiguous," plus "prevent duplicate writes and feedback loops."
"""
from __future__ import annotations

import mlx.core as mx
import pytest

from reference.hz0b_b8_latent_write import init_latent_write_controller
from reference.hz0d_d6_integration import ATTENTION_INDICES, d6_fast_weight_config
from reference.hz0d_d7_state_ordering import d7_process_sequence
from reference.hz0d_fast_weights import init_fast_weights
from reference.hz0d_isolated_simulator import Task
from scripts.hz0b_b11_baseline_comparison import CHECKPOINT, load_frozen_model
from scripts.hz0c_c6_conditional_attention_eval import fixed_matched_trigger

pytestmark = pytest.mark.skipif(
    not (CHECKPOINT / "state.json").exists(),
    reason="frozen HZ-0A checkpoint not present locally (gitignored under outputs/)",
)


def _real_fast_update_task(model, *, seed: int, k_train: int = 24, k_held_out: int = 8, rule_scale: float = 0.05) -> Task:
    layer = model.blocks[ATTENTION_INDICES[0]]
    real_w, real_b = layer.mixer.out.weight, layer.mixer.out.bias
    key = mx.random.key(seed)
    k_a, k_b, k_x = mx.random.split(key, 3)
    true_a = mx.random.normal((768, 16), key=k_a) * rule_scale
    true_b = mx.random.normal((16, 768), key=k_b) * rule_scale
    true_delta = true_a @ true_b
    total = k_train + k_held_out
    x = mx.random.normal((total, 768), key=k_x) * 0.05
    y = x @ (real_w + true_delta).T + real_b
    return Task(
        base_weight=real_w, base_bias=real_b, true_delta=true_delta,
        train_x=x[:k_train], train_y=y[:k_train], held_out_x=x[k_train:], held_out_y=y[k_train:],
    )


def _setup():
    model, _payload = load_frozen_model()
    config = d6_fast_weight_config()
    fast_state = init_fast_weights(config)
    latent_params = init_latent_write_controller(d_model=768, key_dim=64, value_dim=64, seed=0)
    tokens = mx.array([[1, 45, 982, 12, 7, 300, 44, 1023, 55, 66, 77, 88]])
    trigger = fixed_matched_trigger(1, tokens.shape[1], 0.15)
    return model, config, fast_state, latent_params, tokens, trigger


def test_pipeline_is_deterministic():
    """Same inputs, run twice, bit-exact outputs -- D7's exit gate,
    "state transitions are deterministic," checked directly on both the
    logits and the memory write gates, not just assumed from the
    underlying pieces each being individually deterministic."""
    model, config, fast_state, latent_params, tokens, trigger = _setup()
    result_a = d7_process_sequence(model, tokens, trigger, latent_params, fast_state, config)
    result_b = d7_process_sequence(model, tokens, trigger, latent_params, fast_state, config)
    mx.eval(result_a.logits, result_b.logits, result_a.write_gates, result_b.write_gates)
    assert bool(mx.array_equal(result_a.logits, result_b.logits))
    assert bool(mx.array_equal(result_a.write_gates, result_b.write_gates))


def test_no_fast_update_by_default_and_state_unchanged():
    """"At most one," not "exactly one": with no `fast_update_task`
    given, step 7 must be skipped entirely -- `fast_weight_updated` is
    False and the returned `fast_state` is bit-identical to the input
    (not just numerically close)."""
    model, config, fast_state, latent_params, tokens, trigger = _setup()
    result = d7_process_sequence(model, tokens, trigger, latent_params, fast_state, config)
    assert result.fast_weight_updated is False
    assert bool(mx.array_equal(result.fast_state.a_fast, fast_state.a_fast))
    assert bool(mx.array_equal(result.fast_state.b_fast, fast_state.b_fast))
    assert int(result.fast_state.update_count) == int(fast_state.update_count)


def test_fast_update_touches_exactly_the_named_layer_and_nothing_else():
    """"Prevent duplicate writes": a fast-weight update targeting layer
    0 must change ONLY layer 0's factors -- every other of the 6 anchor
    layers' realized delta must stay EXACTLY zero (bit-exact, not just
    small), and `update_count` must increase by exactly one, not more."""
    model, config, fast_state, latent_params, tokens, trigger = _setup()
    task = _real_fast_update_task(model, seed=1)
    result = d7_process_sequence(
        model, tokens, trigger, latent_params, fast_state, config,
        fast_update_layer_index=0, fast_update_task=task,
    )
    mx.eval(result.fast_state.a_fast, result.fast_state.b_fast)

    assert result.fast_weight_updated is True
    assert int(result.fast_state.update_count) == int(fast_state.update_count) + 1
    layer0_delta_norm = float(mx.sqrt(mx.sum((result.fast_state.a_fast[0] @ result.fast_state.b_fast[0]) ** 2)))
    assert layer0_delta_norm > 1e-4, "expected layer 0's delta to be genuinely nonzero after the update"
    for layer in range(1, config.num_layers):
        delta = result.fast_state.a_fast[layer] @ result.fast_state.b_fast[layer]
        assert bool(mx.array_equal(delta, mx.zeros_like(delta))), f"layer {layer} should be untouched by a layer-0 update"


def test_fast_update_does_not_feed_back_into_this_calls_own_logits():
    """No feedback loop: the logits this call produces must be computed
    with the ORIGINAL `fast_state`, not the post-update one -- an
    update, if applied, can only affect FUTURE calls, never retroactively
    change the output already produced this call. Checked directly by
    comparing against a separate call that applies no update at all:
    the logits must match exactly, even though the WITH-update call's
    returned `fast_state` differs."""
    model, config, fast_state, latent_params, tokens, trigger = _setup()
    task = _real_fast_update_task(model, seed=2)

    with_update = d7_process_sequence(
        model, tokens, trigger, latent_params, fast_state, config,
        fast_update_layer_index=0, fast_update_task=task,
    )
    without_update = d7_process_sequence(model, tokens, trigger, latent_params, fast_state, config)
    mx.eval(with_update.logits, without_update.logits)

    assert bool(mx.array_equal(with_update.logits, without_update.logits))
    assert not bool(mx.array_equal(with_update.fast_state.a_fast, without_update.fast_state.a_fast))


def test_memory_write_gate_produces_exactly_one_value_per_token_position():
    """"Perform at most one memory write" per token: `write_gates` has
    exactly one scalar per token position (`[batch, seq]`, not
    `[batch, seq, num_slots]` or anything that could imply multiple
    competing writes per position) -- structurally impossible for a
    position to register more than one write signal."""
    model, config, fast_state, latent_params, tokens, trigger = _setup()
    result = d7_process_sequence(model, tokens, trigger, latent_params, fast_state, config)
    assert result.write_gates.shape == (tokens.shape[0], tokens.shape[1])


def test_fast_update_requires_layer_index():
    """A caller supplying `fast_update_task` without
    `fast_update_layer_index` is an unambiguous programming error, not
    a silently-defaulted "update everything" -- fails loudly."""
    model, config, fast_state, latent_params, tokens, trigger = _setup()
    task = _real_fast_update_task(model, seed=3)
    with pytest.raises(ValueError):
        d7_process_sequence(model, tokens, trigger, latent_params, fast_state, config, fast_update_task=task)
