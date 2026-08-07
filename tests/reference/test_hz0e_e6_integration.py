"""HZ-0E E6: selected upper-FFN integration on the real frozen checkpoint."""
from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import mlx.utils
import pytest

from reference.hz0e_e6_integration import TARGET_LAYERS, cross_entropy_loss, forward_e6, init_e6_layers, load_e6_layers
from scripts.hz0b_b11_baseline_comparison import CHECKPOINT, load_frozen_model
from scripts.hz0c_c3_trigger_simulator import GENERAL_DATA_PATH, load_real_sequences


pytestmark = pytest.mark.skipif(
    not (CHECKPOINT / "state.json").exists() or not Path(GENERAL_DATA_PATH).exists(),
    reason="real HZ-0A checkpoint / corpus not present locally",
)


def _tokens() -> mx.array:
    rows = load_real_sequences(GENERAL_DATA_PATH, 2)
    return mx.array([row[:64] for row in rows], dtype=mx.int32)


def _state_equal(a, b) -> bool:
    if isinstance(a, tuple):
        return isinstance(b, tuple) and all(_state_equal(x, y) for x, y in zip(a, b))
    if a is None or b is None:
        return a is b
    return bool(mx.array_equal(a, b))


def test_inactive_e6_is_bit_identical_to_frozen_backbone():
    model, _ = load_frozen_model()
    tokens = _tokens()
    reference_logits, reference_states = model(tokens)
    e6 = forward_e6(model, tokens, enabled=False)
    mx.eval(reference_logits, e6.logits, *reference_states, *e6.states)
    assert bool(mx.array_equal(reference_logits, e6.logits))
    assert all(_state_equal(a, b) for a, b in zip(reference_states, e6.states))


def test_active_e6_replaces_exactly_the_declared_upper_ffns_and_is_finite():
    model, _ = load_frozen_model()
    tokens = _tokens()
    e6 = forward_e6(model, tokens, moe_layers=init_e6_layers(model, seed=7))
    mx.eval(e6.logits, *e6.states)
    assert sorted(e6.diagnostics) == list(TARGET_LAYERS)
    assert bool(mx.all(mx.isfinite(e6.logits)))
    assert all(int(diag.expert_counts.sum()) + int(diag.fallback_count) == tokens.size for diag in e6.diagnostics.values())


def test_moe_does_not_change_recurrent_or_attention_state_transitions():
    model, _ = load_frozen_model()
    tokens = _tokens()
    dense = forward_e6(model, tokens, enabled=False)
    active = forward_e6(model, tokens, moe_layers=init_e6_layers(model, seed=7))
    mx.eval(*dense.states, *active.states)
    # MoE is inserted after the mixer+residual and never replaces mixer
    # state storage. Layer 27 therefore matches exactly; later mixer states
    # may change legitimately because they consume the new residual stream.
    assert _state_equal(dense.states[TARGET_LAYERS[0]], active.states[TARGET_LAYERS[0]])
    for state in active.states:
        if isinstance(state, tuple):
            assert all(bool(mx.all(mx.isfinite(value))) for value in state)
        elif state is not None:
            assert bool(mx.all(mx.isfinite(state)))


def test_e6_does_not_mutate_frozen_model_parameters():
    model, _ = load_frozen_model()
    before = [(key, mx.array(value)) for key, value in mlx.utils.tree_flatten(model.parameters())]
    e6 = forward_e6(model, _tokens(), moe_layers=init_e6_layers(model, seed=7))
    mx.eval(e6.logits)
    after = mlx.utils.tree_flatten(model.parameters())
    assert [key for key, _ in before] == [key for key, _ in after]
    assert all(bool(mx.array_equal(old, new)) for (_, old), (_, new) in zip(before, after))


def test_e6_calibration_artifact_round_trips(tmp_path):
    model, _ = load_frozen_model()
    source = init_e6_layers(model, seed=7)
    arrays = {
        f"layer_{index}_{name}": value
        for index, params in source.items()
        for name, value in params.__dict__.items()
    }
    path = tmp_path / "e6.npz"
    mx.savez(str(path), **arrays)
    loaded = load_e6_layers(model, path)
    for index in TARGET_LAYERS:
        for name, value in source[index].__dict__.items():
            assert bool(mx.array_equal(value, getattr(loaded[index], name)))


def test_e6_loader_rejects_malformed_tensor_shape(tmp_path):
    model, _ = load_frozen_model()
    source = init_e6_layers(model, seed=7)
    arrays = {
        f"layer_{index}_{name}": value
        for index, params in source.items()
        for name, value in params.__dict__.items()
    }
    arrays["layer_27_router_b"] = mx.zeros((99,))
    path = tmp_path / "bad-e6.npz"
    mx.savez(str(path), **arrays)
    with pytest.raises(ValueError, match="invalid E6 tensor shape"):
        load_e6_layers(model, path)
