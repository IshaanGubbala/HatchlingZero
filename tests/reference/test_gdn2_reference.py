from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from reference.hz0a_gdn2_reference import (  # noqa: E402
    TinyHZ0AModel,
    cross_entropy_loss,
    gdn2_chunk_scan,
    gdn2_scan,
    init_state,
)


def _toy_inputs(batch: int = 2, steps: int = 7, heads: int = 2, d_k: int = 3, d_v: int = 4) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(0)
    q = rng.normal(size=(batch, steps, heads, d_k)).astype(np.float32)
    k = rng.normal(size=(batch, steps, heads, d_k)).astype(np.float32)
    v = rng.normal(size=(batch, steps, heads, d_v)).astype(np.float32)
    decay = rng.normal(size=(batch, steps, heads, d_k)).astype(np.float32)
    erase = rng.normal(size=(batch, steps, heads, d_k)).astype(np.float32)
    write = rng.normal(size=(batch, steps, heads, d_v)).astype(np.float32)
    return q, k, v, decay, erase, write


def test_gdn2_t1_shape_and_finiteness() -> None:
    q, k, v, decay, erase, write = _toy_inputs(steps=1)
    out, state = gdn2_scan(q, k, v, decay, erase, write)
    assert out.shape == (2, 1, 2, 4)
    assert state.shape == (2, 2, 4, 3)
    assert np.isfinite(out).all()
    assert np.isfinite(state).all()


def test_gdn2_chunked_matches_full_sequence() -> None:
    q, k, v, decay, erase, write = _toy_inputs(steps=11)
    full_out, full_state = gdn2_scan(q, k, v, decay, erase, write)
    chunk_out, chunk_state = gdn2_chunk_scan(q, k, v, decay, erase, write, chunk_size=4)
    np.testing.assert_allclose(chunk_out, full_out, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(chunk_state, full_state, rtol=1e-6, atol=1e-6)


def test_gdn2_initial_state_carry_changes_result_and_is_repeatable() -> None:
    q, k, v, decay, erase, write = _toy_inputs(steps=5)
    init = init_state(batch_size=2, num_heads=2, d_v=4, d_k=3)
    init[:] = 0.25
    out_a, state_a = gdn2_scan(q, k, v, decay, erase, write, initial_state=init)
    out_b, state_b = gdn2_scan(q, k, v, decay, erase, write, initial_state=init)
    np.testing.assert_allclose(out_a, out_b, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(state_a, state_b, rtol=1e-6, atol=1e-6)


def test_model_reset_and_determinism() -> None:
    model = TinyHZ0AModel.init(
        rng_seed=123,
        vocab_size=32,
        d_model=16,
        num_layers=3,
        num_heads=4,
        d_k=4,
        d_v=4,
        d_ff=32,
        attention_layer_indices=[1],
    )
    tokens = np.array([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=np.int64)
    logits_a, states_a = model(tokens)
    logits_b, states_b = model(tokens)
    np.testing.assert_allclose(logits_a, logits_b, rtol=1e-6, atol=1e-6)
    assert all(np.isfinite(logits_a).ravel())
    for state_a, state_b in zip(states_a, states_b):
        if state_a is None:
            assert state_b is None
        else:
            np.testing.assert_allclose(state_a, state_b, rtol=1e-6, atol=1e-6)


def test_model_loss_is_finite_for_non_power_of_two_length() -> None:
    model = TinyHZ0AModel.init(
        rng_seed=321,
        vocab_size=48,
        d_model=24,
        num_layers=4,
        num_heads=4,
        d_k=6,
        d_v=6,
        d_ff=48,
        attention_layer_indices=[2],
    )
    tokens = np.array([[1, 5, 9, 2, 7], [3, 8, 4, 6, 0]], dtype=np.int64)
    targets = np.array([[5, 9, 2, 7, 1], [8, 4, 6, 0, 3]], dtype=np.int64)
    logits, _ = model(tokens)
    loss = cross_entropy_loss(logits, targets)
    assert np.isfinite(logits).all()
    assert np.isfinite(loss)


def test_attention_stress_path_is_warning_free_and_finite() -> None:
    model = TinyHZ0AModel.init(
        rng_seed=99,
        vocab_size=32,
        d_model=16,
        num_layers=3,
        num_heads=4,
        d_k=4,
        d_v=4,
        d_ff=32,
        attention_layer_indices=[1],
    )
    tokens = np.arange(32, dtype=np.int64).reshape(2, 16)
    logits, _ = model(tokens)
    assert np.isfinite(logits).all()
