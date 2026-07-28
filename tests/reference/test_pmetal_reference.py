from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from reference.hz0a_gdn2_reference import HZ0ABlock, TinyHZ0AModel, gdn2_scan, init_state  # noqa: E402
from restart.hz0a_pmetal.python.pmetal_reference import (  # noqa: E402
    BlockForwardInputs,
    Gdn2ForwardInputs,
    TinyModelForwardInputs,
    block_forward,
    gdn2_forward,
    tiny_model_forward,
)


def test_pmetal_style_forward_matches_numpy_oracle() -> None:
    rng = np.random.default_rng(42)
    q = rng.normal(size=(2, 5, 3, 4)).astype(np.float32)
    k = rng.normal(size=(2, 5, 3, 4)).astype(np.float32)
    v = rng.normal(size=(2, 5, 3, 6)).astype(np.float32)
    decay = rng.normal(size=(2, 5, 3, 4)).astype(np.float32)
    erase = rng.normal(size=(2, 5, 3, 4)).astype(np.float32)
    write = rng.normal(size=(2, 5, 3, 6)).astype(np.float32)
    state = init_state(batch_size=2, num_heads=3, d_v=6, d_k=4)

    expected_out, expected_state = gdn2_scan(q, k, v, decay, erase, write, initial_state=state)
    actual = gdn2_forward(
        Gdn2ForwardInputs(
            q=q,
            k=k,
            v=v,
            decay_logits=decay,
            erase_logits=erase,
            write_logits=write,
            initial_state=state,
        )
    )

    np.testing.assert_allclose(actual.outputs, expected_out, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(actual.final_state, expected_state, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(actual.backward_cache.q, q)
    np.testing.assert_allclose(actual.backward_cache.initial_state, state)


def test_pmetal_style_recurrent_block_matches_reference_block() -> None:
    rng = np.random.default_rng(7)
    block = HZ0ABlock.init(
        rng=rng,
        d_model=16,
        num_heads=4,
        d_k=4,
        d_v=4,
        d_ff=32,
        is_attention=False,
    )
    x = rng.normal(size=(2, 5, 16)).astype(np.float32)
    state = init_state(batch_size=2, num_heads=4, d_v=4, d_k=4)
    expected_out, expected_state = block(x, state)
    actual = block_forward(BlockForwardInputs(block=block, x=x, state=state))
    np.testing.assert_allclose(actual.outputs, expected_out, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(actual.final_state, expected_state, rtol=1e-6, atol=1e-6)


def test_pmetal_style_attention_block_matches_reference_block() -> None:
    rng = np.random.default_rng(9)
    block = HZ0ABlock.init(
        rng=rng,
        d_model=12,
        num_heads=3,
        d_k=4,
        d_v=4,
        d_ff=24,
        is_attention=True,
    )
    x = rng.normal(size=(2, 4, 12)).astype(np.float32)
    expected_out, expected_state = block(x, None)
    actual = block_forward(BlockForwardInputs(block=block, x=x, state=None))
    np.testing.assert_allclose(actual.outputs, expected_out, rtol=1e-6, atol=1e-6)
    assert expected_state is None
    assert actual.final_state is None


def test_pmetal_style_tiny_model_loss_matches_reference() -> None:
    model = TinyHZ0AModel.init(
        rng_seed=1234,
        vocab_size=32,
        d_model=16,
        num_layers=3,
        num_heads=4,
        d_k=4,
        d_v=4,
        d_ff=32,
        attention_layer_indices=[1],
    )
    token_ids = np.array([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=np.int64)
    targets = np.array([[2, 3, 4, 5], [3, 2, 1, 0]], dtype=np.int64)

    expected_logits, expected_states = model(token_ids)
    actual = tiny_model_forward(TinyModelForwardInputs(model=model, token_ids=token_ids, targets=targets))

    np.testing.assert_allclose(actual.logits, expected_logits, rtol=1e-6, atol=1e-6)
    for actual_state, expected_state in zip(actual.states, expected_states):
        if expected_state is None:
            assert actual_state is None
        else:
            np.testing.assert_allclose(actual_state, expected_state, rtol=1e-6, atol=1e-6)
    assert actual.loss is not None
    assert np.isfinite(actual.loss)
