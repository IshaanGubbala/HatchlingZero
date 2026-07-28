from __future__ import annotations

import numpy as np
import pytest

from reference.hz0a_gdn2_reference import CausalSelfAttention, TinyHZ0AModel
from reference.hz0a_inference import (
    attention_decode_step,
    decode_tokenwise,
    deserialize_attention_cache,
    deserialize_states,
    prefill,
    reset_states,
    serialize_attention_cache,
    serialize_states,
)


def make_model() -> TinyHZ0AModel:
    return TinyHZ0AModel.init(5, 32, 16, 3, 4, 4, 4, 32, attention_layer_indices=[])


def test_prefill_and_tokenwise_decode_are_equivalent() -> None:
    model = make_model()
    tokens = np.arange(14, dtype=np.int64).reshape(2, 7) % 32
    full = prefill(model, tokens)
    incremental = decode_tokenwise(model, tokens)
    np.testing.assert_allclose(incremental.logits, full.logits, rtol=1e-6, atol=1e-6)
    for actual, expected in zip(incremental.states, full.states):
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_decode_state_serialization_and_reset() -> None:
    model = make_model()
    tokens = np.arange(10, dtype=np.int64).reshape(2, 5) % 32
    first = decode_tokenwise(model, tokens[:, :3])
    restored = deserialize_states(serialize_states(first.states))
    continued = decode_tokenwise(model, tokens[:, 3:], restored)
    full = decode_tokenwise(model, tokens)
    np.testing.assert_allclose(continued.logits, full.logits[:, 3:], rtol=1e-6, atol=1e-6)
    reset = reset_states(model, 2)
    fresh = decode_tokenwise(model, tokens[:, :2], reset)
    np.testing.assert_allclose(fresh.logits, full.logits[:, :2], rtol=1e-6, atol=1e-6)


def test_decode_rejects_attention_models() -> None:
    model = TinyHZ0AModel.init(5, 32, 16, 3, 4, 4, 4, 32, attention_layer_indices=[1])
    with pytest.raises(ValueError, match="no attention"):
        prefill(model, np.ones((1, 2), dtype=np.int64))


def test_attention_kv_cache_matches_full_causal_attention_and_serializes() -> None:
    attention = CausalSelfAttention.init(np.random.default_rng(41), d_model=16, num_heads=4)
    tokens = np.arange(10, dtype=np.int64).reshape(2, 5) % 32
    x = np.random.default_rng(42).normal(size=(2, 5, 16)).astype(np.float32)
    full = attention(x)
    cache = None
    outputs = []
    for index in range(x.shape[1]):
        output, cache = attention_decode_step(attention, x[:, index:index + 1], cache)
        outputs.append(output)
    np.testing.assert_allclose(np.concatenate(outputs, axis=1), full, rtol=1e-6, atol=1e-6)
    restored = deserialize_attention_cache(serialize_attention_cache(cache))
    np.testing.assert_array_equal(restored.key, cache.key)
    np.testing.assert_array_equal(restored.value, cache.value)
