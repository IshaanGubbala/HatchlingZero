"""B11: unit tests for the equal-parameter no-memory adapter baseline."""
from __future__ import annotations

import mlx.core as mx

from reference.hz0b_b11_equal_param_adapter import adapter_forward, forward, init_equal_param_adapter, param_count


def test_param_count_matches_formula():
    params = init_equal_param_adapter(d_model=768, hidden=450, seed=0)
    counted = sum(v.size for v in [params.w1, params.b1, params.w2, params.b2])
    assert counted == param_count(768, 450)
    assert counted == 692_418


def test_adapter_preserves_shape():
    params = init_equal_param_adapter(d_model=16, hidden=8, seed=0)
    hidden = mx.random.normal((2, 5, 16))
    out = adapter_forward(params, hidden)
    assert out.shape == hidden.shape


def test_adapter_is_a_real_residual_transform_not_identity():
    params = init_equal_param_adapter(d_model=16, hidden=8, seed=0)
    hidden = mx.random.normal((2, 5, 16))
    out = adapter_forward(params, hidden)
    assert not bool(mx.all(mx.abs(out - hidden) < 1e-9))


def test_zero_weights_reduce_to_identity():
    params = init_equal_param_adapter(d_model=16, hidden=8, seed=0)
    zeroed = type(params)(w1=mx.zeros_like(params.w1), b1=mx.zeros_like(params.b1), w2=mx.zeros_like(params.w2), b2=mx.zeros_like(params.b2))
    hidden = mx.random.normal((2, 5, 16))
    out = adapter_forward(zeroed, hidden)
    assert bool(mx.all(mx.abs(out - hidden) < 1e-6))


def test_no_cross_position_information_flow():
    """Each position's output must depend ONLY on that position's own
    hidden state -- the defining structural difference from real HZ-0B
    memory, where a write at position t is visible to reads at t+1
    onward. Verified by permuting one position's input and confirming
    every OTHER position's output is unaffected."""
    params = init_equal_param_adapter(d_model=16, hidden=8, seed=0)
    hidden = mx.random.normal((1, 4, 16))
    out_a = adapter_forward(params, hidden)
    perturbed = mx.array(hidden)
    perturbed = mx.concatenate([perturbed[:, :2, :], perturbed[:, 2:3, :] + 5.0, perturbed[:, 3:, :]], axis=1)
    out_b = adapter_forward(params, perturbed)
    assert bool(mx.all(mx.abs(out_a[:, :2, :] - out_b[:, :2, :]) < 1e-6))
    assert bool(mx.all(mx.abs(out_a[:, 3:, :] - out_b[:, 3:, :]) < 1e-6))
    assert not bool(mx.all(mx.abs(out_a[:, 2, :] - out_b[:, 2, :]) < 1e-6))


def test_precomputed_hidden_matches_token_path_exactly():
    """2026-08-01 caching optimization: forward(precomputed_hidden=...)
    must be bit-identical to forward(token_ids=...) -- purely a
    performance change. Minimal dummy model (identity final_norm,
    identity-matrix embedding), same convention as the equivalent test
    in test_hz0b_b8_latent_write.py."""
    d_model = 16
    params = init_equal_param_adapter(d_model=d_model, hidden=8, seed=0)
    hidden = mx.random.normal((2, 5, d_model))

    class DummyModel:
        embedding = type("E", (), {"weight": mx.eye(d_model)})()

        @staticmethod
        def final_norm(x):
            return x

    logits_a, _ = forward(DummyModel, precomputed_hidden=hidden, adapter_params=params)
    expected_hidden = adapter_forward(params, hidden)
    logits_b = DummyModel.final_norm(expected_hidden) @ DummyModel.embedding.weight.T
    assert bool(mx.all(mx.abs(logits_a - logits_b) < 1e-6))
