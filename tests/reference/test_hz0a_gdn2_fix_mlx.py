import mlx.core as mx
import numpy as np

from reference.hz0a_mlx_model import GDN2Fix
from reference.hz0a_mlx_metal import native_gdn2_fix_forward
from reference.hz0a_mlx_metal import _fix_reference_forward, native_gdn2_fix_backward_normalized, native_gdn2_fix_forward_differentiable, native_gdn2_fix_forward_normalized
from reference.hz0a_gdn2_fix_reference import gdn2_fix_scan, normalize_keys


def test_mlx_gdn2_fix_forward_and_state_carry_are_finite():
    mx.random.seed(5)
    model = GDN2Fix(dim=16, heads=2)
    first_logits, first_states = model(mx.random.normal((2, 5, 16)))
    second_logits, second_states = model(mx.random.normal((2, 3, 16)), first_states)
    mx.eval(first_logits, second_logits, first_states, second_states)
    assert first_logits.shape == (2, 5, 16)
    assert second_logits.shape == (2, 3, 16)
    assert all(bool(mx.all(mx.isfinite(value)).item()) for value in (first_logits, second_logits, first_states, second_states))


def test_metal_fix_forward_matches_reference_on_tiny_inputs():
    rng = np.random.default_rng(8)
    bsz, steps, heads, width = 1, 3, 1, 4
    arrays = [mx.array(rng.normal(size=(bsz, steps, heads, width)).astype(np.float32)) for _ in range(5)]
    q, k, v, d, e = arrays
    w = mx.array(rng.normal(size=(bsz, steps, heads, width)).astype(np.float32))
    initial = mx.zeros((bsz, heads, width, width), dtype=mx.float32)
    decay_a = mx.array([-6.13], dtype=mx.float32)
    try:
        metal_out, metal_state = native_gdn2_fix_forward(q, k, v, d, e, w, initial, decay_a)
        mx.eval(metal_out, metal_state)
    except Exception as exc:
        import pytest
        pytest.skip(f"Metal kernel unavailable in this runtime: {exc}")
    q_np, k_np, v_np = (np.asarray(item) for item in (q, k, v))
    d_np, e_np, w_np = (np.asarray(item) for item in (d, e, w))
    q_np = normalize_keys(q_np)
    k_np = normalize_keys(k_np)
    softplus = np.maximum(d_np, 0) + np.log1p(np.exp(-np.abs(d_np)))
    alpha = np.exp(-np.exp(-6.13) * softplus)
    expected_out, expected_state = gdn2_fix_scan(
        q_np, k_np, v_np, alpha, 1 / (1 + np.exp(-e_np)), 1 / (1 + np.exp(-w_np)),
        np.zeros((bsz, heads, width, width), dtype=np.float32), normalize_key=False,
    )
    np.testing.assert_allclose(np.asarray(metal_out), expected_out, atol=2e-5, rtol=2e-5)
    np.testing.assert_allclose(np.asarray(metal_state), expected_state, atol=2e-5, rtol=2e-5)


def test_native_fix_vjp_matches_mlx_reference_vjp():
    rng = np.random.default_rng(12)
    shape = (1, 2, 1, 3)
    inputs = [mx.array(rng.normal(size=shape).astype(np.float32)) for _ in range(6)]
    inputs.append(mx.zeros((1, 1, 3, 3), dtype=mx.float32))
    inputs.append(mx.array([-6.13], dtype=mx.float32))

    def objective(fn, values):
        output, state = fn(*values)
        return mx.sum(output * output) + mx.sum(state * state)

    try:
        native_grads = mx.grad(lambda *values: objective(native_gdn2_fix_forward_differentiable, values), argnums=(0, 1, 2, 3, 4, 5, 6))(*inputs)
        reference_grads = mx.grad(lambda *values: objective(_fix_reference_forward, values), argnums=(0, 1, 2, 3, 4, 5, 6))(*inputs)
        mx.eval(*native_grads, *reference_grads)
    except Exception as exc:
        import pytest
        pytest.skip(f"native VJP unavailable in this runtime: {exc}")
    for native, reference in zip(native_grads, reference_grads):
        np.testing.assert_allclose(np.asarray(native), np.asarray(reference), atol=2e-5, rtol=2e-5)


def test_native_normalized_backward_matches_reference():
    rng = np.random.default_rng(19)
    shape = (1, 3, 1, 3)
    raw = [mx.array(rng.normal(size=shape).astype(np.float32)) for _ in range(6)]
    q, k = (item / mx.maximum(mx.linalg.norm(item, axis=-1, keepdims=True), 1e-6) for item in raw[:2])
    v, d, e, w = raw[2:]
    initial = mx.zeros((1, 1, 3, 3), dtype=mx.float32)
    decay_a = mx.array([-6.13], dtype=mx.float32)
    grad_output = mx.array(rng.normal(size=(1, 3, 1, 3)).astype(np.float32))
    grad_final = mx.array(rng.normal(size=(1, 1, 3, 3)).astype(np.float32))

    def reference(q_, k_, v_, d_, e_, w_, initial_, decay_):
        alpha = mx.exp(-mx.exp(decay_) * (mx.maximum(d_, 0) + mx.log1p(mx.exp(-mx.abs(d_)))))
        erase = mx.sigmoid(e_)
        write = mx.sigmoid(w_)
        state = initial_
        outputs = []
        for t in range(q_.shape[1]):
            decayed = state * alpha[:, t, :, None, :]
            old = mx.sum(decayed * (erase[:, t] * k_[:, t])[:, :, None, :], axis=-1)
            state = decayed + (write[:, t] * v_[:, t] - old)[:, :, :, None] * k_[:, t, :, None, :]
            outputs.append(mx.sum(state * q_[:, t, :, None, :], axis=-1))
        return mx.stack(outputs, axis=1), state

    def loss_fn(*values):
        out, state = reference(*values)
        return mx.sum(out * grad_output) + mx.sum(state * grad_final)

    reference_grads = mx.grad(loss_fn, argnums=(0, 1, 2, 3, 4, 5, 6, 7))(q, k, v, d, e, w, initial, decay_a)
    native_grads = native_gdn2_fix_backward_normalized(q, k, v, d, e, w, initial, decay_a, grad_output, grad_final)
    mx.eval(*reference_grads, *native_grads)
    for native, reference_value in zip(native_grads, reference_grads):
        np.testing.assert_allclose(np.asarray(native), np.asarray(reference_value), atol=3e-5, rtol=3e-5)
