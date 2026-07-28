import mlx.core as mx

from reference.hz0a_mlx_metal import _reference_forward, native_gdn2_forward, native_gdn2_forward_differentiable
from reference.hz0a_mlx_model import HZ0AMlxModel


def test_native_metal_gdn2_matches_mlx_recurrence():
    bsz, steps, heads, key_dim, value_dim = 1, 3, 1, 2, 2
    q = mx.arange(bsz * steps * heads * key_dim, dtype=mx.float32).reshape(bsz, steps, heads, key_dim) / 10
    k = q + 0.2
    v = q + 0.4
    d = mx.full((bsz, steps, heads, value_dim), -0.7)
    e = mx.full(d.shape, -1.1)
    w = mx.full(d.shape, -0.3)
    initial = mx.zeros((bsz, heads, value_dim, key_dim), dtype=mx.float32)
    native_y, native_state = native_gdn2_forward(q, k, v, d, e, w, initial)

    state = initial
    outputs = []
    for t in range(steps):
        state = mx.sigmoid(d[:, t, :, :, None]) * (1 - mx.sigmoid(e[:, t, :, :, None])) * state
        state = state + mx.sigmoid(w[:, t, :, :, None]) * v[:, t, :, :, None] * k[:, t, :, None, :]
        outputs.append(mx.sum(state * q[:, t, :, None, :], axis=-1))
    expected_y, expected_state = mx.stack(outputs, axis=1), state
    mx.eval(native_y, native_state, expected_y, expected_state)
    assert mx.allclose(native_y, expected_y, atol=1e-5, rtol=1e-5)
    assert mx.allclose(native_state, expected_state, atol=1e-5, rtol=1e-5)


def test_clean_mlx_model_can_opt_into_native_recurrence():
    model = HZ0AMlxModel(32, 16, 2, 2, 32, (), native_metal=True)
    tokens = mx.arange(6).reshape(1, 6) % 32
    logits, states = model(tokens)
    mx.eval(logits, *states)
    assert logits.shape == (1, 6, 32)
    assert states[0].shape == (1, 2, 8, 8)
    assert bool(mx.all(mx.isfinite(logits)))


def test_native_forward_vjp_produces_finite_gradients():
    q = mx.ones((1, 2, 1, 2), dtype=mx.float32)
    args = [q, q * 0.9, q * 0.8, q * -0.7, q * -1.1, q * -0.3, mx.zeros((1, 1, 2, 2))]

    def loss(*values):
        output, final_state = native_gdn2_forward_differentiable(*values)
        return mx.sum(output) + mx.sum(final_state)

    value, gradients = mx.value_and_grad(loss)(*args)
    mx.eval(value, *gradients)
    assert bool(mx.isfinite(value))
    assert all(bool(mx.all(mx.isfinite(gradient))) for gradient in gradients)


def test_native_backward_matches_reference_vjp():
    mx.random.seed(23)
    values = [mx.random.normal((1, 3, 1, 2)) for _ in range(6)]
    values.append(mx.random.normal((1, 1, 2, 2)))

    def native_loss(*args):
        output, final_state = native_gdn2_forward_differentiable(*args)
        return mx.sum(output * 0.7) + mx.sum(final_state * -0.2)

    def reference_loss(*args):
        output, final_state = _reference_forward(*args)
        return mx.sum(output * 0.7) + mx.sum(final_state * -0.2)

    _, native_gradients = mx.value_and_grad(native_loss)(*values)
    _, reference_gradients = mx.value_and_grad(reference_loss)(*values)
    mx.eval(*native_gradients, *reference_gradients)
    for native, reference in zip(native_gradients, reference_gradients):
        assert bool(mx.allclose(native, reference, atol=2e-5, rtol=2e-5))
