import mlx.core as mx

from reference.hz0a_mlx_metal import native_gdn2_backward, native_gdn2_backward_fused


def _random_inputs(bsz, steps, heads, key_dim, value_dim, seed):
    mx.random.seed(seed)
    q = mx.random.normal((bsz, steps, heads, key_dim))
    k = mx.random.normal((bsz, steps, heads, key_dim))
    v = mx.random.normal((bsz, steps, heads, value_dim))
    d = mx.sigmoid(mx.random.normal((bsz, steps, heads, value_dim)))
    e = mx.sigmoid(mx.random.normal((bsz, steps, heads, value_dim)))
    w = mx.sigmoid(mx.random.normal((bsz, steps, heads, value_dim)))
    initial = mx.random.normal((bsz, heads, value_dim, key_dim))
    grad_output = mx.random.normal((bsz, steps, heads, value_dim))
    grad_final = mx.random.normal((bsz, heads, value_dim, key_dim))
    return q, k, v, d, e, w, initial, grad_output, grad_final


def test_fused_backward_matches_baseline_small_shape():
    args = _random_inputs(bsz=2, steps=5, heads=3, key_dim=4, value_dim=4, seed=11)
    baseline = native_gdn2_backward(*args)
    fused = native_gdn2_backward_fused(*args)
    mx.eval(*baseline, *fused)
    names = ["grad_q", "grad_k", "grad_v", "grad_d", "grad_e", "grad_w", "grad_initial"]
    for name, base_val, fused_val in zip(names, baseline, fused):
        assert bool(mx.allclose(base_val, fused_val, atol=2e-5, rtol=2e-5)), f"{name} mismatch"


def test_fused_backward_matches_baseline_locked_shape():
    # value_dim == key_dim == 64, matching the locked A1 spec the fused
    # kernel's shared-buffer reduction requires.
    args = _random_inputs(bsz=2, steps=17, heads=2, key_dim=64, value_dim=64, seed=23)
    baseline = native_gdn2_backward(*args)
    fused = native_gdn2_backward_fused(*args)
    mx.eval(*baseline, *fused)
    names = ["grad_q", "grad_k", "grad_v", "grad_d", "grad_e", "grad_w", "grad_initial"]
    max_errors = {}
    for name, base_val, fused_val in zip(names, baseline, fused):
        diff = mx.abs(base_val - fused_val)
        max_errors[name] = float(mx.max(diff))
        assert bool(mx.allclose(base_val, fused_val, atol=2e-5, rtol=2e-5)), f"{name} mismatch, max abs diff {max_errors[name]}"


def test_fused_backward_uneven_chunk_length():
    # Non-power-of-two sequence length, chunk-boundary-style shape.
    args = _random_inputs(bsz=1, steps=37, heads=4, key_dim=16, value_dim=16, seed=41)
    baseline = native_gdn2_backward(*args)
    fused = native_gdn2_backward_fused(*args)
    mx.eval(*baseline, *fused)
    for base_val, fused_val in zip(baseline, fused):
        assert bool(mx.allclose(base_val, fused_val, atol=2e-5, rtol=2e-5))


def test_fused_backward_rejects_mismatched_value_key_dims():
    args = _random_inputs(bsz=1, steps=3, heads=2, key_dim=8, value_dim=4, seed=5)
    try:
        native_gdn2_backward_fused(*args)
        assert False, "expected ValueError for value_dim != key_dim"
    except ValueError:
        pass
