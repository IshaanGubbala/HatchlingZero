import mlx.core as mx

from reference.hz0a_mlx_metal import native_gdn2_fix_backward_fused_normalized, native_gdn2_fix_backward_normalized


def _random_inputs(bsz, steps, heads, key_dim, value_dim, seed):
    # decay_a fixed at -6.13 (matching test_hz0a_gdn2_fix_mlx.py's own
    # established convention, not sampled randomly): rate=exp(decay_a)
    # must stay small for the recurrence to remain numerically stable
    # over many real steps. A randomly-sampled decay_a can produce a
    # genuinely exploding (not buggy) recurrence at 17+ steps -- both
    # the baseline and fused kernels then legitimately diverge under
    # float32 rounding once magnitudes exceed ~1e20, which is a real
    # property of chaotic dynamics, not evidence of a kernel bug.
    mx.random.seed(seed)
    # native_gdn2_fix_backward_normalized ("normalized" in the name)
    # expects q/k ALREADY unit-normalized along the key_dim axis --
    # per its own source comment, MLX applies that normalization
    # before this kernel runs. Passing raw, unnormalized N(0,1) q/k
    # violates that invariant and produces genuine (not buggy)
    # numerical blowup over many real steps in BOTH the baseline and
    # fused kernels identically.
    q = mx.random.normal((bsz, steps, heads, key_dim))
    q = q / mx.sqrt(mx.sum(q * q, axis=-1, keepdims=True) + 1e-8)
    k = mx.random.normal((bsz, steps, heads, key_dim))
    k = k / mx.sqrt(mx.sum(k * k, axis=-1, keepdims=True) + 1e-8)
    v = mx.random.normal((bsz, steps, heads, value_dim))
    d = mx.random.normal((bsz, steps, heads, value_dim))
    e = mx.random.normal((bsz, steps, heads, value_dim))
    w = mx.random.normal((bsz, steps, heads, value_dim))
    initial = mx.random.normal((bsz, heads, value_dim, key_dim)) * 0.1
    decay_a = mx.array([-6.13], dtype=mx.float32)
    grad_output = mx.random.normal((bsz, steps, heads, value_dim))
    grad_final = mx.random.normal((bsz, heads, value_dim, key_dim)) * 0.1
    return q, k, v, d, e, w, initial, decay_a, grad_output, grad_final


_NAMES = ["grad_q", "grad_k", "grad_v", "grad_d", "grad_e", "grad_w", "grad_initial", "grad_decay"]


def test_fix_fused_backward_matches_baseline_small_shape():
    args = _random_inputs(bsz=2, steps=5, heads=3, key_dim=4, value_dim=4, seed=11)
    baseline = native_gdn2_fix_backward_normalized(*args)
    fused = native_gdn2_fix_backward_fused_normalized(*args)
    mx.eval(*baseline, *fused)
    for name, base_val, fused_val in zip(_NAMES, baseline, fused):
        assert bool(mx.allclose(base_val, fused_val, atol=2e-5, rtol=2e-5)), f"{name} mismatch"


def test_fix_fused_backward_matches_baseline_locked_shape():
    # value_dim == key_dim == 64, matching the locked A1 spec the fused
    # kernel's shared-buffer reduction requires.
    args = _random_inputs(bsz=2, steps=17, heads=2, key_dim=64, value_dim=64, seed=23)
    baseline = native_gdn2_fix_backward_normalized(*args)
    fused = native_gdn2_fix_backward_fused_normalized(*args)
    mx.eval(*baseline, *fused)
    max_errors = {}
    for name, base_val, fused_val in zip(_NAMES, baseline, fused):
        diff = mx.abs(base_val - fused_val)
        max_errors[name] = float(mx.max(diff))
        assert bool(mx.allclose(base_val, fused_val, atol=2e-5, rtol=2e-5)), f"{name} mismatch, max abs diff {max_errors[name]}"


def test_fix_fused_backward_uneven_chunk_length():
    # Non-power-of-two sequence length, chunk-boundary-style shape.
    args = _random_inputs(bsz=1, steps=37, heads=4, key_dim=16, value_dim=16, seed=41)
    baseline = native_gdn2_fix_backward_normalized(*args)
    fused = native_gdn2_fix_backward_fused_normalized(*args)
    mx.eval(*baseline, *fused)
    for name, base_val, fused_val in zip(_NAMES, baseline, fused):
        assert bool(mx.allclose(base_val, fused_val, atol=2e-5, rtol=2e-5)), f"{name} mismatch"


def test_fix_fused_backward_locked_301m_shape():
    # The real, live-training shape: dim=768/heads=12 -> key_dim=value_dim=64.
    args = _random_inputs(bsz=12, steps=128, heads=12, key_dim=64, value_dim=64, seed=7)
    baseline = native_gdn2_fix_backward_normalized(*args)
    fused = native_gdn2_fix_backward_fused_normalized(*args)
    mx.eval(*baseline, *fused)
    for name, base_val, fused_val in zip(_NAMES, baseline, fused):
        diff = mx.abs(base_val - fused_val)
        assert bool(mx.allclose(base_val, fused_val, atol=2e-5, rtol=2e-5)), f"{name} mismatch, max abs diff {float(mx.max(diff))}"


def test_fix_fused_backward_rejects_mismatched_value_key_dims():
    args = _random_inputs(bsz=1, steps=3, heads=2, key_dim=8, value_dim=4, seed=5)
    try:
        native_gdn2_fix_backward_fused_normalized(*args)
        assert False, "expected ValueError for value_dim != key_dim"
    except ValueError:
        pass
