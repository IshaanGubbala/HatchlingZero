"""Regression tests for the GDN-3 candidate recurrence/mixer -- locks in
the key-normalization fix found while building the real tiny-LM
comparison (`docs/restart/hz0a_gdn3_tiny_lm_comparison_results.md`):
unnormalized learned keys make the delta-rule projection unstable
(NaN), and the fix must keep the mechanism working correctly, not just
avoid the crash.
"""
import mlx.core as mx

from reference.hz0a_gdn3_candidate_mixer import GDN3CandidateMixer
from reference.hz0a_gdn3_candidate_recurrence import (
    step_current,
    step_delta_projection,
    step_delta_projection_plus_decay,
)


def test_delta_projection_stays_finite_with_unnormalized_keys():
    """The exact bug found in the tiny-LM comparison: before the
    normalization fix, a large-norm key produced NaN within one step."""
    state = mx.zeros((8, 8))
    q = mx.random.normal((8,), key=mx.random.key(1))
    large_key = mx.random.normal((8,), key=mx.random.key(2)) * 50.0  # deliberately unnormalized, large
    v = mx.random.normal((8,), key=mx.random.key(3))
    beta = mx.array(0.9)
    output, new_state = step_delta_projection(state, q, large_key, v, beta)
    assert bool(mx.all(mx.isfinite(output)))
    assert bool(mx.all(mx.isfinite(new_state)))


def test_delta_projection_plus_decay_stays_finite_with_unnormalized_keys():
    state = mx.zeros((8, 8))
    q = mx.random.normal((8,), key=mx.random.key(1))
    large_key = mx.random.normal((8,), key=mx.random.key(2)) * 50.0
    v = mx.random.normal((8,), key=mx.random.key(3))
    decay = mx.full((8,), 0.9)
    beta = mx.array(0.9)
    output, new_state = step_delta_projection_plus_decay(state, q, large_key, v, decay, beta)
    assert bool(mx.all(mx.isfinite(output)))
    assert bool(mx.all(mx.isfinite(new_state)))


def test_normalization_is_a_noop_for_already_unit_keys():
    """The overwrite benchmark's own hand-set one-hot keys are already
    unit-norm -- the fix must not change behavior there (re-verified
    against the benchmark's own recorded numbers after the fix landed)."""
    state = mx.zeros((4, 4))
    unit_key = mx.eye(4)[0]
    q = unit_key
    v = mx.array([1.0, 2.0, 3.0, 4.0])
    beta = mx.array(1.0)
    output, new_state = step_delta_projection(state, q, unit_key, v, beta)
    assert bool(mx.allclose(output, v, atol=1e-5))


def test_gdn3_candidate_mixer_forward_is_finite_on_random_input():
    """End-to-end sanity for the actual nn.Module used in the tiny-LM
    comparison, not just the standalone step function."""
    mixer = GDN3CandidateMixer(dim=32, heads=4)
    x = mx.random.normal((2, 16, 32), key=mx.random.key(7))
    output, state = mixer(x)
    assert output.shape == x.shape
    assert bool(mx.all(mx.isfinite(output)))
    assert bool(mx.all(mx.isfinite(state)))


def test_gdn3_candidate_mixer_is_differentiable():
    import mlx.nn as nn

    mixer = GDN3CandidateMixer(dim=16, heads=2)
    x = mx.random.normal((2, 8, 16), key=mx.random.key(7))

    def loss_fn(model):
        output, _ = model(x)
        return mx.sum(output)

    loss, grads = nn.value_and_grad(mixer, loss_fn)(mixer)
    assert bool(mx.isfinite(loss))
    assert bool(mx.isfinite(grads["in_proj"]["weight"]).all())
