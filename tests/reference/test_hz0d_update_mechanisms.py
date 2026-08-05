"""HZ-0D D3: update-mechanism comparison tests
(reference/hz0d_update_mechanisms.py). Locks in the real, measured
findings from `docs/restart/hz0d_d3_update_mechanism_results.md` as
regression tests -- not just "each method runs without error," but the
actual comparative claims the D3 exit gate rests on.
"""
from __future__ import annotations

import mlx.core as mx

from reference.hz0d_fast_weights import FastWeightConfig, FastWeightState, effective_delta, init_fast_weights
from reference.hz0d_isolated_simulator import make_task, held_out_generalization_loss
from reference.hz0d_update_mechanisms import (
    delta_prediction_update, error_conditioned_update, gradient_descent_update, hebbian_delta_rule_update,
)

CONFIG = FastWeightConfig(dim=8, rank=2, num_layers=1, max_delta_norm=10.0)


def _clean_and_noisy_tasks(seed: int):
    task = make_task(CONFIG, seed=seed, k_train=6, k_held_out=16, rule_scale=0.3)
    noise = mx.random.normal(task.train_y.shape, key=mx.random.key(99)) * 0.3
    noisy_task = task.__class__(
        base_weight=task.base_weight, base_bias=task.base_bias, true_delta=task.true_delta,
        train_x=task.train_x, train_y=task.train_y + noise,
        held_out_x=task.held_out_x, held_out_y=task.held_out_y,
    )
    return task, noisy_task


def test_all_four_mechanisms_produce_finite_bounded_state():
    task = make_task(CONFIG, seed=5, k_train=6, k_held_out=16, rule_scale=0.3)
    for state, _ in [
        gradient_descent_update(task, init_fast_weights(CONFIG), CONFIG, steps=50, lr=0.02),
        hebbian_delta_rule_update(task, init_fast_weights(CONFIG), CONFIG, passes=10, lr=0.05),
        delta_prediction_update(task, init_fast_weights(CONFIG), CONFIG),
        error_conditioned_update(task, init_fast_weights(CONFIG), CONFIG, steps=50, base_lr=0.05),
    ]:
        assert bool(mx.all(mx.isfinite(state.a_fast)))
        assert bool(mx.all(mx.isfinite(state.b_fast)))
        delta = effective_delta(state, 0)
        norm = float(mx.sqrt(mx.sum(delta * delta)))
        assert norm <= CONFIG.max_delta_norm + 1e-3


def test_hebbian_leaves_a_fast_unchanged_when_clipping_does_not_engage():
    """Direct check of the method's own defining property: a_fast (the
    decoder factor) is held fixed by the rule itself, not just by
    accident -- verified with a small enough learning rate/pass count
    that the shared clip helper never engages (so any change would have
    to come from the Hebbian rule, not the safety clip)."""
    task = make_task(CONFIG, seed=6, k_train=6, k_held_out=16, rule_scale=0.3)
    initial = init_fast_weights(CONFIG)
    state, _ = hebbian_delta_rule_update(task, initial, CONFIG, passes=2, lr=0.01)
    assert bool(mx.array_equal(state.a_fast, initial.a_fast))
    assert not bool(mx.array_equal(state.b_fast, initial.b_fast))


def test_delta_prediction_fits_clean_training_data_closely_but_not_exactly():
    """With the fix (ridge=1.0 regularization, see the function's own
    docstring for why), the closed-form solve is DELIBERATELY not an
    exact interpolator anymore -- it should still fit clean training
    data closely (small training loss), but the whole point of the
    fix is that it no longer drives training loss to ~0, which is what
    made the unregularized version collapse under label noise (see
    `test_delta_prediction_is_robust_to_label_noise_after_the_ridge_fix`
    below)."""
    task = make_task(CONFIG, seed=7, k_train=6, k_held_out=16, rule_scale=0.3)
    _, diagnostics = delta_prediction_update(task, init_fast_weights(CONFIG), CONFIG, ridge=1.0)
    assert 1e-4 < diagnostics["final_train_loss"] < 0.1


def test_delta_prediction_ridge_strength_trades_off_clean_fit_against_noise_robustness():
    """Direct confirmation that `ridge` behaves as regularization should:
    more ridge -> worse clean-data fit, monotonically."""
    task = make_task(CONFIG, seed=7, k_train=6, k_held_out=16, rule_scale=0.3)
    train_losses = []
    for ridge in [0.1, 1.0, 3.0]:
        _, diagnostics = delta_prediction_update(task, init_fast_weights(CONFIG), CONFIG, ridge=ridge)
        train_losses.append(diagnostics["final_train_loss"])
    assert train_losses == sorted(train_losses), f"training loss should increase with more ridge: {train_losses}"


def test_error_conditioned_gate_is_bounded_and_shrinks_with_error():
    """The gate (`tanh(error_norm / error_scale)`) must stay in [0, 1)
    and trend toward smaller values as training error shrinks -- checked
    via the reported mean_gate across a run that clearly converges
    (compared against a run stopped very early, before much convergence
    has happened)."""
    task = make_task(CONFIG, seed=8, k_train=6, k_held_out=16, rule_scale=0.3)
    _, short_run = error_conditioned_update(task, init_fast_weights(CONFIG), CONFIG, steps=5, base_lr=0.05)
    _, long_run = error_conditioned_update(task, init_fast_weights(CONFIG), CONFIG, steps=400, base_lr=0.05)
    assert 0.0 <= short_run["mean_gate"] < 1.0
    assert 0.0 <= long_run["mean_gate"] < 1.0
    assert long_run["mean_gate"] < short_run["mean_gate"], (
        "expected the gate to shrink on average as the run converges and error drops"
    )


def test_gradient_descent_beats_hebbian_on_clean_data():
    """Real capacity-limitation finding: Hebbian only ever updates half
    the parameters (a_fast held fixed), and no amount of extra tuning
    closes the gap (swept passes/lr up to 100 passes x lr=0.2 in the
    real investigation; held-out loss plateaus around 1.0-1.2, never
    approaching gradient descent's ~0.12)."""
    task = make_task(CONFIG, seed=1, k_train=6, k_held_out=16, rule_scale=0.3)
    gd_state, _ = gradient_descent_update(task, init_fast_weights(CONFIG), CONFIG, steps=400, lr=0.02)
    hebbian_state, _ = hebbian_delta_rule_update(task, init_fast_weights(CONFIG), CONFIG, passes=100, lr=0.1)
    gd_loss = held_out_generalization_loss(task, gd_state)
    hebbian_loss = held_out_generalization_loss(task, hebbian_state)
    assert gd_loss < hebbian_loss * 0.5, f"expected gradient descent to clearly beat Hebbian: {gd_loss} vs {hebbian_loss}"


def _unregularized_pinv_delta_prediction(task, config):
    """Reproduces EXACTLY the first version of `delta_prediction_update`
    (plain Moore-Penrose pseudo-inverse, no ridge) -- kept here, not as
    a `ridge=0.0` call, because `ridge=0.0` through the FIXED function's
    `mx.linalg.solve`-based normal equations has a DIFFERENT failure
    mode (the Gram matrix `X.T @ X` is singular/rank-deficient whenever
    `k_train < dim`, so `solve` is numerically unstable there regardless
    of label noise -- not the same thing as the original pinv-based
    exact-interpolation-overfits-noise finding this test locks in)."""
    residual = task.train_y - (task.train_x @ task.base_weight.T + task.base_bias)
    delta_t = mx.linalg.pinv(task.train_x, stream=mx.cpu) @ residual
    u, s, vt = mx.linalg.svd(delta_t.T, stream=mx.cpu)
    r = config.rank
    sqrt_s = mx.sqrt(mx.clip(s[:r], 0.0, None))
    a_layer = u[:, :r] * sqrt_s[None, :]
    b_layer = sqrt_s[:, None] * vt[:r, :]
    return FastWeightState(a_fast=mx.expand_dims(a_layer, 0), b_fast=mx.expand_dims(b_layer, 0), update_count=mx.array(0))


def test_unregularized_delta_prediction_collapses_under_label_noise():
    """Locks in the ORIGINAL D3 finding as a real regression test --
    the whole reason the ridge fix below exists."""
    task, noisy_task = _clean_and_noisy_tasks(seed=1)
    clean_state = _unregularized_pinv_delta_prediction(task, CONFIG)
    noisy_state = _unregularized_pinv_delta_prediction(noisy_task, CONFIG)
    clean_loss = held_out_generalization_loss(task, clean_state)
    noisy_loss = held_out_generalization_loss(task, noisy_state)
    assert noisy_loss > clean_loss * 20, (
        f"expected the UNREGULARIZED closed-form solve to collapse under label noise: {clean_loss} -> {noisy_loss}"
    )


def test_ridge_regularized_delta_prediction_is_robust_to_label_noise():
    """The fix, verified directly: with the default `ridge=1.0`, delta
    prediction's noisy-data held-out loss stays within the same order
    of magnitude as gradient descent's on the SAME noisy data --
    closing the ~135x gap the unregularized version had, while (checked
    in `test_ridge_regularized_delta_prediction_keeps_its_speed_advantage`
    below) retaining the closed-form method's real speed advantage."""
    task, noisy_task = _clean_and_noisy_tasks(seed=1)
    noisy_delta_state, _ = delta_prediction_update(noisy_task, init_fast_weights(CONFIG), CONFIG, ridge=1.0)
    noisy_gd_state, _ = gradient_descent_update(noisy_task, init_fast_weights(CONFIG), CONFIG, steps=400, lr=0.02)
    noisy_delta_loss = held_out_generalization_loss(task, noisy_delta_state)  # measured vs the CLEAN target
    noisy_gd_loss = held_out_generalization_loss(task, noisy_gd_state)
    assert noisy_delta_loss < noisy_gd_loss * 3, (
        f"expected ridge-regularized delta prediction to stay competitive with gradient descent under noise: "
        f"{noisy_delta_loss} vs {noisy_gd_loss}"
    )


def test_ridge_regularized_delta_prediction_keeps_its_speed_advantage():
    """The fix must not have quietly turned delta prediction into
    iterative optimization -- still one linear solve, still orders of
    magnitude faster than 400 gradient-descent steps."""
    task = make_task(CONFIG, seed=1, k_train=6, k_held_out=16, rule_scale=0.3)
    _, delta_diag = delta_prediction_update(task, init_fast_weights(CONFIG), CONFIG, ridge=1.0)
    _, gd_diag = gradient_descent_update(task, init_fast_weights(CONFIG), CONFIG, steps=400, lr=0.02)
    assert delta_diag["steps"] == 1
    assert delta_diag["wall_seconds"] < gd_diag["wall_seconds"] / 100


def test_gradient_descent_degrades_gracefully_under_label_noise():
    """The robustness half of the real finding: gradient descent's
    noisy-data held-out loss should be in the same order of magnitude as
    its clean-data loss, not blow up the way delta prediction's does."""
    task, noisy_task = _clean_and_noisy_tasks(seed=1)
    clean_state, _ = gradient_descent_update(task, init_fast_weights(CONFIG), CONFIG, steps=400, lr=0.02)
    noisy_state, _ = gradient_descent_update(noisy_task, init_fast_weights(CONFIG), CONFIG, steps=400, lr=0.02)
    clean_loss = held_out_generalization_loss(task, clean_state)
    noisy_loss = held_out_generalization_loss(task, noisy_state)
    assert noisy_loss < clean_loss * 5, f"gradient descent degraded too sharply under noise: {clean_loss} -> {noisy_loss}"
