"""HZ-0D D3: update-mechanism comparison tests
(reference/hz0d_update_mechanisms.py). Locks in the real, measured
findings from `docs/restart/hz0d_d3_update_mechanism_results.md` as
regression tests -- not just "each method runs without error," but the
actual comparative claims the D3 exit gate rests on.
"""
from __future__ import annotations

import mlx.core as mx

from reference.hz0d_fast_weights import FastWeightConfig, FastWeightState, effective_delta, init_fast_weights
from reference.hz0d_isolated_simulator import make_rank_misspecified_task, make_task, held_out_generalization_loss
from reference.hz0d_update_mechanisms import (
    delta_prediction_update, error_conditioned_update, estimate_noise_ratio, gradient_descent_update,
    hebbian_delta_rule_update,
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
    """With the fix (ridge-regularized ALS -- see the function's own
    docstring for the v1/v2/v3 history), the fit is DELIBERATELY not an
    exact interpolator: it should still fit clean training data closely
    (small training loss), but the whole point of the ridge term is that
    it doesn't drive training loss to ~0, which is what made the
    original unregularized version collapse under label noise (see
    `test_unregularized_delta_prediction_collapses_under_label_noise`
    below)."""
    task = make_task(CONFIG, seed=7, k_train=6, k_held_out=16, rule_scale=0.3)
    _, diagnostics = delta_prediction_update(task, init_fast_weights(CONFIG), CONFIG)
    assert 1e-4 < diagnostics["final_train_loss"] < 0.1


def test_delta_prediction_base_ridge_trades_off_clean_fit_against_noise_robustness():
    """Direct confirmation that the (fixed part of the) ridge strength
    behaves as regularization should: more `base_ridge` -> worse
    clean-data fit, monotonically. `ridge_scale=0.0` isolates
    `base_ridge`'s own effect from the adaptive noise-based term."""
    task = make_task(CONFIG, seed=7, k_train=6, k_held_out=16, rule_scale=0.3)
    train_losses = []
    for base_ridge in [0.05, 0.27, 1.0]:
        _, diagnostics = delta_prediction_update(task, init_fast_weights(CONFIG), CONFIG, base_ridge=base_ridge, ridge_scale=0.0)
        train_losses.append(diagnostics["final_train_loss"])
    assert train_losses == sorted(train_losses), f"training loss should increase with more base_ridge: {train_losses}"


def test_noise_ratio_separates_clean_from_noisy_data():
    """The real, structural signal the adaptive ridge is built on:
    `estimate_noise_ratio` should be near-zero on clean data (the rule
    is exactly rank-`config.rank`, so almost no singular-value mass sits
    outside the top `config.rank` directions) and substantially larger
    on label-noise-corrupted data, across seeds -- not just one lucky
    draw."""
    for seed in range(5):
        task, noisy_task = _clean_and_noisy_tasks(seed=seed)
        clean_ratio = estimate_noise_ratio(task, CONFIG)
        noisy_ratio = estimate_noise_ratio(noisy_task, CONFIG)
        assert clean_ratio < 0.01, f"seed {seed}: expected a near-zero clean-data noise ratio, got {clean_ratio}"
        assert noisy_ratio > clean_ratio * 20, (
            f"seed {seed}: expected the noisy-data ratio to clearly separate from clean: {clean_ratio} vs {noisy_ratio}"
        )


def test_adaptive_ridge_delta_prediction_beats_gradient_descent_on_both_clean_and_noisy_data():
    """The specific ask this v4 fix was built for: not just "close
    enough," but actually BETTER than gradient descent, on BOTH clean
    and noisy held-out loss, on the same seed's data -- while delta
    prediction is still ~2 orders of magnitude faster (checked
    separately above). A single-seed check, tight (strict `<`, not a
    tolerance band), backed by the multi-seed means in the function's
    own docstring (7.4% better clean, 14.8% better noisy, across 8
    seeds) so this is not resting on one favorable draw."""
    task, noisy_task = _clean_and_noisy_tasks(seed=1)

    delta_clean_state, _ = delta_prediction_update(task, init_fast_weights(CONFIG), CONFIG)
    gd_clean_state, _ = gradient_descent_update(task, init_fast_weights(CONFIG), CONFIG, steps=400, lr=0.02)
    delta_clean_loss = held_out_generalization_loss(task, delta_clean_state)
    gd_clean_loss = held_out_generalization_loss(task, gd_clean_state)
    assert delta_clean_loss < gd_clean_loss, (
        f"expected adaptive-ridge ALS delta prediction to beat gradient descent on clean data: "
        f"{delta_clean_loss} vs {gd_clean_loss}"
    )

    delta_noisy_state, _ = delta_prediction_update(noisy_task, init_fast_weights(CONFIG), CONFIG)
    gd_noisy_state, _ = gradient_descent_update(noisy_task, init_fast_weights(CONFIG), CONFIG, steps=400, lr=0.02)
    delta_noisy_loss = held_out_generalization_loss(task, delta_noisy_state)  # vs the CLEAN target
    gd_noisy_loss = held_out_generalization_loss(task, gd_noisy_state)
    assert delta_noisy_loss < gd_noisy_loss, (
        f"expected adaptive-ridge ALS delta prediction to beat gradient descent on noisy data: "
        f"{delta_noisy_loss} vs {gd_noisy_loss}"
    )


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
    """The fix, verified directly: with the default (`ridge=0.27`,
    ALS), delta prediction's noisy-data held-out loss stays close to
    gradient descent's on the SAME noisy data -- closing the ~135x gap
    the original unregularized version had down to single-digit
    percent (see the function's own docstring for the full multi-seed
    numbers), while (checked in
    `test_ridge_regularized_delta_prediction_keeps_its_speed_advantage`
    below) retaining the closed-form method's real speed advantage."""
    task, noisy_task = _clean_and_noisy_tasks(seed=1)
    noisy_delta_state, _ = delta_prediction_update(noisy_task, init_fast_weights(CONFIG), CONFIG)
    noisy_gd_state, _ = gradient_descent_update(noisy_task, init_fast_weights(CONFIG), CONFIG, steps=400, lr=0.02)
    noisy_delta_loss = held_out_generalization_loss(task, noisy_delta_state)  # measured vs the CLEAN target
    noisy_gd_loss = held_out_generalization_loss(task, noisy_gd_state)
    assert noisy_delta_loss < noisy_gd_loss * 3, (
        f"expected ridge-regularized delta prediction to stay competitive with gradient descent under noise: "
        f"{noisy_delta_loss} vs {noisy_gd_loss}"
    )


def test_ridge_regularized_delta_prediction_keeps_its_speed_advantage():
    """The fix (now ALS, a small fixed number of closed-form solves)
    must not have quietly turned delta prediction into full iterative
    optimization -- still a small, bounded step count (`iters=15` by
    default, not hundreds), still orders of magnitude faster than 400
    gradient-descent steps."""
    task = make_task(CONFIG, seed=1, k_train=6, k_held_out=16, rule_scale=0.3)
    _, delta_diag = delta_prediction_update(task, init_fast_weights(CONFIG), CONFIG)
    _, gd_diag = gradient_descent_update(task, init_fast_weights(CONFIG), CONFIG, steps=400, lr=0.02)
    assert delta_diag["steps"] <= 20
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


def test_adaptive_ridge_delta_prediction_loses_to_gradient_descent_under_rank_misspecification():
    """The real, disclosed limitation found while investigating the "does
    `estimate_noise_ratio` generalize past this synthetic task's exact-
    rank guarantee?" caveat: it does NOT, once the true rule has
    substantial energy outside `config.rank`. `estimate_noise_ratio`
    cannot tell "spectral mass from label noise" apart from "spectral
    mass from a genuinely higher-rank rule" -- both inflate the same
    ratio, so v4 over-regularizes a target it could otherwise fit better,
    while gradient descent (which never assumes the rule is exactly
    rank-`config.rank`) does not carry this failure mode. Measured
    directly, 8 seeds, `excess_rank_scale=0.3` (the true rule's
    off-rank-2 component is 30% of the rank-2 component's own scale,
    still modest): mean held-out loss `delta=0.8236` vs `gd=0.6671` --
    gradient descent wins here, the mirror image of the label-noise
    case. A leave-one-out linear-predictability check was tried as a
    way to distinguish "noise" from "real excess-rank structure" (real
    structure should be predictable from `x`, noise should not) and
    found too high-variance to discriminate at `k_train=6` (R^2 ranged
    -0.74 to 0.57 for label noise and 0.00 to 0.85 for misspecification
    -- heavily overlapping), the same sample-size problem that already
    ruled out LOOCV/GCV for direct ridge selection. No fix is applied
    here: this is a genuine, disclosed boundary of v4's validity,
    locked in as a regression test so it is not lost, not silently
    patched over with an untested heuristic. Gradient descent staying
    fully implemented (not deleted) is the direct, real consequence."""
    d_losses, g_losses = [], []
    for seed in range(8):
        task = make_rank_misspecified_task(CONFIG, seed=seed, k_train=6, k_held_out=16, rule_scale=0.3, excess_rank_scale=0.3)
        delta_state, _ = delta_prediction_update(task, init_fast_weights(CONFIG), CONFIG)
        gd_state, _ = gradient_descent_update(task, init_fast_weights(CONFIG), CONFIG, steps=400, lr=0.02)
        d_losses.append(held_out_generalization_loss(task, delta_state))
        g_losses.append(held_out_generalization_loss(task, gd_state))
    mean_delta = sum(d_losses) / len(d_losses)
    mean_gd = sum(g_losses) / len(g_losses)
    assert mean_gd < mean_delta, (
        f"expected gradient descent to beat adaptive-ridge delta prediction under rank misspecification "
        f"(the known failure mode): gd={mean_gd} delta={mean_delta}"
    )


def test_rank_misspecified_task_with_zero_excess_is_exactly_rank_2():
    """`make_rank_misspecified_task(..., excess_rank_scale=0.0)` derives
    its keys differently from `make_task` (a 5-way `mx.random.split` vs
    `make_task`'s own 4-way split, so the two are NOT bit-identical even
    at the same seed -- an expected property of key-splitting, not a
    bug), so the real invariant to check is structural, not bitwise:
    with no excess term, `true_delta` must be EXACTLY rank-2 (matching
    `config.rank`), same as `make_task`'s own construction guarantees."""
    task = make_rank_misspecified_task(CONFIG, seed=3, k_train=6, k_held_out=16, rule_scale=0.3, excess_rank_scale=0.0)
    _, s, _ = mx.linalg.svd(task.true_delta, stream=mx.cpu)
    assert float(s[2]) < 1e-5, f"expected true_delta rank <= 2 at excess_rank_scale=0.0, got singular values {s.tolist()}"


def test_rank_misspecified_task_excess_scale_adds_real_off_rank_energy():
    """Sanity check on the stress-test helper itself: increasing
    `excess_rank_scale` should increase the singular-value mass beyond
    rank 2, monotonically -- confirming the helper actually produces
    tasks with genuine rank-misspecification, not a no-op parameter."""
    tail_masses = []
    for excess_rank_scale in [0.0, 0.2, 0.4]:
        task = make_rank_misspecified_task(CONFIG, seed=3, k_train=6, k_held_out=16, rule_scale=0.3, excess_rank_scale=excess_rank_scale)
        _, s, _ = mx.linalg.svd(task.true_delta, stream=mx.cpu)
        tail_masses.append(float(mx.sum(s[2:])))
    assert tail_masses == sorted(tail_masses), f"expected tail mass to grow monotonically with excess_rank_scale: {tail_masses}"
    assert tail_masses[0] < 1e-5
