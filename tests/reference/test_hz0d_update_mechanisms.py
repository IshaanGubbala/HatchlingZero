"""HZ-0D D3: update-mechanism comparison tests
(reference/hz0d_update_mechanisms.py). Locks in the real, measured
findings from `docs/restart/hz0d_d3_update_mechanism_results.md` as
regression tests -- not just "each method runs without error," but the
actual comparative claims the D3 exit gate rests on.
"""
from __future__ import annotations

import mlx.core as mx

from reference.hz0d_fast_weights import FastWeightConfig, effective_delta, init_fast_weights
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


def test_delta_prediction_achieves_near_zero_training_loss_on_clean_data():
    """The closed-form least-squares solve should fit the (noise-free)
    training examples almost exactly -- the real, expected behavior of
    an exact interpolator, and the same property that makes it fragile
    under label noise (tested below)."""
    task = make_task(CONFIG, seed=7, k_train=6, k_held_out=16, rule_scale=0.3)
    _, diagnostics = delta_prediction_update(task, init_fast_weights(CONFIG), CONFIG)
    assert diagnostics["final_train_loss"] < 1e-2


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


def test_delta_prediction_wins_on_clean_data_but_collapses_under_label_noise():
    """The real, decisive D3 finding: the closed-form method is BEST on
    clean data (exact interpolation) but catastrophically worse than
    gradient descent once training labels carry real noise -- because it
    has no implicit regularization the way early-stopped iterative
    gradient descent does. This is why gradient descent, not delta
    prediction, is the method actually selected (see the results doc)."""
    task, noisy_task = _clean_and_noisy_tasks(seed=1)

    clean_delta_state, _ = delta_prediction_update(task, init_fast_weights(CONFIG), CONFIG)
    clean_gd_state, _ = gradient_descent_update(task, init_fast_weights(CONFIG), CONFIG, steps=400, lr=0.02)
    assert held_out_generalization_loss(task, clean_delta_state) < held_out_generalization_loss(task, clean_gd_state), (
        "expected delta prediction to win on clean data"
    )

    noisy_delta_state, _ = delta_prediction_update(noisy_task, init_fast_weights(CONFIG), CONFIG)
    noisy_gd_state, _ = gradient_descent_update(noisy_task, init_fast_weights(CONFIG), CONFIG, steps=400, lr=0.02)
    noisy_delta_loss = held_out_generalization_loss(task, noisy_delta_state)  # measured vs the CLEAN target
    noisy_gd_loss = held_out_generalization_loss(task, noisy_gd_state)
    assert noisy_delta_loss > noisy_gd_loss * 10, (
        f"expected delta prediction to collapse under label noise relative to gradient descent: "
        f"{noisy_delta_loss} vs {noisy_gd_loss}"
    )


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
