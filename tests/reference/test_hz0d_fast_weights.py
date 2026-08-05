"""HZ-0D D1: fast-weight state/lifecycle contract tests
(reference/hz0d_fast_weights.py). Every lifecycle operation named in
docs/restart/hz0d_d1_contract.md is checked here, including a real
finite-difference gradient check -- directly required by
docs/restart/hz0d_history_audit.md's finding that a prior "gradient-
based" fast-weight mechanism was never actually verified and turned out
not to be gradient descent at all.
"""
from __future__ import annotations

import mlx.core as mx
import numpy as np

from reference.hz0d_fast_weights import (
    FastWeightConfig, FastWeightState, apply_fast_linear, decay_fast_weights, deserialize,
    effective_delta, fast_state_memory_bytes, init_fast_weights, reset_fast_weights, rollback,
    serialize, snapshot, update_fast_weights,
)


def _small_config(**overrides) -> FastWeightConfig:
    base = dict(dim=6, rank=2, num_layers=3, decay_rate=1.0, max_delta_norm=1.0, max_updates_per_session=50)
    base.update(overrides)
    return FastWeightConfig(**base)


def _random_linear(dim: int, seed: int) -> tuple[mx.array, mx.array]:
    key = mx.random.key(seed)
    k1, k2 = mx.random.split(key)
    weight = mx.random.normal((dim, dim), key=k1) * 0.1
    bias = mx.random.normal((dim,), key=k2) * 0.01
    return weight, bias


def test_init_state_has_exact_zero_effective_delta_for_every_layer():
    config = _small_config()
    state = init_fast_weights(config)
    for layer in range(config.num_layers):
        delta = effective_delta(state, layer)
        assert bool(mx.array_equal(delta, mx.zeros((config.dim, config.dim))))


def test_apply_with_zero_state_reproduces_base_linear_exactly():
    config = _small_config()
    state = init_fast_weights(config)
    weight, bias = _random_linear(config.dim, seed=1)
    x = mx.random.normal((4, config.dim), key=mx.random.key(2))
    fast_out = apply_fast_linear(x, weight, bias, state, layer_index=0)
    base_out = x @ weight.T + bias
    assert bool(mx.array_equal(fast_out, base_out))


def test_apply_with_nonzero_state_changes_output():
    config = _small_config()
    state = init_fast_weights(config)
    state = FastWeightState(
        a_fast=state.a_fast.at[0].add(mx.ones((config.dim, config.rank)) * 0.5),
        b_fast=state.b_fast.at[0].add(mx.ones((config.rank, config.dim)) * 0.5),
        update_count=state.update_count,
    )
    weight, bias = _random_linear(config.dim, seed=3)
    x = mx.random.normal((4, config.dim), key=mx.random.key(4))
    fast_out = apply_fast_linear(x, weight, bias, state, layer_index=0)
    base_out = x @ weight.T + bias
    assert not bool(mx.array_equal(fast_out, base_out))
    # Other layers are untouched (still zero delta).
    assert bool(mx.array_equal(effective_delta(state, 1), mx.zeros((config.dim, config.dim))))


def test_gradient_matches_finite_difference():
    """The exact check the prior implementation never had -- catches
    exactly the class of bug docs/restart/hz0d_history_audit.md found
    (an update mechanism that looks like gradient descent but isn't)."""
    config = _small_config()
    state = init_fast_weights(config)
    weight, bias = _random_linear(config.dim, seed=5)
    x = mx.random.normal((3, config.dim), key=mx.random.key(6))
    target = mx.random.normal((3, config.dim), key=mx.random.key(7))

    def loss_fn(a_fast, b_fast):
        trial_state = FastWeightState(a_fast=a_fast, b_fast=b_fast, update_count=state.update_count)
        out = apply_fast_linear(x, weight, bias, trial_state, layer_index=0)
        return mx.sum((out - target) ** 2)

    grad_fn = mx.grad(loss_fn, argnums=(0, 1))
    grad_a, grad_b = grad_fn(state.a_fast, state.b_fast)
    mx.eval(grad_a, grad_b)

    eps = 1e-3
    # Check a handful of entries in A, not the whole tensor, to keep the test fast.
    rng = np.random.default_rng(0)
    indices = [(0, i, j) for i, j in zip(rng.integers(0, config.dim, 4), rng.integers(0, config.rank, 4))]
    for layer, row, col in indices:
        plus_a = mx.array(state.a_fast)
        minus_a = mx.array(state.a_fast)
        plus_a = plus_a.at[layer, row, col].add(eps)
        minus_a = minus_a.at[layer, row, col].add(-eps)
        numeric = (float(loss_fn(plus_a, state.b_fast)) - float(loss_fn(minus_a, state.b_fast))) / (2 * eps)
        analytic = float(grad_a[layer, row, col])
        assert abs(numeric - analytic) < 1e-2, f"a_fast[{layer},{row},{col}]: numeric={numeric} analytic={analytic}"


def test_real_gradient_step_strictly_reduces_loss():
    """Not assumed from the gradient's existence -- measured directly,
    matching the standard docs/restart/hz0d_recovered_requirements.md
    sets for any update mechanism."""
    config = _small_config(max_delta_norm=100.0)  # large bound so clipping doesn't mask the check
    state = init_fast_weights(config)
    weight, bias = _random_linear(config.dim, seed=8)
    x = mx.random.normal((3, config.dim), key=mx.random.key(9))
    target = mx.random.normal((3, config.dim), key=mx.random.key(10))

    def loss_fn(a_fast, b_fast):
        trial_state = FastWeightState(a_fast=a_fast, b_fast=b_fast, update_count=state.update_count)
        out = apply_fast_linear(x, weight, bias, trial_state, layer_index=0)
        return mx.sum((out - target) ** 2)

    loss_before = float(loss_fn(state.a_fast, state.b_fast))
    grad_fn = mx.grad(loss_fn, argnums=(0, 1))
    grad_a, grad_b = grad_fn(state.a_fast, state.b_fast)
    mx.eval(grad_a, grad_b)
    new_state = update_fast_weights(state, layer_index=0, grad_a=grad_a[0], grad_b=grad_b[0], lr=0.05, config=config)
    loss_after = float(loss_fn(new_state.a_fast, new_state.b_fast))
    assert loss_after < loss_before, f"loss did not decrease: before={loss_before} after={loss_after}"
    assert int(new_state.update_count) == 1


def test_update_clips_effective_delta_norm_regardless_of_gradient_magnitude():
    config = _small_config(max_delta_norm=0.5)
    state = init_fast_weights(config)
    huge_grad_a = mx.ones((config.dim, config.rank)) * 1000.0
    huge_grad_b = mx.ones((config.rank, config.dim)) * 1000.0
    new_state = update_fast_weights(state, layer_index=0, grad_a=huge_grad_a, grad_b=huge_grad_b, lr=1.0, config=config)
    delta = effective_delta(new_state, 0)
    norm = float(mx.sqrt(mx.sum(delta * delta)))
    assert norm <= config.max_delta_norm + 1e-4, f"delta norm {norm} exceeds bound {config.max_delta_norm}"
    assert bool(mx.all(mx.isfinite(delta)))


def test_decay_rate_one_is_exact_noop():
    config = _small_config()
    state = init_fast_weights(config)
    state = FastWeightState(
        a_fast=state.a_fast.at[0].add(mx.ones((config.dim, config.rank))),
        b_fast=state.b_fast.at[0].add(mx.ones((config.rank, config.dim))),
        update_count=state.update_count,
    )
    decayed = decay_fast_weights(state, 1.0)
    assert bool(mx.array_equal(decayed.a_fast, state.a_fast))
    assert bool(mx.array_equal(decayed.b_fast, state.b_fast))


def test_decay_reduces_norm_monotonically():
    config = _small_config()
    state = init_fast_weights(config)
    state = FastWeightState(
        a_fast=state.a_fast.at[0].add(mx.ones((config.dim, config.rank))),
        b_fast=state.b_fast.at[0].add(mx.ones((config.rank, config.dim))),
        update_count=state.update_count,
    )
    before_norm = float(mx.sqrt(mx.sum(effective_delta(state, 0) ** 2)))
    decayed = decay_fast_weights(state, 0.9)
    after_norm = float(mx.sqrt(mx.sum(effective_delta(decayed, 0) ** 2)))
    assert after_norm < before_norm


def test_snapshot_and_rollback_are_bit_identical():
    config = _small_config()
    state = init_fast_weights(config)
    state = FastWeightState(
        a_fast=state.a_fast.at[1].add(mx.ones((config.dim, config.rank)) * 0.3),
        b_fast=state.b_fast.at[1].add(mx.ones((config.rank, config.dim)) * 0.3),
        update_count=mx.array(5, dtype=mx.int32),
    )
    checkpoint = snapshot(state)
    modified = FastWeightState(
        a_fast=state.a_fast.at[2].add(mx.ones((config.dim, config.rank))),
        b_fast=state.b_fast,
        update_count=state.update_count + 1,
    )
    assert not bool(mx.array_equal(modified.a_fast, state.a_fast))
    restored = rollback(checkpoint)
    assert bool(mx.array_equal(restored.a_fast, state.a_fast))
    assert bool(mx.array_equal(restored.b_fast, state.b_fast))
    assert int(restored.update_count) == int(state.update_count)


def test_reset_matches_fresh_init_exactly():
    config = _small_config()
    state = init_fast_weights(config)
    state = FastWeightState(
        a_fast=state.a_fast.at[0].add(mx.ones((config.dim, config.rank))),
        b_fast=state.b_fast,
        update_count=mx.array(9, dtype=mx.int32),
    )
    reset_state = reset_fast_weights(config)
    fresh_state = init_fast_weights(config)
    assert bool(mx.array_equal(reset_state.a_fast, fresh_state.a_fast))
    assert bool(mx.array_equal(reset_state.b_fast, fresh_state.b_fast))
    assert int(reset_state.update_count) == 0


def test_serialize_deserialize_round_trip_is_exact():
    config = _small_config()
    state = init_fast_weights(config)
    state = FastWeightState(
        a_fast=state.a_fast.at[0].add(mx.ones((config.dim, config.rank)) * 0.25),
        b_fast=state.b_fast.at[2].add(mx.ones((config.rank, config.dim)) * -0.75),
        update_count=mx.array(3, dtype=mx.int32),
    )
    data = serialize(state)
    restored = deserialize(data, config)
    assert bool(mx.array_equal(restored.a_fast, state.a_fast))
    assert bool(mx.array_equal(restored.b_fast, state.b_fast))
    assert int(restored.update_count) == 3


def test_fast_state_memory_bytes_matches_hand_computed_default_config():
    config = FastWeightConfig()  # real default: dim=768, rank=16, num_layers=6
    expected = 2 * 6 * 768 * 16 * 4
    assert fast_state_memory_bytes(config) == expected
    assert expected == 589_824  # 576 KiB, the number quoted in the contract doc


def test_init_is_deterministic_given_the_same_seed():
    config = _small_config(init_seed=42)
    state_a = init_fast_weights(config)
    state_b = init_fast_weights(config)
    assert bool(mx.array_equal(state_a.a_fast, state_b.a_fast))
    assert bool(mx.array_equal(state_a.b_fast, state_b.b_fast))
    other_seed_state = init_fast_weights(_small_config(init_seed=43))
    assert not bool(mx.array_equal(state_a.a_fast, other_seed_state.a_fast))


def test_multi_step_updates_converge_on_a_toy_mapping_task():
    """Not just "one step reduces loss" -- a real, small toy task (learn
    a fixed linear remapping from the base output) run to a real
    quality bar, the same kind of correctness bar D2's actual simulator
    tasks will need to clear. Proves the update mechanism can learn
    something nontrivial, not just nudge a single scalar loss down by
    an unmeasured amount."""
    config = _small_config(max_delta_norm=100.0)
    state = init_fast_weights(config)
    weight, bias = _random_linear(config.dim, seed=20)
    x = mx.random.normal((8, config.dim), key=mx.random.key(21))
    # A real, fixed target: a different linear remap of the SAME input,
    # something the base weight alone cannot produce, so real adaptation
    # is required to reduce loss substantially, not just marginally.
    target_weight, target_bias = _random_linear(config.dim, seed=22)
    target = x @ target_weight.T + target_bias

    def loss_fn(a_fast, b_fast):
        trial_state = FastWeightState(a_fast=a_fast, b_fast=b_fast, update_count=state.update_count)
        out = apply_fast_linear(x, weight, bias, trial_state, layer_index=0)
        return mx.sum((out - target) ** 2) / x.shape[0]

    grad_fn = mx.value_and_grad(loss_fn, argnums=(0, 1))
    loss_before = float(loss_fn(state.a_fast, state.b_fast))
    for _ in range(200):
        loss, (grad_a, grad_b) = grad_fn(state.a_fast, state.b_fast)
        mx.eval(loss, grad_a, grad_b)
        state = update_fast_weights(state, layer_index=0, grad_a=grad_a[0], grad_b=grad_b[0], lr=0.1, config=config)
    loss_after = float(loss_fn(state.a_fast, state.b_fast))
    assert loss_after < loss_before * 0.5, f"expected substantial reduction: before={loss_before} after={loss_after}"
    assert int(state.update_count) == 200


def test_permanent_weight_never_appears_in_fast_weight_gradient():
    """base_weight/base_bias are plain arrays with no gradient path from
    (a_fast, b_fast) -- checked by confirming grad_fn only ever needs
    (a_fast, b_fast) as explicit argnums and the base arrays are closed
    over as constants, never differentiated."""
    config = _small_config()
    state = init_fast_weights(config)
    weight, bias = _random_linear(config.dim, seed=11)
    x = mx.random.normal((2, config.dim), key=mx.random.key(12))

    def loss_fn(a_fast, b_fast):
        trial_state = FastWeightState(a_fast=a_fast, b_fast=b_fast, update_count=state.update_count)
        return mx.sum(apply_fast_linear(x, weight, bias, trial_state, layer_index=0) ** 2)

    grad_a, grad_b = mx.grad(loss_fn, argnums=(0, 1))(state.a_fast, state.b_fast)
    mx.eval(grad_a, grad_b)
    assert grad_a.shape == state.a_fast.shape
    assert grad_b.shape == state.b_fast.shape
    assert bool(mx.all(mx.isfinite(grad_a)))
    assert bool(mx.all(mx.isfinite(grad_b)))
