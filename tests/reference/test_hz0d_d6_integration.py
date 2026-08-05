"""HZ-0D D6: frozen-backbone integration tests (reference/hz0d_d6_integration.py).

Checked against the ACTUAL frozen HZ-0A/HZ-0C checkpoint, not synthetic
hidden states, matching this project's established convention
(tests/reference/test_hz0c_c6_memory_integration.py). Skips if the
checkpoint isn't present locally (gitignored under outputs/). Locks in
D6's own exit gate directly: "inactive fast weights reproduce HZ-0C
behavior; active fast weights improve adaptation."
"""
from __future__ import annotations

import mlx.core as mx
import mlx.utils
import pytest

from reference.hz0d_d6_integration import ATTENTION_INDICES, d6_fast_weight_config, d6_forward_with_fast_weights
from reference.hz0d_fast_weights import FastWeightConfig, FastWeightState, init_fast_weights
from reference.hz0d_isolated_simulator import Task, task_loss, held_out_generalization_loss
from reference.hz0d_update_mechanisms import delta_prediction_update
from scripts.hz0b_b11_baseline_comparison import CHECKPOINT, load_frozen_model
from scripts.hz0c_c6_conditional_attention_eval import conditional_forward, fixed_matched_trigger

pytestmark = pytest.mark.skipif(
    not (CHECKPOINT / "state.json").exists(),
    reason="frozen HZ-0A checkpoint not present locally (gitignored under outputs/)",
)


def _real_layer_task(model, *, seed: int, k_train: int, k_held_out: int, rule_scale: float = 0.05, x_scale: float = 0.05) -> Task:
    """A few-shot low-rank-remapping task built on the REAL frozen
    output-projection weight/bias at the first real anchor layer
    (`model.blocks[ATTENTION_INDICES[0]].mixer.out`), not a synthetic
    random matrix -- matching D2's own task shape
    (`reference/hz0d_isolated_simulator.py::make_task`) but at the D1
    contract's real `dim=768`/`rank=16`."""
    layer = model.blocks[ATTENTION_INDICES[0]]
    real_w, real_b = layer.mixer.out.weight, layer.mixer.out.bias
    dim, rank = real_w.shape[0], 16
    key = mx.random.key(seed)
    k_a, k_b, k_sym = mx.random.split(key, 3)
    true_a = mx.random.normal((dim, rank), key=k_a) * rule_scale
    true_b = mx.random.normal((rank, dim), key=k_b) * rule_scale
    true_delta = true_a @ true_b
    total = k_train + k_held_out
    symbols = mx.random.normal((total, dim), key=k_sym) * x_scale
    targets = symbols @ (real_w + true_delta).T + real_b
    return Task(
        base_weight=real_w, base_bias=real_b, true_delta=true_delta,
        train_x=symbols[:k_train], train_y=targets[:k_train],
        held_out_x=symbols[k_train:], held_out_y=targets[k_train:],
    )


def test_inactive_fast_weights_reproduce_hz0c_conditional_forward_exactly():
    """D6's exit gate, half 1: with `init_fast_weights` (asymmetric
    zero-init, `b_fast=0` so the realized delta is exactly zero), the
    D6 forward pass must be BIT-IDENTICAL to HZ-0C's own real
    `conditional_forward` -- not approximately close, exactly equal
    (`mx.array_equal`), on the real checkpoint and a real trigger
    pattern."""
    model, _payload = load_frozen_model()
    config = d6_fast_weight_config()
    fast_state = init_fast_weights(config)
    tokens = mx.array([[1, 45, 982, 12, 7, 300, 44, 1023, 55, 66, 77, 88]])
    trigger = fixed_matched_trigger(1, tokens.shape[1], 0.15)

    real_logits = conditional_forward(model, tokens, trigger)
    d6_logits = d6_forward_with_fast_weights(model, tokens, trigger, fast_state, config)
    mx.eval(real_logits, d6_logits)

    assert real_logits.shape == d6_logits.shape
    assert bool(mx.array_equal(real_logits, d6_logits))


def test_d6_wiring_never_mutates_frozen_model_parameters():
    """D1's contract guarantee, checked at the real-model integration
    point: running a forward pass with a NONZERO (adapted) fast-weight
    state must not change any of `model`'s own parameters -- only
    `fast_state` carries the adaptation. Snapshot every real parameter
    before and after a forward pass with a genuinely nonzero fast
    state, compare bit-exactly."""
    model, _payload = load_frozen_model()
    config = d6_fast_weight_config()
    before = {k: mx.array(v) for k, v in mlx.utils.tree_flatten(model.parameters())}

    nonzero_state = FastWeightState(
        a_fast=mx.random.normal((config.num_layers, config.dim, config.rank), key=mx.random.key(1)) * 0.02,
        b_fast=mx.random.normal((config.num_layers, config.rank, config.dim), key=mx.random.key(2)) * 0.02,
        update_count=mx.array(1, dtype=mx.int32),
    )
    tokens = mx.array([[1, 45, 982, 12, 7, 300, 44, 1023]])
    trigger = fixed_matched_trigger(1, tokens.shape[1], 0.15)
    logits = d6_forward_with_fast_weights(model, tokens, trigger, nonzero_state, config)
    mx.eval(logits)

    after = {k: v for k, v in mlx.utils.tree_flatten(model.parameters())}
    assert before.keys() == after.keys()
    for key in before:
        assert bool(mx.array_equal(before[key], after[key])), f"parameter {key} changed after a D6 forward pass"


def test_active_fast_weights_reduce_held_out_loss_versus_inactive_at_real_scale():
    """D6's exit gate, half 2: real fast-weight adaptation (D3's
    selected `delta_prediction_update`) must measurably reduce held-out
    loss relative to the inactive (zero-delta) baseline, at the D1
    contract's REAL scale (`dim=768`, `rank=16`) using the REAL frozen
    output-projection weight as the task's base -- not just the
    isolated `dim=8` toy task D2/D3 already covered. Calibration sweep
    (not shown here, see `docs/restart/hz0d_d6_frozen_backbone_integration_results.md`)
    found ~32% mean held-out loss reduction at `k_train=256`, growing to
    ~99% by `k_train=1024`; this test uses `k_train=256` to keep runtime
    bounded, well below the calibrated result for safety margin."""
    model, _payload = load_frozen_model()
    single_layer_config = FastWeightConfig(dim=768, rank=16, num_layers=1, max_delta_norm=10.0)
    zero_losses, active_losses = [], []
    for seed in range(3):
        task = _real_layer_task(model, seed=seed, k_train=256, k_held_out=64)
        zero_state = FastWeightState(
            a_fast=mx.zeros((1, 768, 16)), b_fast=mx.zeros((1, 16, 768)), update_count=mx.array(0, dtype=mx.int32),
        )
        zero_losses.append(float(task_loss(task, zero_state, task.held_out_x, task.held_out_y)))
        active_state, _ = delta_prediction_update(task, init_fast_weights(single_layer_config), single_layer_config)
        active_losses.append(held_out_generalization_loss(task, active_state))

    mean_zero = sum(zero_losses) / len(zero_losses)
    mean_active = sum(active_losses) / len(active_losses)
    assert mean_active < mean_zero * 0.85, (
        f"expected active fast weights to reduce held-out loss by >=15% versus inactive at real scale: "
        f"zero={mean_zero} active={mean_active}"
    )


def test_d6_end_to_end_forward_with_adapted_state_differs_from_inactive_and_stays_finite():
    """A genuinely end-to-end check, not just the isolated linear-
    algebra path: plug a REAL adapted `FastWeightState` (fit via
    `delta_prediction_update` on the real-weight task above) into the
    full `d6_forward_with_fast_weights` real-model forward pass, and
    confirm the resulting logits (a) differ from the inactive baseline
    (adaptation actually reaches the real forward pass) and (b) stay
    finite (no NaN/Inf introduced by the wiring)."""
    model, _payload = load_frozen_model()
    config = d6_fast_weight_config()
    single_layer_config = FastWeightConfig(dim=768, rank=16, num_layers=1, max_delta_norm=10.0)
    task = _real_layer_task(model, seed=0, k_train=256, k_held_out=64)
    adapted_single_layer, _ = delta_prediction_update(task, init_fast_weights(single_layer_config), single_layer_config)

    full_state = FastWeightState(
        a_fast=mx.zeros((config.num_layers, config.dim, config.rank)).at[0].add(adapted_single_layer.a_fast[0]),
        b_fast=mx.zeros((config.num_layers, config.rank, config.dim)).at[0].add(adapted_single_layer.b_fast[0]),
        update_count=mx.array(1, dtype=mx.int32),
    )
    inactive_state = init_fast_weights(config)

    tokens = mx.array([[1, 45, 982, 12, 7, 300, 44, 1023, 55, 66, 77, 88]])
    trigger = fixed_matched_trigger(1, tokens.shape[1], 0.15)
    active_logits = d6_forward_with_fast_weights(model, tokens, trigger, full_state, config)
    inactive_logits = d6_forward_with_fast_weights(model, tokens, trigger, inactive_state, config)
    mx.eval(active_logits, inactive_logits)

    assert bool(mx.all(mx.isfinite(active_logits)))
    assert not bool(mx.array_equal(active_logits, inactive_logits))
