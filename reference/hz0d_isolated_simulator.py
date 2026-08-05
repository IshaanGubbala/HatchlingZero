"""HZ-0D D2: isolated fast-weight simulator.

Per `plans/HZ-0D_Fast_Weights_Total_Restart_Plan.md`'s D2 text: "Test
rapid mapping adaptation, symbol remapping, ... contradictory rule
changes, decay, snapshot/rollback, reset, noisy updates, and malicious
updates. Measure adaptation speed, interference, state norms, rollback
fidelity, and reset fidelity." Exit gate: "temporary mappings work and
prior state is restored exactly."

Deliberately isolated (small synthetic dims, no frozen HZ-0A backbone)
-- matching this project's own established two-phase discipline (HZ-0B's
B2 simulator was synthetic before B6's real-model integration; HZ-0C's
C2/C3 were isolated before C6). D6 ("frozen-backbone integration") is
where this connects to the real model; not here.

The "symbol remapping" / "temporary rule" task: a FIXED, TRUE low-rank
delta (`true_a @ true_b`, rank <= the fast-weight rank so it is exactly
representable) defines a temporary target mapping
`target(x) = x @ (base_weight + true_delta).T + base_bias` for a set of
fixed "symbol" vectors. A few symbols are shown as training examples;
the REST are held out -- so "learned the rule" (generalizes to unseen
symbols under the SAME temporary rule) is distinguished from "memorized
the training examples" (a real, well-posed few-shot generalization
test, not just a training-loss check).
"""
from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

from reference.hz0d_fast_weights import (
    FastWeightConfig, FastWeightState, apply_fast_linear, update_fast_weights,
)

LAYER = 0  # every task here uses a single-layer FastWeightConfig(num_layers=1)


@dataclass(frozen=True)
class Task:
    """A fixed temporary rule plus a train/held-out symbol split, all
    deterministic given a seed."""

    base_weight: mx.array   # [dim, dim], the frozen "pretrained" layer
    base_bias: mx.array     # [dim]
    true_delta: mx.array    # [dim, dim], the temporary rule's realized effect
    train_x: mx.array       # [k_train, dim]
    train_y: mx.array       # [k_train, dim] -- target(train_x) under true_delta
    held_out_x: mx.array    # [k_held_out, dim]
    held_out_y: mx.array    # [k_held_out, dim]


def _random_linear(dim: int, key: mx.array, scale: float = 0.1) -> tuple[mx.array, mx.array]:
    k1, k2 = mx.random.split(key)
    return mx.random.normal((dim, dim), key=k1) * scale, mx.random.normal((dim,), key=k2) * (scale * 0.1)


def make_task(config: FastWeightConfig, *, seed: int, k_train: int = 4, k_held_out: int = 8, rule_scale: float = 0.3) -> Task:
    """Builds one temporary-rule task. `rule_scale` controls how large
    the true delta is relative to the base weight -- large enough that
    the base weight ALONE cannot produce the target (so real adaptation
    is required, not a no-op), small enough to stay within
    `FastWeightConfig.max_delta_norm` once learned."""
    key = mx.random.key(seed)
    k_base, k_rule_a, k_rule_b, k_symbols = mx.random.split(key, 4)
    base_weight, base_bias = _random_linear(config.dim, k_base)
    true_a = mx.random.normal((config.dim, config.rank), key=k_rule_a) * rule_scale
    true_b = mx.random.normal((config.rank, config.dim), key=k_rule_b) * rule_scale
    true_delta = true_a @ true_b

    total_symbols = k_train + k_held_out
    symbols = mx.random.normal((total_symbols, config.dim), key=k_symbols)
    targets = symbols @ (base_weight + true_delta).T + base_bias
    return Task(
        base_weight=base_weight, base_bias=base_bias, true_delta=true_delta,
        train_x=symbols[:k_train], train_y=targets[:k_train],
        held_out_x=symbols[k_train:], held_out_y=targets[k_train:],
    )


def make_rank_misspecified_task(config: FastWeightConfig, *, seed: int, k_train: int = 6, k_held_out: int = 16, rule_scale: float = 0.3, excess_rank_scale: float = 0.0) -> Task:
    """Same construction as `make_task`, plus a small FULL-RANK perturbation
    added to `true_delta` (magnitude `excess_rank_scale * rule_scale`) --
    a task whose true rule is NOT exactly rank-`config.rank`, unlike every
    task `make_task` can produce. Exists to stress-test mechanisms (e.g.
    `reference/hz0d_update_mechanisms.py::estimate_noise_ratio`) that
    assume exact rank; `excess_rank_scale=0.0` reproduces `make_task`
    exactly, so it is not used for D2's own exit-gate evidence, only for
    D3's misspecification check (`docs/restart/hz0d_d3_update_mechanism_results.md`)."""
    key = mx.random.key(seed)
    k_base, k_rule_a, k_rule_b, k_excess, k_symbols = mx.random.split(key, 5)
    base_weight, base_bias = _random_linear(config.dim, k_base)
    true_a = mx.random.normal((config.dim, config.rank), key=k_rule_a) * rule_scale
    true_b = mx.random.normal((config.rank, config.dim), key=k_rule_b) * rule_scale
    true_delta = true_a @ true_b
    if excess_rank_scale > 0.0:
        excess = mx.random.normal((config.dim, config.dim), key=k_excess) * (rule_scale * excess_rank_scale)
        true_delta = true_delta + excess

    total_symbols = k_train + k_held_out
    symbols = mx.random.normal((total_symbols, config.dim), key=k_symbols)
    targets = symbols @ (base_weight + true_delta).T + base_bias
    return Task(
        base_weight=base_weight, base_bias=base_bias, true_delta=true_delta,
        train_x=symbols[:k_train], train_y=targets[:k_train],
        held_out_x=symbols[k_train:], held_out_y=targets[k_train:],
    )


def task_loss(task: Task, state: FastWeightState, x: mx.array, y: mx.array) -> mx.array:
    predicted = apply_fast_linear(x, task.base_weight, task.base_bias, state, LAYER)
    return mx.mean(mx.sum((predicted - y) ** 2, axis=-1))


def adapt_to_task(task: Task, state: FastWeightState, config: FastWeightConfig, *, steps: int, lr: float) -> tuple[FastWeightState, list[float]]:
    """Runs `steps` real gradient-descent updates on the TRAINING split
    only (`task.train_x`/`train_y`) -- held-out performance is never
    used to guide adaptation, only to measure generalization afterward.
    Returns the final state and the per-step training-loss trajectory
    (adaptation-speed evidence)."""
    def loss_fn(a_fast, b_fast):
        trial = FastWeightState(a_fast=a_fast, b_fast=b_fast, update_count=state.update_count)
        return task_loss(task, trial, task.train_x, task.train_y)

    grad_fn = mx.value_and_grad(loss_fn, argnums=(0, 1))
    history = []
    for _ in range(steps):
        loss, (grad_a, grad_b) = grad_fn(state.a_fast, state.b_fast)
        mx.eval(loss, grad_a, grad_b)
        history.append(float(loss))
        state = update_fast_weights(state, LAYER, grad_a[LAYER], grad_b[LAYER], lr=lr, config=config)
    return state, history


def held_out_generalization_loss(task: Task, state: FastWeightState) -> float:
    return float(task_loss(task, state, task.held_out_x, task.held_out_y))


def state_delta_norm(state: FastWeightState) -> float:
    delta = state.a_fast[LAYER] @ state.b_fast[LAYER]
    return float(mx.sqrt(mx.sum(delta * delta)))
