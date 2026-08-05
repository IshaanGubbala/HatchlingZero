"""HZ-0D D3: update-mechanism comparison.

Per the plan's own D3 text: "Compare Hebbian updates, learned gradient-
like updates, low-rank delta prediction, and error-conditioned adapter
updates... Exit gate: one bounded method clearly beats simple
alternatives." Four real, distinct mechanisms, all operating on the SAME
`reference/hz0d_fast_weights.py` state/lifecycle contract and the SAME
`reference/hz0d_isolated_simulator.py` task, so the comparison in
`docs/restart/hz0d_d3_update_mechanism_results.md` is fair (identical
task, identical clipping/decay/state machinery -- only the update rule
differs).

Every mechanism returns `(new_state, diagnostics)`; `diagnostics` always
includes `wall_seconds` and `final_train_loss` so cost, not just
quality, is comparable -- per D0's own lesson (a plausible-sounding
mechanism was declared "production-ready" once already without a real,
apples-to-apples check against alternatives).
"""
from __future__ import annotations

import time

import mlx.core as mx

from reference.hz0d_fast_weights import FastWeightConfig, FastWeightState, clip_layer_factors, update_fast_weights
from reference.hz0d_isolated_simulator import LAYER, Task, task_loss


def gradient_descent_update(task: Task, state: FastWeightState, config: FastWeightConfig, *, steps: int, lr: float) -> tuple[FastWeightState, dict]:
    """"Learned gradient-like updates" -- real `mx.grad`-based joint
    optimization of BOTH `a_fast` and `b_fast`, the mechanism D1/D2
    already built and verified (finite-difference-checked, real few-shot
    generalization shown). The strong baseline every alternative here is
    compared against, not a strawman."""
    started = time.perf_counter()

    def loss_fn(a_fast, b_fast):
        trial = FastWeightState(a_fast=a_fast, b_fast=b_fast, update_count=state.update_count)
        return task_loss(task, trial, task.train_x, task.train_y)

    grad_fn = mx.value_and_grad(loss_fn, argnums=(0, 1))
    loss = None
    for _ in range(steps):
        loss, (grad_a, grad_b) = grad_fn(state.a_fast, state.b_fast)
        mx.eval(loss, grad_a, grad_b)
        state = update_fast_weights(state, LAYER, grad_a[LAYER], grad_b[LAYER], lr=lr, config=config)
    elapsed = time.perf_counter() - started
    return state, {"method": "gradient_descent", "wall_seconds": elapsed, "final_train_loss": float(loss), "steps": steps}


def hebbian_delta_rule_update(task: Task, state: FastWeightState, config: FastWeightConfig, *, passes: int, lr: float) -> tuple[FastWeightState, dict]:
    """"Hebbian updates" -- the classical error-corrective Hebbian rule
    (Widrow-Hoff/LMS: `delta_B[j, c] += signal[j] * x[c]`, an outer
    product of a local error-derived signal and the input, computed
    directly rather than via `mx.grad`). `a_fast` (the DECODER factor,
    randomly initialized by `init_fast_weights` -- see contract doc
    section 2's asymmetric init) is held fixed BY THIS RULE -- only
    `b_fast` (the ENCODER factor, zero-initialized) ever receives a
    Hebbian update, one training example at a time, online, for a small
    number of full passes over the training set. (Post-hoc clipping via
    `clip_layer_factors`, applied uniformly to every mechanism in this
    module for a fair comparison, CAN still rescale `a_fast` alongside
    `b_fast` if the realized delta exceeds the configured bound -- that
    is the shared safety mechanism acting, not this rule "changing its
    mind" about updating `a_fast`.) Cheaper than joint gradient descent
    (half the parameters ever change, no backprop machinery needed, a
    single local formula per example) -- and, unlike D1's ORIGINAL
    symmetric zero-init bug, well-posed from step one, since `a_fast`
    starting nonzero-random means the very first signal computed is
    nonzero."""
    started = time.perf_counter()
    a_layer = state.a_fast[LAYER]  # held fixed for the entire method -- never updated
    b_layer = state.b_fast[LAYER]
    k = task.train_x.shape[0]
    for _ in range(passes):
        for i in range(k):
            x_i = task.train_x[i:i + 1]      # [1, dim]
            target_i = task.train_y[i:i + 1]  # [1, dim]
            predicted = x_i @ (task.base_weight + a_layer @ b_layer).T + task.base_bias
            error = predicted - target_i      # [1, dim]
            signal = error @ a_layer          # [1, rank] -- dL/d(code), chain rule through the FIXED decoder A
            grad_b = signal.T @ x_i           # [rank, dim] outer product -- the classical Hebbian/LMS update term
            b_layer = b_layer - lr * grad_b
            mx.eval(b_layer)
    a_layer, b_layer = clip_layer_factors(a_layer, b_layer, config.max_delta_norm)
    new_state = FastWeightState(
        a_fast=state.a_fast.at[LAYER].add(a_layer - state.a_fast[LAYER]),
        b_fast=state.b_fast.at[LAYER].add(b_layer - state.b_fast[LAYER]),
        update_count=state.update_count + 1,
    )
    elapsed = time.perf_counter() - started
    final_loss = float(task_loss(task, new_state, task.train_x, task.train_y))
    return new_state, {"method": "hebbian_delta_rule", "wall_seconds": elapsed, "final_train_loss": final_loss, "steps": passes * k}


def delta_prediction_update(task: Task, state: FastWeightState, config: FastWeightConfig) -> tuple[FastWeightState, dict]:
    """"Low-rank delta prediction" -- no iterative optimization at all.
    Solves for the dense effective delta that best fits the training
    examples in ONE closed-form least-squares solve (`mx.linalg.pinv`),
    then truncates it to the configured rank via SVD
    (`mx.linalg.svd`) and reads `a_fast`/`b_fast` directly off the
    truncated factors. A real, single-shot "prediction" of the delta,
    not a search for one."""
    started = time.perf_counter()
    residual = task.train_y - (task.train_x @ task.base_weight.T + task.base_bias)  # [k, dim]
    # Solve train_x @ delta.T = residual for delta.T via the pseudo-inverse (real least squares, not an approximation).
    delta_t = mx.linalg.pinv(task.train_x, stream=mx.cpu) @ residual  # [dim, dim] == delta.T
    dense_delta = delta_t.T
    u, s, vt = mx.linalg.svd(dense_delta, stream=mx.cpu)
    r = config.rank
    sqrt_s = mx.sqrt(mx.clip(s[:r], 0.0, None))
    a_layer = u[:, :r] * sqrt_s[None, :]
    b_layer = sqrt_s[:, None] * vt[:r, :]
    a_layer, b_layer = clip_layer_factors(a_layer, b_layer, config.max_delta_norm)
    new_state = FastWeightState(
        a_fast=state.a_fast.at[LAYER].add(a_layer - state.a_fast[LAYER]),
        b_fast=state.b_fast.at[LAYER].add(b_layer - state.b_fast[LAYER]),
        update_count=state.update_count + 1,
    )
    elapsed = time.perf_counter() - started
    final_loss = float(task_loss(task, new_state, task.train_x, task.train_y))
    return new_state, {"method": "delta_prediction", "wall_seconds": elapsed, "final_train_loss": final_loss, "steps": 1}


def error_conditioned_update(task: Task, state: FastWeightState, config: FastWeightConfig, *, steps: int, base_lr: float, error_scale: float = 1.0) -> tuple[FastWeightState, dict]:
    """"Error-conditioned adapter updates" / "the controller predicts...
    how strongly to update" -- real joint gradient descent (same
    mechanism as `gradient_descent_update`), but the learning rate at
    each step is GATED by the current training error magnitude:
    `effective_lr = base_lr * tanh(error_norm / error_scale)`. Large
    error -> close to full `base_lr`; small (near-converged) error ->
    automatically small steps, without a separate hand-tuned decay
    schedule. This is the one candidate here with an explicit "whether/
    how strongly to update" gate, matching D3's own text most directly."""
    started = time.perf_counter()

    def loss_fn(a_fast, b_fast):
        trial = FastWeightState(a_fast=a_fast, b_fast=b_fast, update_count=state.update_count)
        return task_loss(task, trial, task.train_x, task.train_y)

    grad_fn = mx.value_and_grad(loss_fn, argnums=(0, 1))
    loss = None
    gates = []
    for _ in range(steps):
        loss, (grad_a, grad_b) = grad_fn(state.a_fast, state.b_fast)
        mx.eval(loss, grad_a, grad_b)
        error_norm = float(mx.sqrt(loss))
        gate = float(mx.tanh(mx.array(error_norm / error_scale)))
        gates.append(gate)
        state = update_fast_weights(state, LAYER, grad_a[LAYER], grad_b[LAYER], lr=base_lr * gate, config=config)
    elapsed = time.perf_counter() - started
    return state, {
        "method": "error_conditioned", "wall_seconds": elapsed, "final_train_loss": float(loss),
        "steps": steps, "mean_gate": sum(gates) / len(gates) if gates else 0.0,
    }
