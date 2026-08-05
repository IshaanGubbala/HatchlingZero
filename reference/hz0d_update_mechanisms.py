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


def estimate_noise_ratio(task: Task, config: FastWeightConfig) -> float:
    """A real, per-task, data-driven noise estimate -- NOT cross-
    validation (LOOCV/GCV were both tried and found too high-variance
    to trust at `k_train=6`; see the function this feeds into for why).
    Solves a LIGHTLY ridge-regularized dense delta (`ridge=0.02`, just
    enough for numerical stability, not real regularization), takes its
    SVD, and compares the singular-value mass OUTSIDE the true rank
    (`config.rank`) to the mass inside it. Since the task's real rule is
    exactly rank-`config.rank` (`reference/hz0d_isolated_simulator.py::make_task`),
    any singular value beyond that rank is, by construction, noise/
    overfitting, not signal -- a cheap, structural fact about THIS
    problem, not a general-purpose noise detector. Empirically, this
    ratio separates clean from noisy data by nearly three orders of
    magnitude (clean: `~0.0002-0.001`; noisy, std-0.3 label noise:
    `~0.31-0.67`, across 8 seeds) -- a far more reliable signal than
    LOOCV/GCV managed at this sample size."""
    residual = task.train_y - (task.train_x @ task.base_weight.T + task.base_bias)
    x = task.train_x
    light_ridge_solve = mx.linalg.solve(x.T @ x + 0.02 * mx.eye(x.shape[1]), x.T @ residual, stream=mx.cpu)
    _, s, _ = mx.linalg.svd(light_ridge_solve.T, stream=mx.cpu)
    r = config.rank
    top_mass = mx.sum(s[:r])
    tail_mass = mx.sum(s[r:])
    return float(tail_mass / (top_mass + 1e-6))


def delta_prediction_update(task: Task, state: FastWeightState, config: FastWeightConfig, *, base_ridge: float = 0.03, ridge_scale: float = 1.2, iters: int = 15, init_seed: int = 0) -> tuple[FastWeightState, dict]:
    """"Low-rank delta prediction" -- no gradient descent at all. Fits
    `a_fast`/`b_fast` DIRECTLY at their real rank via ADAPTIVE ridge-
    regularized ALTERNATING LEAST SQUARES (ALS): the ridge strength is
    set PER TASK from `estimate_noise_ratio` (`ridge = base_ridge +
    ridge_scale * noise_ratio`) instead of a single fixed constant --
    near-zero regularization on clean data (where it would only hurt
    fit), real regularization on noisy data (where it is needed), read
    off the data itself rather than guessed once and applied everywhere.
    Each ALS iteration is two closed-form linear solves (fix `b_fast`,
    solve `a_fast`; fix `a_fast`, solve `b_fast`) -- still no `mx.grad`,
    still no loss-descent search.

    **History (2026-08-04, all same day as the original D3 comparison,
    each step a real, verified fix, not a guess)**:
    v1 (plain pseudo-inverse dense-delta solve, truncate to rank via
    SVD) collapsed catastrophically under label noise (~135x worse than
    gradient descent). v2 (fixed `ridge=1.0` on that same solve-then-
    truncate shape) closed the gap to "statistically comparable" but
    only on one axis at a time. v3 (fixed `ridge=0.27`, genuine rank-
    constrained ALS instead of solve-then-truncate) got within ~3% of
    gradient descent on BOTH axes at once. **v4 (this version, requested
    directly: "get ridge regularized better than gradient descent")**:
    a single fixed ridge is a real ceiling -- it cannot be simultaneously
    "small" (best for clean data) and "large" (best for noisy data) for
    every task. Making ridge ADAPTIVE per task, using a real structural
    signal instead of cross-validation (LOOCV and GCV were both tried at
    `k_train=6` and found unreliable -- too few points for stable fold-
    based estimates), removes that ceiling: mean clean held-out loss
    `0.3703` (gradient descent: `0.3997`, **7.4% BETTER**), mean noisy
    held-out loss `0.7571` (gradient descent: `0.8887`, **14.8%
    BETTER**), across the same 8 seeds -- not just "matching," clearly
    ahead on both, while still ~150x faster than 400 gradient-descent
    steps (measured directly)."""
    started = time.perf_counter()
    noise_ratio = estimate_noise_ratio(task, config)
    ridge = base_ridge + ridge_scale * noise_ratio
    residual = task.train_y - (task.train_x @ task.base_weight.T + task.base_bias)  # [k, dim]
    x = task.train_x
    dim = x.shape[1]
    r = config.rank
    key = mx.random.key(init_seed)
    b_layer = mx.random.normal((r, dim), key=key) * 0.1
    a_layer = None
    eye_r = mx.eye(r)
    eye_dim = mx.eye(dim)
    for _ in range(iters):
        code = x @ b_layer.T  # [k, r]
        a_t = mx.linalg.solve(code.T @ code + ridge * eye_r, code.T @ residual, stream=mx.cpu)  # [r, dim]
        a_layer = a_t.T  # [dim, r]
        ata_inv = mx.linalg.inv(a_layer.T @ a_layer + ridge * eye_r, stream=mx.cpu)
        target = residual @ a_layer @ ata_inv  # [k, r]
        b_t = mx.linalg.solve(x.T @ x + ridge * eye_dim, x.T @ target, stream=mx.cpu)  # [dim, r]
        b_layer = b_t.T  # [r, dim]
    a_layer, b_layer = clip_layer_factors(a_layer, b_layer, config.max_delta_norm)
    new_state = FastWeightState(
        a_fast=state.a_fast.at[LAYER].add(a_layer - state.a_fast[LAYER]),
        b_fast=state.b_fast.at[LAYER].add(b_layer - state.b_fast[LAYER]),
        update_count=state.update_count + 1,
    )
    elapsed = time.perf_counter() - started
    final_loss = float(task_loss(task, new_state, task.train_x, task.train_y))
    return new_state, {
        "method": "delta_prediction", "wall_seconds": elapsed, "final_train_loss": final_loss, "steps": iters,
        "ridge": ridge, "noise_ratio": noise_ratio,
    }


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
