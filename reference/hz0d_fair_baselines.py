"""HZ-0D D4: fair adaptation baselines.

Per the plan's own D4 text: "Compare with no adaptation, ordinary
in-context learning, longer context, HZ-0B memory only, HZ-0C only,
retrieval, static adapters, gradient-updated adapters, and permanent
LoRA. Exit gate: gains are attributable to temporary fast adaptation."

Still isolated (per `plans/HZ-0D_Fast_Weights_Total_Restart_Plan.md`'s
own D4/D5 text: "D4 (still isolated) may proceed regardless" of the
HZ-0C dependency gate) -- every baseline here runs on the SAME
`reference/hz0d_isolated_simulator.py` few-shot symbol-remapping task
D2/D3 already used, not the real HZ-0C model.

Two of the plan's named baselines -- "HZ-0B memory only" (associative
read/write memory) and "HZ-0C only" (surprise-triggered anchor
attention) -- have no faithful analog in this synthetic per-example
regression task: both are mechanisms over a TOKEN SEQUENCE with real
temporal/surprise structure, and this task has neither (each example is
an independent (x, y) pair, not a sequence position). Substituting the
mechanically closest generic analog for each, and saying so plainly
rather than pretending it is the real thing:
- "HZ-0B memory only" -> `knn_retrieval_baseline` (explicit key-value
  lookup over stored (x, y) pairs is what an associative memory read
  IS, mechanically, once there is no sequence to condition retrieval
  on).
- "HZ-0C only" -> `in_context_attention_baseline` (soft attention over
  stored context is what anchor attention IS, mechanically, once there
  is no surprise signal to gate it).
Real HZ-0B/HZ-0C-specific baselines belong at D6, where the real
backbone exists.
"""
from __future__ import annotations

import time

import mlx.core as mx

from reference.hz0d_fast_weights import FastWeightConfig, FastWeightState, apply_fast_linear, update_fast_weights
from reference.hz0d_isolated_simulator import LAYER, Task, task_loss


def no_adaptation_baseline(task: Task, config: FastWeightConfig) -> tuple[float, dict]:
    """Zero delta -- the frozen base model alone, no context, no
    weights changed. The floor every other baseline must beat to be
    worth anything at all."""
    started = time.perf_counter()
    zero_state = FastWeightState(
        a_fast=mx.zeros((config.num_layers, config.dim, config.rank)),
        b_fast=mx.zeros((config.num_layers, config.rank, config.dim)),
        update_count=mx.array(0, dtype=mx.int32),
    )
    loss = float(task_loss(task, zero_state, task.held_out_x, task.held_out_y))
    return loss, {"method": "no_adaptation", "wall_seconds": time.perf_counter() - started}


def _kernel_regression_held_out_loss(task: Task, context_x: mx.array, context_y: mx.array, *, temperature: float) -> float:
    """Nadaraya-Watson kernel regression on the RESIDUAL
    (`context_y - base_pred(context_x)`), evaluated at every held-out
    point: `predicted(x) = base_pred(x) + sum_i softmax_i(-||x -
    context_x_i||^2 / temperature) * residual_i`. No weight ever
    changes -- this is a frozen model attending over its context, the
    real mechanical content of "ordinary in-context learning" (and the
    substitute used here for "HZ-0C only", see module docstring)."""
    base_context_pred = context_x @ task.base_weight.T + task.base_bias
    residual = context_y - base_context_pred  # [k, dim]
    base_held_out_pred = task.held_out_x @ task.base_weight.T + task.base_bias

    diffs = task.held_out_x[:, None, :] - context_x[None, :, :]  # [k_held_out, k, dim]
    sq_dists = mx.sum(diffs * diffs, axis=-1)  # [k_held_out, k]
    weights = mx.softmax(-sq_dists / temperature, axis=-1)  # [k_held_out, k]
    predicted_residual = weights @ residual  # [k_held_out, dim]
    predicted = base_held_out_pred + predicted_residual
    return float(mx.mean(mx.sum((predicted - task.held_out_y) ** 2, axis=-1)))


def _median_heuristic_temperature(x: mx.array) -> float:
    """Data-driven kernel bandwidth from `x` alone (no labels, no
    held-out data -- no leakage): the median squared pairwise distance
    among the context points, standard practice for kernel-method
    bandwidth selection."""
    diffs = x[:, None, :] - x[None, :, :]
    sq_dists = mx.sum(diffs * diffs, axis=-1)
    k = x.shape[0]
    off_diag = [float(sq_dists[i, j]) for i in range(k) for j in range(k) if i != j]
    off_diag.sort()
    n = len(off_diag)
    median = off_diag[n // 2] if n % 2 == 1 else (off_diag[n // 2 - 1] + off_diag[n // 2]) / 2
    return max(median, 1e-3)


def in_context_attention_baseline(task: Task, config: FastWeightConfig) -> tuple[float, dict]:
    """"Ordinary in-context learning": attends over the SAME `k_train`
    examples every other mechanism gets, no weight change. See
    `_kernel_regression_held_out_loss`."""
    del config
    started = time.perf_counter()
    temperature = _median_heuristic_temperature(task.train_x)
    loss = _kernel_regression_held_out_loss(task, task.train_x, task.train_y, temperature=temperature)
    return loss, {"method": "in_context_attention", "wall_seconds": time.perf_counter() - started, "temperature": temperature}


def longer_context_baseline(task: Task, config: FastWeightConfig, *, extra_k: int, seed: int) -> tuple[float, dict]:
    """Same in-context attention mechanism, but with `extra_k` MORE
    examples of the SAME true rule (generated fresh from
    `task.true_delta`/`task.base_weight`, never touching `held_out_x`/
    `held_out_y`). Tests whether just seeing more examples -- still no
    weight change -- can substitute for a small number of examples plus
    a real low-rank weight update."""
    del config
    started = time.perf_counter()
    key = mx.random.key(seed)
    extra_x = mx.random.normal((extra_k, task.base_weight.shape[0]), key=key)
    extra_y = extra_x @ (task.base_weight + task.true_delta).T + task.base_bias
    context_x = mx.concatenate([task.train_x, extra_x], axis=0)
    context_y = mx.concatenate([task.train_y, extra_y], axis=0)
    temperature = _median_heuristic_temperature(context_x)
    loss = _kernel_regression_held_out_loss(task, context_x, context_y, temperature=temperature)
    return loss, {
        "method": "longer_context", "wall_seconds": time.perf_counter() - started,
        "context_size": task.train_x.shape[0] + extra_k, "temperature": temperature,
    }


def knn_retrieval_baseline(task: Task, config: FastWeightConfig, *, k: int) -> tuple[float, dict]:
    """"Retrieval": for each held-out point, directly copy (average of)
    the `k` nearest training examples' OUTPUTS -- no base-model residual
    correction, no weight change, pure nearest-neighbor lookup. The
    substitute used here for "HZ-0B memory only" (see module
    docstring): an associative memory read IS a key-value lookup once
    there is no sequence to condition retrieval on."""
    del config
    started = time.perf_counter()
    diffs = task.held_out_x[:, None, :] - task.train_x[None, :, :]  # [k_held_out, k_train, dim]
    sq_dists = mx.sum(diffs * diffs, axis=-1)  # [k_held_out, k_train]
    order = mx.argsort(sq_dists, axis=-1)  # [k_held_out, k_train] ascending distance
    nearest_idx = order[:, :k]  # [k_held_out, k]
    gathered = task.train_y[nearest_idx.reshape(-1)].reshape(task.held_out_x.shape[0], k, -1)
    predicted = mx.mean(gathered, axis=1)
    loss = float(mx.mean(mx.sum((predicted - task.held_out_y) ** 2, axis=-1)))
    return loss, {"method": f"knn_retrieval_k{k}", "wall_seconds": time.perf_counter() - started}


def static_random_adapter_baseline(config: FastWeightConfig, task: Task, *, seed: int) -> tuple[float, dict]:
    """A low-rank adapter with the SAME shape as a real fast-weight
    state, initialized with BOTH factors random and nonzero (unlike
    `init_fast_weights`'s asymmetric zero-init) so it has a genuinely
    nonzero, unadapted delta -- then NEVER updated for this task.
    Isolates "having extra low-rank capacity" from "adapting that
    capacity to this session's examples": if this baseline does no
    better than `no_adaptation_baseline`, the capacity alone buys
    nothing, and any gain from the real mechanisms must come from
    genuine adaptation, not just from having spare parameters."""
    started = time.perf_counter()
    key = mx.random.key(seed)
    k1, k2 = mx.random.split(key)
    state = FastWeightState(
        a_fast=mx.random.normal((config.num_layers, config.dim, config.rank), key=k1) * config.init_scale,
        b_fast=mx.random.normal((config.num_layers, config.rank, config.dim), key=k2) * config.init_scale,
        update_count=mx.array(0, dtype=mx.int32),
    )
    loss = float(task_loss(task, state, task.held_out_x, task.held_out_y))
    return loss, {"method": "static_random_adapter", "wall_seconds": time.perf_counter() - started}


def meta_lora_baseline(config: FastWeightConfig, eval_task: Task, *, meta_train_seeds: list[int], steps: int, lr: float, meta_lr: float, task_factory) -> tuple[float, dict]:
    """"Permanent LoRA": ONE low-rank adapter, gradient-trained across
    MANY DIFFERENT tasks (`meta_train_seeds`, each with its own
    independently random `true_delta`, via `task_factory(seed)`), then
    FROZEN and evaluated on `eval_task`'s held-out data with NO further,
    session-specific adaptation at all. Tests whether a general,
    pretrained adapter -- the kind that would ship permanently with the
    model, never reset per session -- can capture the SAME gains as a
    fresh, temporary fast-weight update computed specifically for
    `eval_task`. Since each meta-training task's true rule is
    independent random noise relative to every other one (and to
    `eval_task`'s own rule), the only thing a single shared adapter CAN
    learn that generalizes is whatever is common across ALL of
    them -- for i.i.d. random rules, that is nothing systematic, so this
    baseline is expected to do no better than no adaptation. That
    expectation is checked here empirically, not assumed."""
    started = time.perf_counter()
    state = FastWeightState(
        a_fast=mx.random.normal((config.num_layers, config.dim, config.rank), key=mx.random.key(0)) * config.init_scale,
        b_fast=mx.zeros((config.num_layers, config.rank, config.dim)),
        update_count=mx.array(0, dtype=mx.int32),
    )
    for seed in meta_train_seeds:
        meta_task = task_factory(seed)

        def loss_fn(a_fast, b_fast):
            trial = FastWeightState(a_fast=a_fast, b_fast=b_fast, update_count=state.update_count)
            return task_loss(meta_task, trial, meta_task.train_x, meta_task.train_y)

        grad_fn = mx.value_and_grad(loss_fn, argnums=(0, 1))
        for _ in range(steps):
            _, (grad_a, grad_b) = grad_fn(state.a_fast, state.b_fast)
            mx.eval(grad_a, grad_b)
            state = update_fast_weights(state, LAYER, grad_a[LAYER], grad_b[LAYER], lr=meta_lr, config=config)
    # Frozen from here -- no adaptation to eval_task at all.
    loss = float(task_loss(eval_task, state, eval_task.held_out_x, eval_task.held_out_y))
    elapsed = time.perf_counter() - started
    return loss, {"method": "permanent_meta_lora", "wall_seconds": elapsed, "meta_train_tasks": len(meta_train_seeds)}
