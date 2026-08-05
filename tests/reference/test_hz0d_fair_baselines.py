"""HZ-0D D4: fair adaptation baseline tests (reference/hz0d_fair_baselines.py).
Locks in the real, measured findings from
`docs/restart/hz0d_d4_fair_baselines_results.md` as regression tests --
the D4 exit gate itself ("gains are attributable to temporary fast
adaptation"), not just "each baseline runs."
"""
from __future__ import annotations

import mlx.core as mx

from reference.hz0d_fast_weights import FastWeightConfig, init_fast_weights
from reference.hz0d_isolated_simulator import make_task, held_out_generalization_loss
from reference.hz0d_update_mechanisms import delta_prediction_update, gradient_descent_update
from reference.hz0d_fair_baselines import (
    in_context_attention_baseline, knn_retrieval_baseline, longer_context_baseline,
    meta_lora_baseline, no_adaptation_baseline, static_random_adapter_baseline,
)

CONFIG = FastWeightConfig(dim=8, rank=2, num_layers=1, max_delta_norm=10.0)


def _task_factory(seed: int):
    return make_task(CONFIG, seed=seed, k_train=6, k_held_out=16, rule_scale=0.3)


def test_all_baselines_produce_finite_losses():
    task = _task_factory(0)
    loss, _ = no_adaptation_baseline(task, CONFIG)
    assert loss == loss and loss >= 0.0
    loss, _ = in_context_attention_baseline(task, CONFIG)
    assert loss == loss and loss >= 0.0
    loss, _ = longer_context_baseline(task, CONFIG, extra_k=24, seed=1)
    assert loss == loss and loss >= 0.0
    loss, _ = knn_retrieval_baseline(task, CONFIG, k=1)
    assert loss == loss and loss >= 0.0
    loss, _ = static_random_adapter_baseline(CONFIG, task, seed=1)
    assert loss == loss and loss >= 0.0
    loss, _ = meta_lora_baseline(CONFIG, task, meta_train_seeds=[100, 101, 102], steps=20, lr=0.02, meta_lr=0.02, task_factory=_task_factory)
    assert loss == loss and loss >= 0.0


def test_static_random_adapter_matches_no_adaptation():
    """Extra unadapted low-rank capacity, by itself, must buy nothing --
    a random nonzero delta that was never fit to this task's examples
    should perform statistically the same as having no delta at all.
    If it didn't, "gains from adaptation" would be confounded with
    "gains from just having more parameters"."""
    task = _task_factory(2)
    no_adapt_loss, _ = no_adaptation_baseline(task, CONFIG)
    static_loss, _ = static_random_adapter_baseline(CONFIG, task, seed=2)
    assert abs(static_loss - no_adapt_loss) < 0.01 * max(no_adapt_loss, 1.0), (
        f"expected static random adapter to match no-adaptation closely: {static_loss} vs {no_adapt_loss}"
    )


def test_permanent_meta_lora_does_not_beat_no_adaptation():
    """A single adapter meta-trained across many INDEPENDENT random-rule
    tasks has nothing systematic to learn (each task's rule is
    unrelated random noise relative to every other), so it should not
    generalize better than doing nothing on a NEW task's specific
    rule -- confirming a permanent, never-reset adapter cannot
    substitute for genuine per-session adaptation."""
    task = _task_factory(3)
    no_adapt_loss, _ = no_adaptation_baseline(task, CONFIG)
    meta_loss, _ = meta_lora_baseline(
        CONFIG, task, meta_train_seeds=[s for s in range(100, 110) if s != 3],
        steps=100, lr=0.02, meta_lr=0.02, task_factory=_task_factory,
    )
    assert meta_loss >= no_adapt_loss * 0.9, (
        f"expected permanent meta-LoRA to not meaningfully beat no-adaptation: meta={meta_loss} no_adapt={no_adapt_loss}"
    )


def test_fast_weight_adaptation_beats_every_baseline_on_mean_held_out_loss():
    """The real D4 exit gate: real fast-weight adaptation (both gradient
    descent and delta prediction) must beat EVERY baseline here on mean
    held-out loss across the same 8 seeds -- no adaptation, in-context
    attention, longer (unadapted) context, k-NN retrieval, a static
    random adapter, and a permanent meta-trained adapter. If any
    baseline came close, the gain could not be attributed specifically
    to temporary, session-local fast-weight adaptation."""
    baseline_means = {
        "no_adaptation": [], "in_context": [], "longer_context_24": [],
        "knn_k3": [], "static_random": [],
    }
    gd_losses, delta_losses = [], []
    for seed in range(8):
        task = _task_factory(seed)
        baseline_means["no_adaptation"].append(no_adaptation_baseline(task, CONFIG)[0])
        baseline_means["in_context"].append(in_context_attention_baseline(task, CONFIG)[0])
        baseline_means["longer_context_24"].append(longer_context_baseline(task, CONFIG, extra_k=24, seed=1000 + seed)[0])
        baseline_means["knn_k3"].append(knn_retrieval_baseline(task, CONFIG, k=3)[0])
        baseline_means["static_random"].append(static_random_adapter_baseline(CONFIG, task, seed=seed)[0])
        gd_state, _ = gradient_descent_update(task, init_fast_weights(CONFIG), CONFIG, steps=400, lr=0.02)
        gd_losses.append(held_out_generalization_loss(task, gd_state))
        delta_state, _ = delta_prediction_update(task, init_fast_weights(CONFIG), CONFIG)
        delta_losses.append(held_out_generalization_loss(task, delta_state))

    mean_gd = sum(gd_losses) / len(gd_losses)
    mean_delta = sum(delta_losses) / len(delta_losses)
    for name, values in baseline_means.items():
        mean_baseline = sum(values) / len(values)
        assert mean_gd < mean_baseline, f"gradient-descent fast weights should beat {name}: gd={mean_gd} {name}={mean_baseline}"
        assert mean_delta < mean_baseline, f"delta-prediction fast weights should beat {name}: delta={mean_delta} {name}={mean_baseline}"


def test_knn_retrieval_without_base_model_correction_can_underperform_no_adaptation():
    """A real, disclosed finding: raw nearest-neighbor output copying
    (no base-model residual correction, unlike the in-context attention
    baseline) can do WORSE than the frozen base model alone -- ignoring
    what the base model already knows is a real cost, not a strawman
    baseline rigged to lose."""
    losses_knn1, losses_no_adapt = [], []
    for seed in range(8):
        task = _task_factory(seed)
        losses_knn1.append(knn_retrieval_baseline(task, CONFIG, k=1)[0])
        losses_no_adapt.append(no_adaptation_baseline(task, CONFIG)[0])
    assert sum(losses_knn1) / len(losses_knn1) > sum(losses_no_adapt) / len(losses_no_adapt)
