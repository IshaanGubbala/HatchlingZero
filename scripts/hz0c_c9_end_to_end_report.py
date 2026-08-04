"""HZ-0C C9: end-to-end quality, cost, latency, and adversarial-failure
report.

Closes the last open C9 Required Artifact ("End-to-end quality, cost,
latency, and adversarial-failure report") and the plan's own completion-
definition item 8 ("Trigger cost, latency, and failure modes are
documented", `plans/HZ-0C_Surprise_Anchors_Total_Restart_Plan.md`). Prior
C4/C9 work (`scripts/hz0c_c4_fair_baselines.py`,
`scripts/hz0c_c9_matched_cost_report.py`) already covers recall/
precision/rate/execution-seconds/peak-RSS at matched cost across the 8
real C3 scenarios; this adds three genuinely new measurements those did
not:

1. Real per-call LATENCY (not just total execution seconds across a
   whole batch) at each trigger rate, with warmup and repeated timed
   calls.
2. The actual downstream LOSS COST of the deployed controller's missed
   triggers -- not just how many ground-truth positions it misses, but
   what missing each one actually costs in LM loss (single-position
   ablation, reusing the same causal downstream-benefit measurement
   `hz0c_c6_conditional_attention_eval.py::causal_attention_benefit`
   already established for C6).
3. One ADVERSARIAL scenario (`scenario_gradual_drift`, new) purpose-
   built to stress-test the mechanism's own documented weakness (the
   onset-vs-sustained pattern: only the FIRST token of a multi-position
   anomaly is strongly surprising, `reference/hz0c_surprise_trigger.py`'s
   module docstring) in its most extreme form -- a topic change with NO
   sharp onset at all, only a gradual linear drift from one real source
   to another, so no single token is ever locally surprising even though
   a real semantic shift has genuinely occurred by the midpoint.
"""
from __future__ import annotations

import argparse
import json
import random
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from reference.hz0b_b6_hz0a_integration import frozen_hidden_states
from reference.hz0c_surprise_trigger import normalize_score, token_loss_score
from scripts.hz0b_b11_baseline_comparison import load_frozen_model
from scripts.hz0c_c3_trigger_simulator import GENERAL_DATA_PATH, SEQ_LEN, load_real_sequences
from scripts.hz0c_c6_conditional_attention_eval import (
    causal_attention_benefit, conditional_forward, fixed_matched_trigger,
)
from scripts.hz0c_c7_rl_trigger_controller import controller_input, fit_controller
from scripts.hz0c_c9_matched_cost_report import build_scenarios, exact_topk, score_actions

TARGET_RATE = 0.15


def scenario_gradual_drift(count: int, rng: random.Random, general: list[list[int]]) -> tuple[mx.array, list[list[int]]]:
    """ADVERSARIAL: a real topic shift from source A to source B with NO
    sharp onset -- each position's probability of being drawn from B
    rises linearly from 0 at position 0 to ~1 at the last position,
    rather than `scenario_topic_shift`'s single abrupt boundary. Both A
    and B are real, in-distribution corpus content (this project's own
    established construction discipline -- arbitrary token IDs do not
    form a real "expectation" for a language-trained model, per
    `docs/restart/hz0c_c2_surprise_validation_results.md`), so no
    INDIVIDUAL token is locally anomalous; only the CUMULATIVE drift is
    real. Ground truth is the midpoint position where the mixture
    crosses 50% B -- the point by which a real, substantial topic change
    has genuinely happened, even though no single position triggered it.
    This directly tests whether the onset-vs-sustained weakness already
    documented for abrupt multi-token intrusions (only the first token
    is strongly surprising) becomes a complete miss when there is no
    onset to begin with."""
    rows, gts = [], []
    for _ in range(count):
        a, b = rng.sample(general, 2)
        row = []
        for i in range(SEQ_LEN):
            prob_b = i / (SEQ_LEN - 1)
            source = b if rng.random() < prob_b else a
            row.append(source[i % len(source)])
        midpoint = SEQ_LEN // 2
        rows.append(row)
        gts.append([midpoint])
    return mx.array(rows, dtype=mx.int32), gts


def measure_latency(model, tokens: mx.array, trigger: mx.array, *, repeats: int = 12, warmup: int = 3) -> dict:
    for _ in range(warmup):
        logits = conditional_forward(model, tokens, trigger)
        mx.eval(logits)
    timings = []
    for _ in range(repeats):
        started = time.perf_counter()
        logits = conditional_forward(model, tokens, trigger)
        mx.eval(logits)
        timings.append(time.perf_counter() - started)
    timings = np.asarray(timings, dtype=np.float64)
    batch, seq = tokens.shape
    return {
        "mean_seconds_per_call": float(timings.mean()),
        "std_seconds_per_call": float(timings.std()),
        "mean_ms_per_token": float(timings.mean() * 1000.0 / (batch * seq)),
    }


def train_controller(model, train_seeds: list[int], examples: int = 32) -> np.ndarray:
    features, labels = [], []
    for seed in train_seeds:
        for tokens, _ in build_scenarios(seed, examples):
            hidden, _ = frozen_hidden_states(model, tokens)
            mx.eval(hidden)
            features.append(controller_input(model, hidden, tokens))
            teacher = normalize_score(token_loss_score(model, hidden, tokens))
            labels.append(exact_topk(np.asarray(teacher), TARGET_RATE))
    return fit_controller(np.concatenate(features, axis=0), np.concatenate(labels, axis=0), steps=1200, lr=0.2, positive_weight=2.0)


def controller_trigger_for(model, params: np.ndarray, tokens: mx.array) -> mx.array:
    hidden, _ = frozen_hidden_states(model, tokens)
    mx.eval(hidden)
    feature = controller_input(model, hidden, tokens)
    logits = feature @ params[:-1] + params[-1]
    return mx.array(exact_topk(logits, TARGET_RATE))


def missed_trigger_cost(model, tokens: mx.array, gts: list[list[int]], trigger: mx.array) -> dict:
    """For every ground-truth position the deployed trigger MISSED,
    measure the real downstream LM-loss cost of adding just that one
    position back in -- not just how often triggers are missed, but
    what missing them actually costs. Reuses C6's own single-position
    downstream-benefit measurement rather than a new metric, so this is
    directly comparable to C6's causal-teacher screen."""
    trigger_np = np.asarray(trigger)
    missed = []
    for row, positions in enumerate(gts):
        for position in positions:
            if position < trigger_np.shape[1] and trigger_np[row, position] <= 0.0:
                missed.append((row, position))
    if not missed:
        return {"missed_count": 0, "mean_loss_cost": None, "examples": 0}
    costs = []
    for row, position in missed:
        # Single-position ablation, isolated to just this one example row so
        # adding the missed position back in cannot change any OTHER row's
        # loss -- a real, direct "what did missing THIS position cost".
        example_tokens = tokens[row:row + 1]
        baseline_row_trigger = trigger[row:row + 1]
        augmented_row = np.asarray(trigger_np[row]).copy()
        augmented_row[position] = 1.0
        augmented_row_trigger = mx.array(augmented_row[None, :])
        baseline_logits = conditional_forward(model, example_tokens, baseline_row_trigger)
        augmented_logits = conditional_forward(model, example_tokens, augmented_row_trigger)
        baseline_loss = mx.mean(nn.losses.cross_entropy(baseline_logits[:, :-1].astype(mx.float32), example_tokens[:, 1:]))
        augmented_loss = mx.mean(nn.losses.cross_entropy(augmented_logits[:, :-1].astype(mx.float32), example_tokens[:, 1:]))
        mx.eval(baseline_loss, augmented_loss)
        costs.append(float(baseline_loss) - float(augmented_loss))
    return {
        "missed_count": len(missed),
        "mean_loss_cost": float(np.mean(costs)),
        "std_loss_cost": float(np.std(costs)),
        "examples": len(gts),
    }


def main(train_seeds: list[int] | None = None, eval_seed: int = 557, examples: int = 32, adversarial_examples: int = 32, latency_repeats: int = 12) -> None:
    model, payload = load_frozen_model()
    selected_train_seeds = train_seeds or [555, 556]
    params = train_controller(model, selected_train_seeds, examples)

    eval_scenarios = build_scenarios(eval_seed, examples)
    quality = {}
    cost = {}
    for index, (tokens, gts) in enumerate(eval_scenarios):
        trigger = controller_trigger_for(model, params, tokens)
        quality[f"scenario_{index}"] = score_actions(np.asarray(trigger), gts)
        cost[f"scenario_{index}"] = missed_trigger_cost(model, tokens, gts, trigger)

    latency_tokens = mx.array(load_real_sequences(GENERAL_DATA_PATH, 8), dtype=mx.int32)[:, :SEQ_LEN]
    latency = {}
    for name, trigger in {
        "no_anchor": mx.zeros(latency_tokens.shape),
        "fixed_15pct": fixed_matched_trigger(*latency_tokens.shape, rate=TARGET_RATE),
        "full_attention": mx.ones(latency_tokens.shape),
    }.items():
        latency[name] = measure_latency(model, latency_tokens, trigger, repeats=latency_repeats)

    rng = random.Random(eval_seed + 1000)
    general = load_real_sequences(GENERAL_DATA_PATH, 200)
    adversarial_tokens, adversarial_gts = scenario_gradual_drift(adversarial_examples, rng, general)
    adversarial_trigger = controller_trigger_for(model, params, adversarial_tokens)
    adversarial_quality = score_actions(np.asarray(adversarial_trigger), adversarial_gts)
    adversarial_cost = missed_trigger_cost(model, adversarial_tokens, adversarial_gts, adversarial_trigger)

    missed_total = sum(c["missed_count"] for c in cost.values())
    finite_costs = [c["mean_loss_cost"] for c in cost.values() if c["mean_loss_cost"] is not None]

    report = {
        "stage": "HZ-0C-C9-end-to-end-report",
        "checkpoint_step": payload["step"],
        "train_seeds": selected_train_seeds,
        "eval_seed": eval_seed,
        "target_rate": TARGET_RATE,
        "quality_by_scenario": quality,
        "missed_trigger_cost_by_scenario": cost,
        "missed_trigger_total_count": missed_total,
        "missed_trigger_mean_loss_cost_across_scenarios": float(np.mean(finite_costs)) if finite_costs else None,
        "latency": latency,
        "adversarial_gradual_drift": {
            "quality": adversarial_quality,
            "missed_trigger_cost": adversarial_cost,
            "note": "no single token in this scenario is locally anomalous; ground truth is a cumulative-drift midpoint, not an onset",
        },
        "finite": bool(np.isfinite(missed_total)) and all(np.isfinite(v) for v in finite_costs),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-seeds", type=int, nargs="+", default=None)
    parser.add_argument("--eval-seed", type=int, default=557)
    parser.add_argument("--examples", type=int, default=32)
    parser.add_argument("--adversarial-examples", type=int, default=32)
    parser.add_argument("--latency-repeats", type=int, default=12)
    args = parser.parse_args()
    main(args.train_seeds, args.eval_seed, args.examples, args.adversarial_examples, args.latency_repeats)
