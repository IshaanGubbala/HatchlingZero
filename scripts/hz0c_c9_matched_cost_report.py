"""Bounded HZ-0C C9 report for causal trigger quality and execution cost.

This intentionally skips transformer training. It evaluates the verified
inference-safe trigger policies on the same eight C3 scenarios and emits one
machine-readable record suitable for later C6/C8 cost comparisons.
"""
from __future__ import annotations

import argparse
import json
import random
import resource
import sys
import time

import mlx.core as mx
import numpy as np

from scripts.hz0c_c4_fair_baselines import (
    CODE_DATA_PATH,
    GENERAL_DATA_PATH,
    JSON_DATA_PATH,
    TARGET_RATE,
    evaluate_all_baselines,
    load_frozen_model,
    load_real_sequences,
    scenario_code_json_boundary,
    scenario_contradiction,
    scenario_distractor_heavy_retrieval,
    scenario_long_range_reappearance,
    scenario_rare_token_burst,
    scenario_repeated_pattern_anomaly,
    scenario_topic_shift,
    scenario_variable_rebinding,
)
from reference.hz0b_b6_hz0a_integration import frozen_hidden_states
from reference.hz0c_surprise_trigger import normalize_score, token_loss_score
from scripts.hz0c_c7_rl_trigger_controller import controller_input, fit_controller


def exact_topk(score: np.ndarray, rate: float) -> np.ndarray:
    count = max(1, round(rate * score.shape[-1]))
    order = np.argsort(-score, axis=-1)
    actions = np.zeros_like(score, dtype=np.float32)
    rows = np.arange(score.shape[0])[:, None]
    actions[rows, order[:, :count]] = 1.0
    return actions


def score_actions(actions: np.ndarray, gts: list[list[int]]) -> dict[str, float]:
    tp = fp = fn = 0
    for row, positions in enumerate(gts):
        truth = set(positions)
        selected = set(np.flatnonzero(actions[row] > 0.5).tolist())
        tp += len(truth & selected)
        fp += len(selected - truth)
        fn += len(truth - selected)
    return {
        "precision": tp / max(1, tp + fp),
        "recall": tp / max(1, tp + fn),
        "rate": float(actions.mean()),
    }


def distilled_policy(model, train_scenarios: list[tuple[mx.array, list[list[int]]]], eval_scenarios: list[tuple[mx.array, list[list[int]]]], *, distill_steps: int = 1200, positive_weight: float = 2.0) -> list[dict[str, float]]:
    """Train on one scenario split, then score a separate C9 split."""
    features, labels = [], []
    for tokens, _ in train_scenarios:
        hidden, _ = frozen_hidden_states(model, tokens)
        mx.eval(hidden)
        features.append(controller_input(model, hidden, tokens))
        teacher = normalize_score(token_loss_score(model, hidden, tokens))
        labels.append(exact_topk(np.asarray(teacher), TARGET_RATE))
    params = fit_controller(
        np.concatenate(features, axis=0), np.concatenate(labels, axis=0),
        steps=distill_steps, lr=0.2, positive_weight=positive_weight,
    )
    results = []
    for tokens, gts in eval_scenarios:
        hidden, _ = frozen_hidden_states(model, tokens)
        mx.eval(hidden)
        feature = controller_input(model, hidden, tokens)
        results.append(score_actions(
            exact_topk(feature @ params[:-1] + params[-1], TARGET_RATE), gts
        ))
    return results


def build_scenarios(seed: int, examples: int) -> list[tuple[mx.array, list[list[int]]]]:
    general = load_real_sequences(GENERAL_DATA_PATH, 200)
    code = load_real_sequences(CODE_DATA_PATH, 100)
    json_data = load_real_sequences(JSON_DATA_PATH, 100)
    rng = random.Random(seed)
    return [
        scenario_repeated_pattern_anomaly(examples, rng, general),
        scenario_topic_shift(examples, rng, general),
        scenario_long_range_reappearance(examples, rng, general),
        scenario_variable_rebinding(examples, rng, general),
        scenario_code_json_boundary(examples, rng, code, json_data),
        scenario_contradiction(examples, rng, general),
        scenario_rare_token_burst(examples, rng, general, 24576),
        scenario_distractor_heavy_retrieval(examples, rng, general, code),
    ]


def peak_rss_bytes() -> int:
    """Return process peak RSS with macOS/Linux unit differences handled."""
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def main(seed: int = 555, examples: int = 32, train_seed: int | None = None, train_seeds: list[int] | None = None, distill_steps: int = 1200, positive_weight: float = 2.0) -> None:
    model, payload = load_frozen_model()
    scenarios = build_scenarios(seed, examples)
    selected_train_seeds = train_seeds or [seed if train_seed is None else train_seed]
    train_scenarios = [
        scenario
        for current_seed in selected_train_seeds
        for scenario in build_scenarios(current_seed, examples)
    ]
    results_by_policy: dict[str, list[dict[str, float]]] = {}
    # Run one complete baseline pass per scenario and synchronize before
    # recording time. Keeping scenario boundaries preserves the C4 protocol.
    started = time.perf_counter()
    for tokens, ground_truth in scenarios:
        results = evaluate_all_baselines(model, None, tokens, ground_truth, seed=seed)
        for name, value in results.items():
            results_by_policy.setdefault(name, []).append(value)
    results_by_policy["causal_distilled_controller"] = distilled_policy(
        model, train_scenarios, scenarios,
        distill_steps=distill_steps, positive_weight=positive_weight,
    )
    mx.eval(model.parameters())
    elapsed = time.perf_counter() - started
    aggregate = {}
    for name, values in results_by_policy.items():
        precision_values = [v["precision"] for v in values if np.isfinite(v["precision"])]
        aggregate[name] = {
            "precision_mean": (
                sum(precision_values) / len(precision_values)
                if precision_values else None
            ),
            "recall_mean": sum(v["recall"] for v in values) / len(values),
            "rate_mean": sum(v.get("avg_rate", v.get("rate", 0.0)) for v in values) / len(values),
        }
    finite = all(
        np.isfinite(value)
        for policy in aggregate.values()
        for key, value in policy.items()
        if value is not None
    )
    report = {
        "stage": "HZ-0C-C9-matched-cost-trigger-report",
        "seed": seed,
        "train_seed": seed if train_seed is None else train_seed,
        "train_seeds": selected_train_seeds,
        "scenarios": len(scenarios),
        "examples_per_scenario": examples,
        "target_rate": TARGET_RATE,
        "checkpoint_step": payload["step"],
        "policies": aggregate,
        "execution_seconds": elapsed,
        "peak_memory_bytes": peak_rss_bytes(),
        "memory_note": "process peak RSS; MLX device allocator peak is not exposed",
        "finite": bool(finite),
        "distill_steps": distill_steps,
        "positive_weight": positive_weight,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=555)
    parser.add_argument("--examples", type=int, default=32)
    parser.add_argument("--train-seed", type=int, default=None)
    parser.add_argument("--train-seeds", type=int, nargs="+", default=None)
    parser.add_argument("--distill-steps", type=int, default=1200)
    parser.add_argument("--positive-weight", type=float, default=2.0)
    args = parser.parse_args()
    main(args.seed, args.examples, args.train_seed, args.train_seeds,
         args.distill_steps, args.positive_weight)
