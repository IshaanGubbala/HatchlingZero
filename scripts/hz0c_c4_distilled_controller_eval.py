"""Evaluate the C7 causal distilled controller under the C4 exact-rate metric."""
from __future__ import annotations

import json
import random

import mlx.core as mx
import numpy as np

from reference.hz0b_b6_hz0a_integration import frozen_hidden_states
from reference.hz0c_surprise_trigger import normalize_score, state_novelty_score, token_loss_score
from scripts.hz0b_b11_baseline_comparison import load_frozen_model
from scripts.hz0c_c3_trigger_simulator import (
    CODE_DATA_PATH, GENERAL_DATA_PATH, JSON_DATA_PATH, TARGET_RATE,
    load_real_sequences, scenario_code_json_boundary, scenario_contradiction,
    scenario_distractor_heavy_retrieval, scenario_long_range_reappearance,
    scenario_rare_token_burst, scenario_repeated_pattern_anomaly,
    scenario_topic_shift, scenario_variable_rebinding,
)
from scripts.hz0c_c7_rl_trigger_controller import controller_input, fit_controller
from scripts.hz0c_c4_fair_baselines import (
    causal_uncertainty_components, layer_aware_demand_components,
    top_rate_trigger,
)


def exact_topk(score: np.ndarray, rate: float) -> np.ndarray:
    count = max(1, round(rate * score.shape[-1]))
    order = np.argsort(-score, axis=-1)
    result = np.zeros_like(score, dtype=np.float32)
    rows = np.arange(score.shape[0])[:, None]
    result[rows, order[:, :count]] = 1.0
    return result


def metric(actions: np.ndarray, gts: list[list[int]]) -> dict[str, float]:
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
        "anchor_rate": float(actions.mean()),
    }


def main(seed: int = 555, positive_weight: float = 2.0, eval_seed: int | None = None, train_seeds: list[int] | None = None) -> None:
    model, _ = load_frozen_model()
    general = load_real_sequences(GENERAL_DATA_PATH, 200)
    code = load_real_sequences(CODE_DATA_PATH, 100)
    json_data = load_real_sequences(JSON_DATA_PATH, 100)
    def make_scenarios(scenario_seed: int):
        rng = random.Random(scenario_seed)
        return [
            scenario_repeated_pattern_anomaly(32, rng, general),
            scenario_topic_shift(32, rng, general),
            scenario_long_range_reappearance(32, rng, general),
            scenario_variable_rebinding(32, rng, general),
            scenario_code_json_boundary(32, rng, code, json_data),
            scenario_contradiction(32, rng, general),
            scenario_rare_token_burst(32, rng, general, 24576),
            scenario_distractor_heavy_retrieval(32, rng, general, code),
        ]
    train_scenarios = [scenario for train_seed in (train_seeds or [seed]) for scenario in make_scenarios(train_seed)]
    scenarios = make_scenarios(seed if eval_seed is None else eval_seed)
    features, labels, selected_scores, hidden_batches = [], [], [], []
    for tokens, _ in train_scenarios:
        hidden, _ = frozen_hidden_states(model, tokens)
        mx.eval(hidden)
        features.append(controller_input(model, hidden, tokens))
        labels.append(exact_topk(np.asarray(normalize_score(token_loss_score(model, hidden, tokens))), TARGET_RATE))
    eval_features = []
    for tokens, _ in scenarios:
        hidden, _ = frozen_hidden_states(model, tokens)
        mx.eval(hidden)
        eval_features.append(controller_input(model, hidden, tokens))
        novelty = normalize_score(state_novelty_score(hidden, window=4))
        uncertainty = causal_uncertainty_components(model, hidden)[..., 0]
        demand_mean, _, demand_max = layer_aware_demand_components(model, tokens)
        selected_scores.append(normalize_score(
            novelty + normalize_score(uncertainty)
            + 0.25 * normalize_score(demand_mean)
            + 0.5 * normalize_score(demand_max)
        ))
    x = np.concatenate(features, axis=0)
    y = np.concatenate(labels, axis=0)
    params = fit_controller(x, y, steps=1200, lr=0.2, positive_weight=positive_weight)

    per_scenario = []
    selected_metrics = []
    for feature, selected, (_, gts) in zip(eval_features, selected_scores, scenarios):
        scores = feature @ params[:-1] + params[-1]
        per_scenario.append(metric(exact_topk(scores, TARGET_RATE), gts))
        selected_metrics.append(metric(np.asarray(top_rate_trigger(selected, TARGET_RATE)), gts))
    report = {
        "stage": "C4-causal-distilled-controller",
        "seed": seed,
        "eval_seed": seed if eval_seed is None else eval_seed,
        "train_seeds": train_seeds or [seed],
        "positive_weight": positive_weight,
        "scenarios": per_scenario,
        "mean_precision": float(np.mean([x["precision"] for x in per_scenario])),
        "mean_recall": float(np.mean([x["recall"] for x in per_scenario])),
        "mean_anchor_rate": float(np.mean([x["anchor_rate"] for x in per_scenario])),
        "selected_c4_mean_recall": float(np.mean([x["recall"] for x in selected_metrics])),
        "selected_c4_mean_precision": float(np.mean([x["precision"] for x in selected_metrics])),
        "finite": bool(np.all(np.isfinite(params))),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=555)
    parser.add_argument("--positive-weight", type=float, default=2.0)
    parser.add_argument("--eval-seed", type=int, default=None)
    parser.add_argument("--train-seeds", type=int, nargs="+", default=None)
    args = parser.parse_args()
    main(args.seed, args.positive_weight, args.eval_seed, args.train_seeds)
