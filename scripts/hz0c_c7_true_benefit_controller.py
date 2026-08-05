"""HZ-0C C7: a controller trained AND evaluated against a CONSISTENT
ground-truth definition -- real measured downstream benefit, not the C3
scenarios' hand-labeled construction points.

The fourth C7 plateau investigation
(`docs/restart/hz0c_c9_attention_pattern_feature_results.md`) tried
`causal_attention_benefit` (C6's real per-position downstream LM-loss
benefit measurement) as a teacher and found it scored dramatically WORSE
than `token_loss_score` against the C3 scenarios' hand-labeled ground
truth -- and traced this to an evaluation-target MISMATCH, not a signal
defect: `causal_attention_benefit` answers "which position truly reduces
downstream loss," a different question from "which position matches
this scenario's construction point" (e.g. a topic-shift's exact
boundary token). Training against one definition and evaluating against
the other was never a fair test of either.

This removes the mismatch by using the SAME ground-truth definition
everywhere: both the distillation teacher AND the evaluation metric are
now the top-15%-by-real-measured-downstream-benefit positions, computed
at full (unrestricted) candidate budget. This asks the actual question
this investigation was implicitly circling without saying so: can a
controller using only CAUSAL, inference-safe features learn to
approximate genuinely-useful anchor positions, when "genuinely useful"
is measured consistently at both training and evaluation time?
"""
from __future__ import annotations

import argparse
import json

import mlx.core as mx
import numpy as np

from reference.hz0b_b6_hz0a_integration import frozen_hidden_states
from scripts.hz0b_b11_baseline_comparison import load_frozen_model
from scripts.hz0c_c6_conditional_attention_eval import causal_attention_benefit
from scripts.hz0c_c7_rl_trigger_controller import controller_input, fit_controller, fit_ranking_controller
from scripts.hz0c_c9_matched_cost_report import build_scenarios, exact_topk, score_actions
from scripts.hz0c_c3_trigger_simulator import TARGET_RATE


def true_benefit_labels(model, tokens: mx.array) -> np.ndarray:
    """Top-15%-by-real-downstream-benefit positions, full candidate
    budget (every position is scored, none are pre-filtered by a cheap
    proxy) -- the SAME quantity used as both teacher and evaluation
    ground truth below."""
    seq = tokens.shape[1]
    benefit = causal_attention_benefit(model, tokens, candidates=seq)
    return exact_topk(benefit, TARGET_RATE)


def gts_from_labels(labels: np.ndarray) -> list[list[int]]:
    return [list(np.flatnonzero(row > 0.5)) for row in labels]


def main(train_seeds: list[int] | None = None, eval_seed: int = 557, examples: int = 32, objective: str = "bce") -> None:
    model, payload = load_frozen_model()
    selected_train_seeds = train_seeds or [555, 556]

    train_features, train_labels = [], []
    for seed in selected_train_seeds:
        for tokens, _ in build_scenarios(seed, examples):
            hidden, _ = frozen_hidden_states(model, tokens)
            mx.eval(hidden)
            train_features.append(controller_input(model, hidden, tokens))
            train_labels.append(true_benefit_labels(model, tokens))
    x = np.concatenate(train_features, axis=0)
    y = np.concatenate(train_labels, axis=0)

    if objective == "ranking":
        params = fit_ranking_controller(x, y, steps=1200, lr=0.2)
    else:
        params = fit_controller(x, y, steps=1200, lr=0.2, positive_weight=2.0)

    eval_scenarios = build_scenarios(eval_seed, examples)
    controller_results, old_hand_labeled_results = [], []
    for tokens, hand_labeled_gts in eval_scenarios:
        hidden, _ = frozen_hidden_states(model, tokens)
        mx.eval(hidden)
        feature = controller_input(model, hidden, tokens)
        logits = feature @ params[:-1] + params[-1]
        controller_actions = exact_topk(logits, TARGET_RATE)

        true_benefit = true_benefit_labels(model, tokens)
        true_benefit_gts = gts_from_labels(true_benefit)

        controller_results.append(score_actions(controller_actions, true_benefit_gts))
        old_hand_labeled_results.append(score_actions(controller_actions, hand_labeled_gts))

    def aggregate(results: list[dict]) -> dict:
        precision_values = [r["precision"] for r in results if np.isfinite(r["precision"])]
        return {
            "recall_mean": float(np.mean([r["recall"] for r in results])),
            "precision_mean": float(np.mean(precision_values)) if precision_values else None,
        }

    report = {
        "stage": "HZ-0C-C7-true-benefit-consistent-ground-truth",
        "checkpoint_step": payload["step"],
        "train_seeds": selected_train_seeds,
        "eval_seed": eval_seed,
        "objective": objective,
        "target_rate": TARGET_RATE,
        "recall_vs_true_benefit_ground_truth": aggregate(controller_results),
        "recall_vs_old_hand_labeled_ground_truth_for_context": aggregate(old_hand_labeled_results),
        "note": "The first number is the fair, consistent evaluation (same ground-truth definition used for training and evaluation). The second is reported only for context against the pre-existing hand-labeled-ground-truth baseline (0.5182/0.1068) -- it is NOT expected to be comparable, since this controller was never trained to match that definition.",
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-seeds", type=int, nargs="+", default=None)
    parser.add_argument("--eval-seed", type=int, default=557)
    parser.add_argument("--examples", type=int, default=32)
    parser.add_argument("--objective", choices=("bce", "ranking"), default="bce")
    args = parser.parse_args()
    main(args.train_seeds, args.eval_seed, args.examples, args.objective)
