"""Run the C7 controller replay over a deterministic seed set.

This keeps the per-seed controller implementation unchanged while producing
one machine-readable aggregate report for audit and tracker updates.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import statistics

from scripts.hz0c_c7_rl_trigger_controller import main


def run(seed: int, args: argparse.Namespace) -> dict:
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        main(
            seed=seed,
            rl_steps=args.rl_steps,
            rl_lr=args.rl_learning_rate,
            distill_steps=args.distillation_steps,
            distill_lr=args.distillation_learning_rate,
            positive_weight=args.distillation_positive_weight,
            causal_teacher_sequences=args.causal_teacher_sequences,
            causal_teacher_candidates=args.causal_teacher_candidates,
            causal_teacher_blend=args.causal_teacher_blend,
        )
    return json.loads(captured.getvalue())


def main_report(args: argparse.Namespace) -> dict:
    reports = [run(seed, args) for seed in args.seeds]
    recall = [item["controller_mean_event_recall"] for item in reports]
    reward = [item["final_group_reward"] for item in reports]
    rate = [item["controller_mean_anchor_rate"] for item in reports]
    return {
        "stage": "C7-RL-trigger-controller-multi-seed",
        "seeds": args.seeds,
        "per_seed": reports,
        "controller_mean_event_recall": {
            "mean": statistics.fmean(recall),
            "population_std": statistics.pstdev(recall),
        },
        "final_group_reward": {
            "mean": statistics.fmean(reward),
            "population_std": statistics.pstdev(reward),
        },
        "controller_mean_anchor_rate": {
            "mean": statistics.fmean(rate),
            "min": min(rate),
            "max": max(rate),
        },
        "finite_all": all(item["finite"] for item in reports),
        "hard_rate_bounds": reports[0]["hard_rate_bounds"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate deterministic C7 seed replays.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[555, 556, 557])
    parser.add_argument("--rl-steps", type=int, default=200)
    parser.add_argument("--rl-learning-rate", type=float, default=0.05)
    parser.add_argument("--distillation-steps", type=int, default=1200)
    parser.add_argument("--distillation-learning-rate", type=float, default=0.2)
    parser.add_argument("--distillation-positive-weight", type=float, default=2.0)
    parser.add_argument("--causal-teacher-sequences", type=int, default=0)
    parser.add_argument("--causal-teacher-candidates", type=int, default=4)
    parser.add_argument("--causal-teacher-blend", type=float, default=0.5)
    args = parser.parse_args()
    print(json.dumps(main_report(args), indent=2, sort_keys=True))
