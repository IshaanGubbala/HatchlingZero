#!/usr/bin/env python3
"""HZ-Bench Milestone 1 runner for `combined_best` BDH checkpoints
(plans/Hatchling world.md section 0.8) -- same real `lm-evaluation-
harness` plumbing validated for `HZLanguageModel`
(`scripts/hz_bench_milestone1.py`), pointed at
`scripts/hz_bdh_lm_eval_adapter.py`'s `"hz_bdh"` model instead. NOT a
capability claim -- a corpus-only, ~500-step Stage 0 systems-test
checkpoint is expected to score at or near the random-choice floor on
every task; this proves the harness/tokenization/scoring convention
work for this architecture too, establishing the floor future, longer
HZ-Bench-100M/600M runs will be measured against.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hz_bdh_lm_eval_adapter  # noqa: E402,F401  (registers "hz_bdh" with lm_eval)
from lm_eval import simple_evaluate  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--n-embd", type=int, default=1440)
    parser.add_argument("--n-layer", type=int, default=8)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--mult", type=int, default=16)
    parser.add_argument("--tasks", type=str, default="arc_easy,arc_challenge,hellaswag,piqa,winogrande,boolq")
    parser.add_argument("--limit", type=int, default=30, help="examples per task -- keep small, this is plumbing")
    parser.add_argument("--results-file", type=Path, default=Path("results/local/hz_bdh_bench_milestone1.json"))
    args = parser.parse_args()

    model_args = (f"checkpoint={args.checkpoint},n_embd={args.n_embd},n_layer={args.n_layer},"
                  f"n_head={args.n_head},mult={args.mult}")
    tasks = args.tasks.split(",")
    print(f"[hz-bdh-bench-m1] model_args={model_args}", flush=True)
    print(f"[hz-bdh-bench-m1] tasks={tasks} limit={args.limit}", flush=True)

    results = simple_evaluate(model="hz_bdh", model_args=model_args, tasks=tasks, limit=args.limit,
                               log_samples=False, random_seed=0, numpy_random_seed=0, torch_random_seed=0)

    summary = {task: metrics for task, metrics in results["results"].items()}
    print("\n[hz-bdh-bench-m1] ===== RESULTS =====", flush=True)
    for task, metrics in summary.items():
        print(f"[hz-bdh-bench-m1] {task}: {metrics}", flush=True)

    args.results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.results_file, "w") as f:
        json.dump({"checkpoint": str(args.checkpoint), "tasks": tasks, "limit": args.limit,
                    "results": summary}, f, indent=2, default=str)
    print(f"\n[hz-bdh-bench-m1] wrote {args.results_file}", flush=True)


if __name__ == "__main__":
    main()
