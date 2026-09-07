#!/usr/bin/env python3
"""HZ-Bench Milestone 1 runner: proves the `lm-evaluation-harness`
plumbing works end-to-end on a real HZ checkpoint (plans/Hatchling
world.md), per the user's explicit direction to build the benchmark
harness before further scaling. NOT a capability claim -- the current
mainline checkpoint is a 5,056,229-param model trained on a handful of
real corpus paragraphs; scores at or near the random-choice floor on
every task are the EXPECTED result here. The point is establishing
that harness + tokenizer + scoring convention work, so later scaled
checkpoints (HZ-Bench-100M, HZ-Bench-600M) can be dropped in and
produce real, comparable numbers with zero adapter changes.

Run with the venv that has lm-eval installed:
  .venv/bin/python3 scripts/hz_bench_milestone1.py --tasks arc_easy --limit 20
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hz_lm_eval_adapter  # noqa: E402,F401  (registers "hz" with lm_eval)
from lm_eval import simple_evaluate  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("results/local/hz_world_run2/step_36000.pt"),
                         help="current Hatchling World mainline checkpoint (Arm A, 12k->36k)")
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--memory-slots", type=int, default=16)
    parser.add_argument("--workspace-slots", type=int, default=64)
    parser.add_argument("--n-rounds-l1", type=int, default=8)
    parser.add_argument("--tasks", type=str, default="arc_easy",
                         help="comma-separated lm-eval task names, e.g. arc_easy,piqa,boolq")
    parser.add_argument("--limit", type=int, default=20, help="examples per task -- keep small, this is plumbing")
    parser.add_argument("--results-file", type=Path, default=Path("results/local/hz_bench_milestone1.json"))
    args = parser.parse_args()

    model_args = (f"checkpoint={args.checkpoint},d_model={args.d_model},"
                  f"memory_slots={args.memory_slots},workspace_slots={args.workspace_slots},"
                  f"n_rounds_l1={args.n_rounds_l1}")
    tasks = args.tasks.split(",")
    print(f"[hz-bench-m1] model_args={model_args}", flush=True)
    print(f"[hz-bench-m1] tasks={tasks} limit={args.limit}", flush=True)

    results = simple_evaluate(model="hz", model_args=model_args, tasks=tasks, limit=args.limit,
                               log_samples=False, random_seed=0, numpy_random_seed=0, torch_random_seed=0)

    summary = {task: metrics for task, metrics in results["results"].items()}
    print("\n[hz-bench-m1] ===== RESULTS =====", flush=True)
    for task, metrics in summary.items():
        print(f"[hz-bench-m1] {task}: {metrics}", flush=True)

    args.results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.results_file, "w") as f:
        json.dump({"checkpoint": str(args.checkpoint), "tasks": tasks, "limit": args.limit,
                    "results": summary}, f, indent=2, default=str)
    print(f"\n[hz-bench-m1] wrote {args.results_file}", flush=True)


if __name__ == "__main__":
    main()
