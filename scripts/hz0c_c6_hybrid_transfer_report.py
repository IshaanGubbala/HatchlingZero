"""Run the bounded HZ-0C hybrid controller transfer protocol safely.

Runs one evaluation at a time, kills the whole process group on timeout, and
emits one strict JSON report. This is intentionally separate from the single
split evaluator so an interrupted MLX run cannot leave later splits running.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from statistics import mean


def _descendants(root_pid: int) -> list[int]:
    rows = subprocess.check_output(["ps", "-axo", "pid=,ppid="], text=True).splitlines()
    parents: dict[int, list[int]] = {}
    for row in rows:
        fields = row.split()
        if len(fields) == 2:
            parents.setdefault(int(fields[1]), []).append(int(fields[0]))
    found: list[int] = []
    pending = [root_pid]
    while pending:
        parent = pending.pop()
        for child in parents.get(parent, []):
            found.append(child)
            pending.append(child)
    return found


def _terminate_tree(root_pid: int) -> None:
    victims = _descendants(root_pid)
    for pid in reversed(victims):
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        os.kill(root_pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    time.sleep(0.5)
    for pid in [root_pid, *victims]:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def run_split(eval_seed: int, train_seeds: list[int], args: argparse.Namespace) -> dict:
    command = [
        sys.executable, "-m", "scripts.hz0c_c6_conditional_attention_eval",
        "--seed", str(eval_seed), "--train-seeds", *(str(seed) for seed in train_seeds),
        "--causal-teacher-sequences", str(args.teacher_sequences),
        "--causal-teacher-candidates", str(args.teacher_candidates),
        "--causal-teacher-blend", str(args.blend),
        "--distill-steps", str(args.distill_steps),
        "--positive-weight", str(args.positive_weight),
    ]
    process = subprocess.Popen(
        command, cwd=args.cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, start_new_session=True,
    )
    try:
        output, _ = process.communicate(timeout=args.timeout)
    except subprocess.TimeoutExpired:
        _terminate_tree(process.pid)
        process.communicate()
        raise RuntimeError(f"split eval seed {eval_seed} exceeded {args.timeout}s")
    if process.returncode != 0:
        _terminate_tree(process.pid)
        raise RuntimeError(f"split eval seed {eval_seed} failed:\n{output[-4000:]}")
    try:
        report = json.loads(output)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"split eval seed {eval_seed} did not emit strict JSON: {exc}") from exc
    policies = report["policies"]
    fixed = float(policies["fixed_periodic"]["loss"])
    hybrid = float(policies["learned_controller"]["loss"])
    return {
        "train_seeds": train_seeds,
        "eval_seed": eval_seed,
        "fixed_loss": fixed,
        "hybrid_loss": hybrid,
        "improvement": fixed - hybrid,
        "anchor_rate": float(policies["learned_controller"]["anchor_rate"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=[555, 556, 557, 558, 559])
    parser.add_argument("--train-seeds", type=int, nargs="+", default=[555, 556, 557])
    parser.add_argument("--teacher-sequences", type=int, default=8)
    parser.add_argument("--teacher-candidates", type=int, default=4)
    parser.add_argument("--blend", type=float, default=0.5)
    parser.add_argument("--distill-steps", type=int, default=300)
    parser.add_argument("--positive-weight", type=float, default=4.0)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--cwd", default=".")
    args = parser.parse_args()
    splits = []
    for eval_seed in args.eval_seeds:
        train_seeds = [seed for seed in args.train_seeds if seed != eval_seed]
        if not train_seeds:
            raise SystemExit(f"eval seed {eval_seed} has no disjoint training seeds")
        splits.append(run_split(eval_seed, train_seeds, args))
    improvements = [row["improvement"] for row in splits]
    result = {
        "protocol": {
            "controller": "linear",
            "teacher_sequences": args.teacher_sequences,
            "teacher_candidates": args.teacher_candidates,
            "blend": args.blend,
            "distill_steps": args.distill_steps,
            "positive_weight": args.positive_weight,
            "sequential": True,
            "timeout_seconds": args.timeout,
        },
        "splits": splits,
        "summary": {
            "wins": sum(value > 0.0 for value in improvements),
            "splits": len(splits),
            "mean_improvement": mean(improvements),
            "finite": all(
                value == value and abs(value) != float("inf")
                for value in improvements
            ),
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
