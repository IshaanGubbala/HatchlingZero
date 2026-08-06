#!/usr/bin/env python3
"""Benchmark the frozen-backbone MoE trainer with reproducible settings.

The benchmark compares the original reference path with frozen-prefix
caching and optional MLX gradient compilation. It reports JSON so training
automation can track throughput without treating a microbenchmark as a
quality result.
"""
from __future__ import annotations

import argparse
import json
import time

import mlx.core as mx

from reference.hz0e_e3_routing_objectives import train_moe_layer
from reference.hz0e_moe_contract import MoeConfig
from scripts.hz0b_b11_baseline_comparison import load_frozen_model
from scripts.hz0c_c3_trigger_simulator import load_real_sequences


def run(steps: int, seq_len: int, seed: int, cache_backbone: bool, compile_step: bool, record_history: bool = True, eval_interval: int = 1) -> dict:
    model, _ = load_frozen_model()
    sequences = load_real_sequences("data/packed/repro_1024_train.jsonl", steps)
    batches = [mx.array([tokens[:seq_len]]) for tokens in sequences]
    config = MoeConfig()
    started = time.perf_counter()
    train_moe_layer(
        model,
        batches,
        config,
        learning_rate=1e-5,
        init_seed=seed,
        cache_backbone=cache_backbone,
        compile_step=compile_step,
        record_history=record_history,
        eval_interval=eval_interval,
    )
    elapsed = time.perf_counter() - started
    return {
        "steps": steps,
        "seq_len": seq_len,
        "seed": seed,
        "cache_backbone": cache_backbone,
        "compile_step": compile_step,
        "record_history": record_history,
        "eval_interval": eval_interval,
        "elapsed_seconds": elapsed,
        "steps_per_second": steps / elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--mode", choices=("baseline", "cached", "compiled", "fast", "all"), default="all")
    args = parser.parse_args()
    modes = {
        "baseline": (False, False),
        "cached": (True, False),
        "compiled": (True, True),
        "fast": (True, True, False, 8),
    }
    selected = modes.keys() if args.mode == "all" else (args.mode,)
    results = [run(args.steps, args.seq_len, args.seed, *modes[name]) for name in selected]
    baseline = next((item for item in results if item["cache_backbone"] is False), None)
    if baseline is not None:
        baseline_rate = baseline["steps_per_second"]
        for item in results:
            item["speedup_vs_baseline"] = item["steps_per_second"] / baseline_rate
    print(json.dumps({"benchmarks": results}))


if __name__ == "__main__":
    main()
