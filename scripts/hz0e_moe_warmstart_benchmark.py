#!/usr/bin/env python3
"""Reproducible benchmark for HZ-0E supervised router warm-starts."""
from __future__ import annotations

import argparse
import json
import time

import mlx.core as mx

from reference.hz0e_e3_routing_objectives import supervised_warm_start
from reference.hz0e_e8_curriculum import DOMAIN_TO_EXPERT, TRAIN_DOMAIN_DATA_PATHS, load_domain_batches
from reference.hz0e_e6_integration import init_e6_layers
from reference.hz0e_moe_contract import MoeConfig
from scripts.hz0b_b11_baseline_comparison import load_frozen_model


def run(steps: int, seed: int, cache_backbone: bool, compile_step: bool) -> dict:
    model, _ = load_frozen_model()
    domains = load_domain_batches(TRAIN_DOMAIN_DATA_PATHS, count=8, seq_len=64)
    initial = init_e6_layers(model, seed=seed)[27]
    started = time.perf_counter()
    supervised_warm_start(
        model,
        domains,
        DOMAIN_TO_EXPERT,
        MoeConfig(),
        layer_index=27,
        steps=steps,
        learning_rate=1e-3,
        start_params=initial,
        cache_backbone=cache_backbone,
        compile_step=compile_step,
    )
    elapsed = time.perf_counter() - started
    return {
        "steps": steps,
        "seed": seed,
        "cache_backbone": cache_backbone,
        "compile_step": compile_step,
        "elapsed_seconds": elapsed,
        "steps_per_second": steps / elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", choices=("baseline", "cached", "compiled", "all"), default="all")
    args = parser.parse_args()
    modes = {
        "baseline": (False, False),
        "cached": (True, False),
        "compiled": (True, True),
    }
    selected = modes.keys() if args.mode == "all" else (args.mode,)
    results = [run(args.steps, args.seed, *modes[name]) for name in selected]
    baseline = next((item for item in results if not item["cache_backbone"]), None)
    if baseline is not None:
        rate = baseline["steps_per_second"]
        for item in results:
            item["speedup_vs_baseline"] = item["steps_per_second"] / rate
    print(json.dumps({"benchmarks": results}))


if __name__ == "__main__":
    main()
