#!/usr/bin/env python3
"""Evaluate the explicit BDH-vs-Transformer training targets.

Training and inference are separate gates. This command never treats a
compile-only speedup or a different token budget as a training win.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def evaluate(candidate: dict, transformer: dict, *, ram_limit: float = 0.70, speed_floor: float = 1.30) -> dict:
    # These fields are deliberately required rather than assumed. A ratio from
    # different GPUs, batch-token counts, or compile/optimizer policies is not
    # a fair training-efficiency measurement.
    required_execution_fields = (
        "device", "hardware_id", "effective_batch_tokens", "compile_step",
        "compile_mode", "fused_optimizer",
    )
    checks = {
        "parameter_ratio": candidate["parameter_count"] / transformer["parameter_count"],
        "token_budget_equal": candidate.get("target_tokens") == transformer.get("target_tokens") and candidate.get("tokens_seen") == transformer.get("tokens_seen"),
        "dtype_equal": candidate.get("dtype") == transformer.get("dtype"),
    }
    for field in required_execution_fields:
        checks[f"{field}_equal"] = field in candidate and field in transformer and candidate[field] == transformer[field]
    checks["parameter_match"] = checks["parameter_ratio"] <= 1.01
    throughput_ratio = candidate["tokens_per_second"] / transformer["tokens_per_second"]
    time_ratio = candidate["training_seconds"] / transformer["training_seconds"]
    ram_ratio = candidate["peak_memory_bytes"] / transformer["peak_memory_bytes"]
    same_conditions = all(checks.values())
    result = {
        "candidate_parameters": candidate["parameter_count"],
        "transformer_parameters": transformer["parameter_count"],
        "parameter_ratio": checks["parameter_ratio"],
        "token_budget_equal": checks["token_budget_equal"],
        "dtype_equal": checks["dtype_equal"],
        "execution_conditions": {key: value for key, value in checks.items() if key.endswith("_equal") and key not in ("token_budget_equal", "dtype_equal")},
        "training_throughput_ratio": throughput_ratio,
        "training_time_ratio": time_ratio,
        "peak_training_ram_ratio": ram_ratio,
        "speed_gate": same_conditions and throughput_ratio >= speed_floor and time_ratio <= (1.0 / speed_floor),
        "ram_gate": same_conditions and ram_ratio <= ram_limit,
        "claim_eligible": False,
        "quality_gate": "not evaluated by this training-only report",
        "reason": "quality, seeds, contamination, and matched evaluation are separate required gates",
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate", type=Path)
    parser.add_argument("transformer", type=Path)
    parser.add_argument("--ram-limit", type=float, default=0.70)
    parser.add_argument("--speed-floor", type=float, default=1.30)
    args = parser.parse_args()
    result = evaluate(json.loads(args.candidate.read_text()), json.loads(args.transformer.read_text()), ram_limit=args.ram_limit, speed_floor=args.speed_floor)
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["speed_gate"] and result["ram_gate"] else 2)


if __name__ == "__main__":
    main()
