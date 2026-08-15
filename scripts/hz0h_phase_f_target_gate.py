#!/usr/bin/env python3
"""Evaluate the explicit BDH-vs-Transformer RAM/speed gates.

This gate deliberately evaluates execution evidence only. It refuses to call a
result a quality/superiority win because quality, seeds, and contamination
checks are separate promotion requirements.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def evaluate(report: dict, context: str, arm: str, *, ram_limit: float = 0.70, speed_floor: float = 1.30) -> dict:
    rows = report["by_context_length"]
    row = rows[str(context)] if str(context) in rows else rows[context]
    transformer = row["transformer_decode_kv_cache"]
    candidate = row[arm]
    candidate_params = report.get({
        "bdh_decode_streaming_state": "bdh_parameter_count",
        "vb_decode_streaming_state_speed_mode": "vb_parameter_count",
        "vb_decode_int8_base_delta_state_memory_mode": "vb_parameter_count",
    }[arm])
    transformer_params = report["transformer_parameter_count"]
    parameter_ratio = candidate_params / transformer_params
    ram_ratio = candidate["peak_memory_bytes"] / transformer["peak_memory_bytes"]
    speed_ratio = candidate["tokens_per_second"] / transformer["tokens_per_second"]
    parameter_match = (1.0 / 1.01) <= parameter_ratio <= 1.01
    result = {
        "context": int(context), "arm": arm,
        "candidate_parameters": candidate_params,
        "transformer_parameters": transformer_params,
        "parameter_ratio": parameter_ratio,
        "parameter_match": parameter_match,
        "peak_ram_ratio": ram_ratio,
        "decode_throughput_ratio": speed_ratio,
        "ram_gate": parameter_match and ram_ratio <= ram_limit,
        "speed_gate": parameter_match and speed_ratio >= speed_floor,
        "quality_gate": "not evaluated by this execution-only report",
        "claim_eligible": False,
        "reason": "quality, seeds, and frozen evaluation are separate required gates",
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--context", required=True, type=int)
    parser.add_argument("--arm", default="vb_decode_streaming_state_speed_mode", choices=(
        "bdh_decode_streaming_state",
        "vb_decode_streaming_state_speed_mode",
        "vb_decode_int8_base_delta_state_memory_mode",
    ))
    parser.add_argument("--ram-limit", type=float, default=0.70)
    parser.add_argument("--speed-floor", type=float, default=1.30)
    args = parser.parse_args()
    result = evaluate(json.loads(args.report.read_text()), str(args.context), args.arm, ram_limit=args.ram_limit, speed_floor=args.speed_floor)
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["ram_gate"] and result["speed_gate"] else 2)


if __name__ == "__main__":
    main()
