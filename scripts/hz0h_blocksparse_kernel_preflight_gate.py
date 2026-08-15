#!/usr/bin/env python3
"""Gate an untrained sparse chunk_gla CUDA preflight conservatively.

Passing means only that the kernel deserves a trained quality run. It never
sets claim_eligible true and is not a substitute for the training target gate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def evaluate(report: dict, *, speed_floor: float = 1.30, ram_limit: float = 0.70,
             max_logit_difference: float = 0.10, max_loss_difference: float = 0.05,
             max_gradient_relative_l2_difference: float = 0.05) -> dict:
    numerical = report.get("numerical_preflight", {})
    fused, transformer = report.get("chunk_gla", {}), report.get("matched_rope_transformer", {})
    checks = {
        "cuda_report": report.get("device") == "cuda",
        "parameter_match": report.get("parameter_ratio_to_transformer", float("inf")) <= 1.01,
        "raw_fused_logits": numerical.get("max_logit_difference", float("inf")) <= max_logit_difference,
        "raw_fused_loss": numerical.get("loss_difference", float("inf")) <= max_loss_difference,
        "raw_fused_encoder_gradient": numerical.get("encoder_gradient_relative_l2_difference", float("inf")) <= max_gradient_relative_l2_difference,
        "finite_numerical_gradients": numerical.get("encoder_gradients_finite") is True,
        "finite_fused_step": fused.get("finite_loss") is True and fused.get("finite_gradients") is True,
        "finite_transformer_step": transformer.get("finite_loss") is True and transformer.get("finite_gradients") is True,
        "speed": report.get("chunk_gla_over_transformer_speed_ratio", 0.0) >= speed_floor,
        "ram": report.get("chunk_gla_over_transformer_peak_memory_ratio", float("inf")) <= ram_limit,
    }
    return {
        "checks": checks,
        "kernel_preflight_pass": all(checks.values()),
        "claim_eligible": False,
        "reason": "untrained fixed-route kernel screen only; trained quality, full budget, seeds, and contamination-controlled evaluation remain mandatory",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--speed-floor", type=float, default=1.30)
    parser.add_argument("--ram-limit", type=float, default=0.70)
    parser.add_argument("--max-logit-difference", type=float, default=0.10)
    parser.add_argument("--max-loss-difference", type=float, default=0.05)
    parser.add_argument("--max-gradient-relative-l2-difference", type=float, default=0.05)
    args = parser.parse_args()
    result = evaluate(json.loads(args.report.read_text()), speed_floor=args.speed_floor, ram_limit=args.ram_limit,
        max_logit_difference=args.max_logit_difference, max_loss_difference=args.max_loss_difference,
        max_gradient_relative_l2_difference=args.max_gradient_relative_l2_difference)
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["kernel_preflight_pass"] else 2)


if __name__ == "__main__":
    main()
