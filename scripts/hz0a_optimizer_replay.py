#!/usr/bin/env python3
"""Run the deterministic A9 optimizer replay contract on a tiny parameter set."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from restart.hz0a_pmetal.python.pmetal_reference import adamw_step


def run_replay(*, seed: int, steps: int, learning_rate: float) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    parameters = rng.normal(0.0, 0.02, size=32).astype(np.float64)
    target = rng.normal(0.0, 0.02, size=32).astype(np.float64)
    batch_offsets = rng.normal(0.0, 0.002, size=(8, 32)).astype(np.float64)
    state = None
    metrics = []
    for step in range(steps):
        batch_index = step % len(batch_offsets)
        desired = target + batch_offsets[batch_index]
        gradient = parameters - desired
        loss = float(0.5 * np.mean(np.square(gradient)))
        result = adamw_step(parameters, gradient, state, learning_rate=learning_rate, weight_decay=0.01)
        state = result.state
        parameters = result.parameters
        metrics.append({
            "step": step + 1,
            "batch_index": batch_index,
            "loss": loss,
            "gradient_norm": float(np.linalg.norm(gradient)),
            "update_norm": result.update_norm,
            "parameter_norm": float(np.linalg.norm(parameters)),
        })
    payload = {
        "seed": seed,
        "steps": steps,
        "learning_rate": learning_rate,
        "batch_order": [metric["batch_index"] for metric in metrics],
        "metrics": metrics,
        "final_parameter_sha256": hashlib.sha256(parameters.tobytes()).hexdigest(),
        "final_loss": metrics[-1]["loss"] if metrics else None,
        "stable_finite": bool(np.isfinite(parameters).all() and all(np.isfinite(metric["loss"]) for metric in metrics)),
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the HZ-0A deterministic optimizer replay.")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    payload = run_replay(seed=args.seed, steps=args.steps, learning_rate=args.learning_rate)
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
