"""Emit machine-readable evidence for the tiny native HZ-0E MoE graph."""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from restart.hz0a_pmetal.python.native_model import NativeTinyHZ0AModel
from restart.hz0a_pmetal.python.training import PmetalOptimizerPath


def run(steps: int = 1) -> dict:
    if steps <= 0:
        raise ValueError("steps must be positive")
    started = time.perf_counter()
    model = NativeTinyHZ0AModel(
        32, 16, 3, 2, 8, 8, 32, [1], seed=99,
        moe_layers=[1], moe_num_experts=2, moe_expert_d_ff=8,
        moe_capacity_factor=0.5,
    )
    optimizer = PmetalOptimizerPath(model.flat_parameters(), total_steps=steps)
    rng = np.random.default_rng(17)
    losses = []
    overflow_counts = []
    for _ in range(steps):
        tokens = rng.integers(0, 32, (2, 4), dtype=np.int64)
        targets = rng.integers(0, 32, (2, 4), dtype=np.int64)
        model.zero_grad()
        loss, _ = model.loss_and_backward(tokens, targets)
        losses.append(float(loss))
        overflow_counts.append(int(model.blocks[1].mlp._cache[4].sum()))
        gradient = np.concatenate([parameter.grad.reshape(-1) for parameter in model.parameters()])
        optimizer_metrics = optimizer.add_microbatch(gradient, tokens=tokens.size)
        model.load_flat_parameters(optimizer.state.parameters)
    values = model.flat_parameters()
    return {
        "steps": steps,
        "tokens": steps * 8,
        "losses": losses,
        "overflow_counts": overflow_counts,
        "optimizer_metrics": optimizer_metrics,
        "parameter_fingerprint": model.parameter_fingerprint(),
        "finite": bool(np.isfinite(values).all()),
        "gradient_finite": bool(np.isfinite(gradient).all()),
        "execution_seconds": time.perf_counter() - started,
        "peak_memory_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1)
    args = parser.parse_args()
    print(json.dumps(run(args.steps), indent=2, sort_keys=True))
