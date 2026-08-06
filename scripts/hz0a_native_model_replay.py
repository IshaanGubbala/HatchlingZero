"""Run a deterministic native HZ-0A replay and emit a parity report."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from restart.hz0a_pmetal.python.native_model import NativeTinyHZ0AModel
from restart.hz0a_pmetal.python.training import PmetalOptimizerPath


def fingerprint(model: NativeTinyHZ0AModel) -> str:
    return hashlib.sha256(model.flat_parameters().tobytes()).hexdigest()


def run_replay(steps: int, checkpoint: Path | None, *, moe: bool = False) -> dict:
    rng = np.random.default_rng(17)
    batches = [(rng.integers(0, 32, (2, 4), dtype=np.int64), rng.integers(0, 32, (2, 4), dtype=np.int64)) for _ in range(steps)]
    model_kwargs = dict(moe_layers=[1], moe_num_experts=2, moe_expert_d_ff=8, moe_capacity_factor=1.5) if moe else {}
    model = NativeTinyHZ0AModel(32, 16, 3, 2, 8, 8, 32, [1], seed=99, **model_kwargs)
    optimizer = PmetalOptimizerPath(model.flat_parameters(), total_steps=steps)
    started = time.perf_counter()
    for tokens, targets in batches:
        model.zero_grad()
        model.loss_and_backward(tokens, targets)
        optimizer.add_microbatch(np.concatenate([p.grad.reshape(-1) for p in model.parameters()]), tokens=tokens.size)
        model.load_flat_parameters(optimizer.state.parameters)
    full_fingerprint = fingerprint(model)
    resumed = NativeTinyHZ0AModel(32, 16, 3, 2, 8, 8, 32, [1], seed=99, **model_kwargs)
    resumed_optimizer = PmetalOptimizerPath(resumed.flat_parameters(), total_steps=steps)
    for index, (tokens, targets) in enumerate(batches):
        resumed.zero_grad(); resumed.loss_and_backward(tokens, targets)
        resumed_optimizer.add_microbatch(np.concatenate([p.grad.reshape(-1) for p in resumed.parameters()]), tokens=tokens.size)
        resumed.load_flat_parameters(resumed_optimizer.state.parameters)
        if checkpoint and index + 1 == steps // 2:
            resumed_optimizer.checkpoint(checkpoint)
    checkpoint_resume = PmetalOptimizerPath.restore(checkpoint) if checkpoint else None
    if checkpoint_resume:
        resumed_optimizer = checkpoint_resume
        resumed.load_flat_parameters(resumed_optimizer.state.parameters)
        for tokens, targets in batches[steps // 2:]:
            resumed.zero_grad(); resumed.loss_and_backward(tokens, targets)
            resumed_optimizer.add_microbatch(np.concatenate([p.grad.reshape(-1) for p in resumed.parameters()]), tokens=tokens.size)
            resumed.load_flat_parameters(resumed_optimizer.state.parameters)
    return {"steps": steps, "tokens": optimizer.state.tokens_seen, "moe": moe, "native_fingerprint": full_fingerprint, "resumed_fingerprint": fingerprint(resumed), "exact_resume": full_fingerprint == fingerprint(resumed), "finite": bool(np.isfinite(model.flat_parameters()).all()), "execution_seconds": time.perf_counter() - started}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--checkpoint", type=Path, default=Path("/tmp/hz0a_native_replay.json"))
    parser.add_argument("--moe", action="store_true", help="enable a trainable MoE block at layer 1")
    args = parser.parse_args()
    print(json.dumps(run_replay(args.steps, args.checkpoint, moe=args.moe), indent=2, sort_keys=True))
