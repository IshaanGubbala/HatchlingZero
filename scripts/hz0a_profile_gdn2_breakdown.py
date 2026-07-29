"""Fine-grained forward/backward breakdown of a single GDN2 block, isolated
from the other 30 layers -- answers which specific sub-step (projection,
recurrence, output projection) actually dominates hybrid's per-block cost,
not just the whole-model forward/backward split hz0a_profile_training_step.py
already gives.

Caveat, stated up front: inserting mx.eval() between sub-steps to get a
real per-stage timing is itself a synchronization point that MLX's lazy
evaluation would not otherwise take -- each of these numbers therefore
includes a small amount of forced-sync overhead the fully-lazy end-to-end
path doesn't pay. Useful for relative comparison between sub-steps, not
as an exact sum-to-total decomposition (the parts will not add up exactly
to the whole-block time measured without inserted evals).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx
import mlx.nn as nn

from reference.hz0a_mlx_model import GDN2
from reference.hz0a_mlx_metal import native_gdn2_forward_differentiable


def time_it(fn, warmup: int, steps: int) -> dict:
    for _ in range(warmup):
        mx.eval(fn())
    times = []
    for _ in range(steps):
        t0 = time.perf_counter()
        mx.eval(fn())
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return {"mean_ms": sum(times) / len(times), "median_ms": times[len(times) // 2], "min_ms": times[0], "max_ms": times[-1]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--chunk-length", type=int, default=128)
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--output", type=Path, default=Path("outputs/gdn2_block_breakdown.json"))
    args = parser.parse_args()

    mx.random.seed(7)
    block = GDN2(args.dim, args.heads, native_metal=True)
    mx.eval(block.parameters())
    x = mx.random.normal((args.batch_size, args.chunk_length, args.dim))
    mx.eval(x)
    head_dim = args.dim // args.heads

    # ---- forward sub-steps ----
    def in_proj_only():
        return block.in_proj(x)

    projected = block.in_proj(x)
    mx.eval(projected)

    def unpack_only():
        q, k, v, d, e, w = mx.split(projected.reshape(args.batch_size, args.chunk_length, 6, args.heads, head_dim), 6, axis=2)
        return mx.squeeze(q, axis=2) + mx.squeeze(k, axis=2) + mx.squeeze(v, axis=2) + mx.squeeze(d, axis=2) + mx.squeeze(e, axis=2) + mx.squeeze(w, axis=2)

    q, k, v, d, e, w = mx.split(projected.reshape(args.batch_size, args.chunk_length, 6, args.heads, head_dim), 6, axis=2)
    q, k, v, d, e, w = (mx.squeeze(item, axis=2) for item in (q, k, v, d, e, w))
    mx.eval(q, k, v, d, e, w)
    initial = mx.zeros((args.batch_size, args.heads, head_dim, head_dim))
    mx.eval(initial)

    def recurrence_forward_only():
        output, _ = native_gdn2_forward_differentiable(q, k, v, d, e, w, initial)
        return output

    mixed, _ = native_gdn2_forward_differentiable(q, k, v, d, e, w, initial)
    mx.eval(mixed)
    mixed_flat = mixed.reshape(args.batch_size, args.chunk_length, args.dim)
    mx.eval(mixed_flat)

    def out_proj_only():
        return block.out(mixed_flat)

    def full_block_forward():
        out, _ = block(x, None)
        return out

    # ---- backward sub-steps ----
    def recurrence_forward_and_backward():
        def loss(q_, k_, v_, d_, e_, w_):
            out, state = native_gdn2_forward_differentiable(q_, k_, v_, d_, e_, w_, initial)
            return mx.sum(out) + mx.sum(state)
        _, grads = mx.value_and_grad(loss, argnums=(0, 1, 2, 3, 4, 5))(q, k, v, d, e, w)
        return grads[0]

    def block_loss(current, inp):
        out, _ = current(inp, None)
        return mx.sum(out)
    block_value_and_grad = nn.value_and_grad(block, block_loss)

    def full_block_forward_and_backward():
        _, grads = block_value_and_grad(block, x)
        return grads

    report = {
        "config": {"batch_size": args.batch_size, "chunk_length": args.chunk_length, "dim": args.dim, "heads": args.heads},
        "in_proj_forward": time_it(in_proj_only, args.warmup, args.steps),
        "unpack": time_it(unpack_only, args.warmup, args.steps),
        "recurrence_forward_only": time_it(recurrence_forward_only, args.warmup, args.steps),
        "out_proj_forward": time_it(out_proj_only, args.warmup, args.steps),
        "full_block_forward": time_it(full_block_forward, args.warmup, args.steps),
        "recurrence_forward_and_backward": time_it(recurrence_forward_and_backward, args.warmup, args.steps),
        "full_block_forward_and_backward": time_it(full_block_forward_and_backward, args.warmup, args.steps),
    }
    print(json.dumps(report, indent=2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
