"""Break a training step into forward / backward / optimizer time.

Answers: where does hybrid's ~2.75x slower wall-clock throughput actually
go, and is the native Metal GDN-2 kernel actually helping?

Measures, per warmed-up step: forward ms, backward ms (forward+backward
combined minus standalone forward -- MLX's value_and_grad doesn't expose a
separate backward-only hook, so this is the standard practical estimate,
not a hardware-level split), optimizer ms, active/cache/peak memory,
tokens/sec.

Kernel-launch counts are NOT included: MLX's public Python API has no
counter for this (mx.metal.start_capture/stop_capture write an Xcode
.gputrace file for manual inspection in Instruments, which isn't something
this script can parse). If that level of detail is needed, run with
--gpu-trace to capture one and inspect it in Xcode separately.
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
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from reference.hz0a_mlx_model import HZ0AMlxModel


def build_model(architecture: str, dim: int, layers: int, heads: int, hybrid_d_ff: int, transformer_d_ff: int, native_metal: bool):
    if architecture == "transformer":
        attention = tuple(range(layers))
        d_ff = transformer_d_ff
    else:
        attention = tuple(index for index in (4, 9, 14, 19, 24, 29) if index < layers)
        d_ff = hybrid_d_ff
    model = HZ0AMlxModel(24576, dim, layers, heads, d_ff, attention, native_metal=native_metal)
    mx.eval(model.parameters())
    return model


def profile(model, batch_size: int, sequence_length: int, chunk_length: int, warmup: int, steps: int) -> dict:
    tokens = mx.random.randint(0, 24576, (batch_size, sequence_length))
    mx.eval(tokens)

    def loss_fn(current, toks):
        states = None
        parts = []
        for start in range(0, sequence_length, chunk_length):
            logits, states = current(toks[:, start:start + chunk_length], states)
            parts.append(logits)
        logits = mx.concatenate(parts, axis=1)
        return mx.mean(nn.losses.cross_entropy(logits[:, :-1], toks[:, 1:]))

    value_and_grad = nn.value_and_grad(model, loss_fn)
    optimizer = optim.AdamW(learning_rate=1e-4)

    for _ in range(warmup):
        loss, grads = value_and_grad(model, tokens)
        mx.eval(loss, grads)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

    mx.reset_peak_memory()
    forward_ms, combined_ms, optimizer_ms = [], [], []
    t_start = time.perf_counter()
    for _ in range(steps):
        t0 = time.perf_counter()
        loss = loss_fn(model, tokens)
        mx.eval(loss)
        t1 = time.perf_counter()

        loss, grads = value_and_grad(model, tokens)
        mx.eval(loss, grads)
        t2 = time.perf_counter()

        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        t3 = time.perf_counter()

        forward_ms.append((t1 - t0) * 1000)
        combined_ms.append((t2 - t1) * 1000)
        optimizer_ms.append((t3 - t2) * 1000)
    total_s = time.perf_counter() - t_start

    def stats(values):
        return {"mean_ms": sum(values) / len(values), "min_ms": min(values), "max_ms": max(values)}

    forward_stats = stats(forward_ms)
    combined_stats = stats(combined_ms)
    backward_mean_ms = combined_stats["mean_ms"] - forward_stats["mean_ms"]

    return {
        "forward_ms": forward_stats,
        "forward_plus_backward_ms": combined_stats,
        "backward_ms_estimate": max(backward_mean_ms, 0.0),
        "optimizer_ms": stats(optimizer_ms),
        "active_memory_gb": mx.get_active_memory() / 1e9,
        "cache_memory_gb": mx.get_cache_memory() / 1e9,
        "peak_memory_gb": mx.get_peak_memory() / 1e9,
        "steps_measured": steps,
        "tokens_per_step": batch_size * sequence_length,
        "tokens_per_second": (steps * batch_size * sequence_length) / total_s,
        "parameter_count": sum(v.size for _, v in tree_flatten(model.parameters())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--chunk-length", type=int, default=128)
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--layers", type=int, default=31)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--hybrid-d-ff", type=int, default=2304)
    parser.add_argument("--transformer-d-ff", type=int, default=2944)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--reference-steps", type=int, default=5, help="Steps for the slow native_metal=False reference path (Python per-token loop)")
    parser.add_argument("--output", type=Path, default=Path("outputs/stage1_profile_report.json"))
    parser.add_argument("--skip-reference", action="store_true", help="Skip the slow native_metal=False comparison")
    args = parser.parse_args()

    mx.random.seed(7)
    report = {}

    mx.random.seed(7)
    print("Profiling: transformer (all-attention)...", flush=True)
    model = build_model("transformer", args.dim, args.layers, args.heads, args.hybrid_d_ff, args.transformer_d_ff, native_metal=True)
    report["transformer"] = profile(model, args.batch_size, args.sequence_length, args.chunk_length, args.warmup, args.steps)
    del model

    mx.random.seed(7)
    print("Profiling: hybrid, native Metal GDN-2 kernel...", flush=True)
    model = build_model("hybrid", args.dim, args.layers, args.heads, args.hybrid_d_ff, args.transformer_d_ff, native_metal=True)
    report["hybrid_native_metal"] = profile(model, args.batch_size, args.sequence_length, args.chunk_length, args.warmup, args.steps)
    del model

    if not args.skip_reference:
        mx.random.seed(7)
        print(f"Profiling: hybrid, reference (native_metal=False, plain MLX per-token loop), {args.reference_steps} steps only (slow)...", flush=True)
        model = build_model("hybrid", args.dim, args.layers, args.heads, args.hybrid_d_ff, args.transformer_d_ff, native_metal=False)
        report["hybrid_reference"] = profile(model, args.batch_size, args.sequence_length, args.chunk_length, warmup=2, steps=args.reference_steps)
        del model

    print(json.dumps(report, indent=2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
