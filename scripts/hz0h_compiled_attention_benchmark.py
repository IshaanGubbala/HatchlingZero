#!/usr/bin/env python3
"""Measure BDH training-step throughput: raw forward vs torch.compile on attention.

This probe measures whether torch.compile can fuse the three unfused ops in
Attention.forward (RoPE, scores, matmul) into fewer kernel launches, improving
real wall-clock training performance. Not a quality claim -- purely a systems
measurement of throughput (tokens/second) and memory usage for identical math.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reference.hz0h_bdh_torch import BDH, BDHConfig
from reference.hz0h_bdh_compiled_attention_torch import (
    bdh_compiled_forward,
    is_compile_available,
)


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def reset_peak(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()


def peak_memory(device: torch.device) -> int | None:
    if device.type == "cuda":
        return int(torch.cuda.max_memory_allocated())
    if device.type == "mps":
        return int(torch.mps.current_allocated_memory())
    return None


def run_variant(
    model: BDH,
    tokens: torch.Tensor,
    targets: torch.Tensor,
    *,
    variant_name: str,
    use_compiled: bool,
    warmup: int,
    steps: int,
    device: torch.device,
    optimizer: optim.Optimizer,
) -> dict:
    model.train()

    # Warmup: let torch.compile JIT and warm up caches
    for _ in range(warmup):
        optimizer.zero_grad(set_to_none=True)
        if use_compiled:
            _, loss = bdh_compiled_forward(model, tokens, targets=targets)
        else:
            _, loss = model(tokens, targets=targets)
        loss.backward()
        optimizer.step()
        sync(device)

    # Timed steps: measure real training throughput
    reset_peak(device)
    sync(device)
    started = time.perf_counter()
    last_loss = None
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        if use_compiled:
            _, loss = bdh_compiled_forward(model, tokens, targets=targets)
        else:
            _, loss = model(tokens, targets=targets)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach().cpu())
        sync(device)

    elapsed = time.perf_counter() - started
    tok_per_sec = float(tokens.numel() * steps / elapsed)

    return {
        "variant": variant_name,
        "steps": steps,
        "tokens_per_second": tok_per_sec,
        "seconds_per_step": elapsed / steps,
        "last_loss": last_loss,
        "peak_memory_bytes": peak_memory(device),
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description="Benchmark BDH training throughput: raw vs compiled attention"
    )
    p.add_argument(
        "--device", choices=("cpu", "mps", "cuda"), default="cpu"
    )
    p.add_argument("--batch-size", type=int, default=12)
    p.add_argument("--sequence-length", type=int, default=256)
    p.add_argument("--steps", type=int, default=15)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--n-embd", type=int, default=512)
    p.add_argument("--n-layer", type=int, default=8)
    p.add_argument("--n-head", type=int, default=8)
    p.add_argument(
        "--mlp-internal-dim-multiplier", type=int, default=32
    )
    p.add_argument("--vocab-size", type=int, default=256)
    p.add_argument(
        "--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16"
    )
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable")

    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]

    torch.manual_seed(42)

    cfg = BDHConfig(
        n_layer=args.n_layer,
        n_embd=args.n_embd,
        n_head=args.n_head,
        mlp_internal_dim_multiplier=args.mlp_internal_dim_multiplier,
        vocab_size=args.vocab_size,
        dropout=0.0,
    )

    # Create models
    eager_model = BDH(cfg).to(device=device, dtype=dtype)
    compiled_model = BDH(cfg).to(device=device, dtype=dtype)
    compiled_model.load_state_dict(eager_model.state_dict())

    # BF16 gotcha: freqs must stay float32
    eager_model.attn.freqs = eager_model.attn.freqs.to(torch.float32)
    compiled_model.attn.freqs = compiled_model.attn.freqs.to(torch.float32)

    # Input data
    tokens = torch.randint(
        0, args.vocab_size,
        (args.batch_size, args.sequence_length),
        device=device,
    )
    targets = torch.randint(
        0, args.vocab_size,
        (args.batch_size, args.sequence_length),
        device=device,
    )

    # Optimizers (separate for each model to keep state independent)
    eager_optim = optim.AdamW(eager_model.parameters(), lr=1e-3)
    compiled_optim = optim.AdamW(compiled_model.parameters(), lr=1e-3)

    # Run benchmarks
    result = {
        "architecture": "BDH_compiled_attention_probe",
        "device": str(device),
        "dtype": args.dtype,
        "parameter_count": sum(p.numel() for p in eager_model.parameters()),
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "torch_compile_available": is_compile_available(),
        "trained_weights": False,
    }

    # Run eager variant
    print("Running eager (unfused) attention baseline...", file=sys.stderr)
    result["eager"] = run_variant(
        eager_model,
        tokens,
        targets,
        variant_name="eager",
        use_compiled=False,
        warmup=args.warmup,
        steps=args.steps,
        device=device,
        optimizer=eager_optim,
    )

    # Run compiled variant only if available
    if is_compile_available():
        print("Running compiled attention variant...", file=sys.stderr)
        result["compiled"] = run_variant(
            compiled_model,
            tokens,
            targets,
            variant_name="compiled",
            use_compiled=True,
            warmup=args.warmup,
            steps=args.steps,
            device=device,
            optimizer=compiled_optim,
        )
        result["speedup_ratio_compiled_over_eager"] = (
            result["compiled"]["tokens_per_second"]
            / result["eager"]["tokens_per_second"]
        )
    else:
        result["compiled"] = None
        result["speedup_ratio_compiled_over_eager"] = None
        result["note"] = "torch.compile not available on this platform"

    output = json.dumps(result, indent=2, sort_keys=True)
    print(output)
    if args.output:
        args.output.write_text(output + "\n")


if __name__ == "__main__":
    main()
