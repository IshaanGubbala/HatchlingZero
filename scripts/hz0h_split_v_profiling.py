#!/usr/bin/env python3
"""Profile BDHSplitV vs exact BDH to locate the performance bottleneck.

Real motivation (see reference/hz0h_bdh_split_v_torch.py's docstring):
BDHSplitV is ~18% slower than exact BDH at the same config (3180.5 vs
3898.1 tok/s on Mac MPS, n_head=8), despite the math predicting ~8x fewer
FLOPs in the attention step specifically (narrower V per-head). This script
uses torch.profiler to break down which operations account for the extra
time in BDHSplitV and identify what, if anything, can be optimized.

Measures:
- Forward+backward+optimizer loop throughput (tok/s) for both models
- Per-operation timing breakdown via torch.profiler
- Top 15 most-expensive operations in each, for side-by-side comparison

Config: n_embd=512, n_layer=8, n_head=8, mlp_internal_dim_multiplier=32,
vocab_size=256, dropout=0.0, batch=12, seq=256, bf16 (with freqs in float32
per the known gotcha).

Warmup: 5 steps (to stabilize GPU/memory state)
Timed: 15 steps (to average over for stable tok/s measurement)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reference.hz0h_bdh_torch import BDH, BDHConfig
from reference.hz0h_bdh_split_v_torch import BDHSplitV, BDHSplitVConfig


def resolve_device() -> torch.device:
    """Auto-select best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync(device: torch.device) -> None:
    """Synchronize device to ensure timing includes all async work."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def run_training_step(
    model: torch.nn.Module,
    tokens: torch.Tensor,
    targets: torch.Tensor,
    optimizer: torch.optim.Optimizer,
) -> float:
    """One forward-backward-step cycle. Returns loss value."""
    model.zero_grad(set_to_none=True)
    logits, loss = model(tokens, targets=targets)
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu())


def profile_model(
    model: torch.nn.Module,
    tokens: torch.Tensor,
    targets: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    model_name: str,
    warmup: int = 5,
    timed_steps: int = 15,
    profile_steps: int = 10,
) -> dict[str, Any]:
    """Profile a model's training throughput and operation timing.

    Returns a dict with:
    - tokens_per_second: throughput in tok/s
    - model_name: identifier
    - device: device name
    - dtype: model dtype
    - profile_results: top 15 ops by self CPU time (from torch.profiler)
    """
    model.train()

    # Warmup
    for _ in range(warmup):
        run_training_step(model, tokens, targets, optimizer)
        sync(device)

    # Time timed_steps to measure throughput
    sync(device)
    t_start = time.perf_counter()
    for _ in range(timed_steps):
        run_training_step(model, tokens, targets, optimizer)
        sync(device)
    elapsed = time.perf_counter() - t_start
    tokens_per_second = float(tokens.numel() * timed_steps / elapsed)

    # Profile with torch.profiler over profile_steps
    # (separate from the timed loop to avoid profiler overhead affecting results)
    sync(device)

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        record_shapes=True,
        with_stack=False,
    ) as prof:
        for _ in range(profile_steps):
            run_training_step(model, tokens, targets, optimizer)
            sync(device)

    # Extract top-15-by-self-cpu-time ops
    key_averages = prof.key_averages()
    # Sort by self_cpu_time_total (self time, not including sub-ops)
    key_averages_sorted = sorted(
        key_averages, key=lambda x: x.self_cpu_time_total, reverse=True
    )

    top_ops = []
    for i, op in enumerate(key_averages_sorted[:15]):
        op_dict = {
            "rank": i + 1,
            "op_name": op.key,
            "self_cpu_time_ms": op.self_cpu_time_total / 1000.0,
            "cpu_time_ms": op.cpu_time_total / 1000.0,
            "count": op.count,
        }
        if device.type == "cuda":
            # PyTorch renamed CUDA-specific profiler fields to be
            # device-agnostic in recent versions (cuda_time_total ->
            # device_time_total). Real bug caught running this on CUDA
            # (PyTorch 2.7.1+cu118 only has device_time_total) -- try the
            # new name first, fall back to the old one for older installs.
            op_dict["cuda_time_ms"] = getattr(op, "device_time_total", getattr(op, "cuda_time_total", 0.0)) / 1000.0
        top_ops.append(op_dict)

    return {
        "model_name": model_name,
        "device": str(device),
        "dtype": str(next(model.parameters()).dtype),
        "tokens_per_second": tokens_per_second,
        "seconds_per_step": elapsed / timed_steps,
        "top_15_ops": top_ops,
    }


def main() -> None:
    device = resolve_device()
    print(f"Device: {device}", flush=True)

    # Fixed config per task requirements
    torch.manual_seed(42)
    config = BDHConfig(
        n_layer=8,
        n_embd=512,
        n_head=8,
        mlp_internal_dim_multiplier=32,
        vocab_size=256,
        dropout=0.0,
    )

    # Batch and sequence size
    batch_size = 12
    seq_length = 256

    # Prepare models
    print("Building BDH (vanilla)...", flush=True)
    bdh_model = BDH(config).to(device=device, dtype=torch.bfloat16)
    bdh_model.attn.freqs = bdh_model.attn.freqs.to(torch.float32)  # Known gotcha
    bdh_optimizer = torch.optim.AdamW(bdh_model.parameters(), lr=1e-4)

    print("Building BDHSplitV...", flush=True)
    split_v_config = BDHSplitVConfig(
        n_layer=8,
        n_embd=512,
        n_head=8,
        mlp_internal_dim_multiplier=32,
        vocab_size=256,
        dropout=0.0,
    )
    split_v_model = BDHSplitV(split_v_config).to(device=device, dtype=torch.bfloat16)
    split_v_model.attn.freqs = split_v_model.attn.freqs.to(torch.float32)  # Known gotcha
    split_v_optimizer = torch.optim.AdamW(split_v_model.parameters(), lr=1e-4)

    # Synthetic data
    tokens = torch.randint(0, config.vocab_size, (batch_size, seq_length), device=device)
    targets = torch.randint(0, config.vocab_size, (batch_size, seq_length), device=device)

    print("\nProfiling BDH (vanilla)...", flush=True)
    bdh_results = profile_model(
        bdh_model,
        tokens,
        targets,
        bdh_optimizer,
        device,
        model_name="BDH",
        warmup=5,
        timed_steps=15,
        profile_steps=10,
    )

    print("\nProfiling BDHSplitV...", flush=True)
    split_v_results = profile_model(
        split_v_model,
        tokens,
        targets,
        split_v_optimizer,
        device,
        model_name="BDHSplitV",
        warmup=5,
        timed_steps=15,
        profile_steps=10,
    )

    # Summary
    bdh_tok_s = bdh_results["tokens_per_second"]
    split_v_tok_s = split_v_results["tokens_per_second"]
    ratio = split_v_tok_s / bdh_tok_s

    print("\n" + "=" * 80)
    print("PROFILING RESULTS")
    print("=" * 80)
    print(f"\nBDH (vanilla):          {bdh_tok_s:.1f} tok/s")
    print(f"BDHSplitV:              {split_v_tok_s:.1f} tok/s")
    print(f"Ratio (SplitV / BDH):   {ratio:.3f}x")
    slowdown_pct = (1 - ratio) * 100 if ratio < 1 else (ratio - 1) * 100
    if ratio < 1:
        print(f"BDHSplitV is {slowdown_pct:.1f}% SLOWER")
    else:
        print(f"BDHSplitV is {slowdown_pct:.1f}% FASTER")

    print("\n" + "=" * 80)
    print("TOP 15 OPERATIONS BY SELF-CPU TIME (BDH)")
    print("=" * 80)
    for op in bdh_results["top_15_ops"]:
        print(
            f"{op['rank']:2d}. {op['op_name']:50s} "
            f"self={op['self_cpu_time_ms']:8.2f}ms count={op['count']:6d}"
        )

    print("\n" + "=" * 80)
    print("TOP 15 OPERATIONS BY SELF-CPU TIME (BDHSplitV)")
    print("=" * 80)
    for op in split_v_results["top_15_ops"]:
        print(
            f"{op['rank']:2d}. {op['op_name']:50s} "
            f"self={op['self_cpu_time_ms']:8.2f}ms count={op['count']:6d}"
        )

    # Detailed JSON output for analysis
    output = {
        "config": {
            "n_layer": config.n_layer,
            "n_embd": config.n_embd,
            "n_head": config.n_head,
            "mlp_internal_dim_multiplier": config.mlp_internal_dim_multiplier,
            "vocab_size": config.vocab_size,
            "dropout": config.dropout,
            "batch_size": batch_size,
            "sequence_length": seq_length,
            "dtype": "bfloat16",
        },
        "bdh": bdh_results,
        "split_v": split_v_results,
        "summary": {
            "bdh_tok_s": bdh_tok_s,
            "split_v_tok_s": split_v_tok_s,
            "ratio_split_v_over_bdh": ratio,
            "slowdown_percentage": slowdown_pct if ratio < 1 else None,
            "speedup_percentage": slowdown_pct if ratio > 1 else None,
        },
    }

    output_path = Path(__file__).resolve().parent.parent / "profiling_output.json"
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    print(f"\nFull results saved to: {output_path}")


if __name__ == "__main__":
    main()
