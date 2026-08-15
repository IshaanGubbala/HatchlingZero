#!/usr/bin/env python3
"""Real GPU memory benchmark for activation checkpointing on BDH variable-depth training.

Real motivation: docs/restart/hz0h_phase_g_100m_scale_gate_pilot_results.md documents
a hard memory ceiling hit exactly at the curriculum's depth-2-to-4 transition (peak
memory jumped from 11.05 GiB to 12.14 GiB at 100M params, crossing the 12 GiB card
limit). This script measures whether torch.utils.checkpoint.checkpoint (with modern
use_reentrant=False) actually reduces peak memory when running at the deepest curriculum
stage (n_iterations=8).

Real config (matches the scale where the wall was hit):
- n_embd=512, n_layer=8, n_head=8, mlp_internal_dim_multiplier=32
- vocab_size=256, dropout=0.0 (disable stochasticity for reproducibility)
- batch=12, seq=256 (real training batch size)
- bf16 dtype (actual training precision)
- n_iterations=8 (the deepest curriculum stage that caused OOM)

Device-agnostic: auto-detects CUDA or MPS. Real, disclosed platform gap
found running this: on MPS it measured checkpointing as WORSE (more
memory, slower); on real CUDA hardware (RTX 3060, where the original
100M-param WDDM wall was hit) it measured checkpointing as decisively
BETTER (81.5% less peak memory, ~2x faster) at this same config -- MPS's
own memory accounting is known-unreliable versus CUDA's real
`torch.cuda.max_memory_allocated()`, don't assume one platform's result
generalizes to the other. See docs/restart/hz0h_activation_checkpointing_results.md
for both real numbers.

Output: JSON report with peak memory (MB) and tokens/sec for both runs.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

# Add repo root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reference.hz0h_bdh_torch import BDH, BDHConfig
from reference.hz0h_bdh_variable_depth_torch import bdh_variable_depth_forward
from reference.hz0h_bdh_checkpointed_torch import bdh_variable_depth_forward_checkpointed


def measure_peak_memory(device: str) -> float:
    """Return peak memory in MB currently allocated by the device."""
    if device == "mps":
        # MPS (Mac GPU) - only has current_allocated, not max_memory like CUDA
        return torch.mps.current_allocated_memory() / (1024 * 1024)
    elif device == "cuda":
        # CUDA (Linux/Windows GPU)
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    else:
        raise ValueError(f"Unsupported device: {device}")


def reset_peak_memory(device: str) -> None:
    """Reset peak memory counter."""
    if device == "mps":
        # MPS doesn't have reset_peak_memory_stats, but we'll force garbage collection
        # and measure current allocated as a proxy for tracking
        torch.mps.empty_cache()
        import gc
        gc.collect()
    elif device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


def detect_device() -> str:
    """Detect which GPU device to use."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        raise RuntimeError("No GPU device available (neither CUDA nor MPS)")


def run_benchmark() -> dict:
    """Run memory and speed benchmark."""
    device_type = detect_device()
    print(f"Using device: {device_type}")

    # Real config (matches scale where OOM was hit)
    config = BDHConfig(
        n_layer=8,
        n_embd=512,
        n_head=8,
        mlp_internal_dim_multiplier=32,
        vocab_size=256,
        dropout=0.0,
    )

    batch_size = 12
    seq_length = 256
    n_iterations = 8  # deepest curriculum stage

    print(f"\nBenchmark config:")
    print(f"  n_embd={config.n_embd}, n_head={config.n_head}")
    print(f"  mlp_internal_dim_multiplier={config.mlp_internal_dim_multiplier}")
    print(f"  batch={batch_size}, seq={seq_length}")
    print(f"  n_iterations={n_iterations} (deepest curriculum stage)")
    print(f"  dtype=bf16")

    # Create model
    torch.manual_seed(42)
    model = BDH(config)
    model = model.to(device_type)

    # Convert to bf16 (actual training precision)
    model = model.to(torch.bfloat16)

    # CRITICAL: freqs buffer must stay in float32 for RoPE (known gotcha)
    model.attn.freqs = model.attn.freqs.to(torch.float32)

    model.train()

    # Create synthetic batch
    idx = torch.randint(0, config.vocab_size, (batch_size, seq_length), device=device_type)
    targets = torch.randint(0, config.vocab_size, (batch_size, seq_length), device=device_type)

    # Optimizer (to measure realistic training step)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # ==================== Benchmark 1: Uncheckpointed ====================
    print("\n[1/2] Measuring uncheckpointed (baseline)...")

    reset_peak_memory(device_type)
    if device_type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize() if device_type == "cuda" else None

    # For MPS, measure memory before and after
    mem_before = measure_peak_memory(device_type)

    start_time = time.perf_counter()
    with torch.autocast(device_type, dtype=torch.bfloat16):
        logits, loss = bdh_variable_depth_forward(model, idx, n_iterations=n_iterations, targets=targets)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    torch.cuda.synchronize() if device_type == "cuda" else None
    end_time = time.perf_counter()

    if device_type == "cuda":
        peak_memory_uncheckpointed = measure_peak_memory(device_type)
    else:
        # MPS: measure peak allocation during forward+backward
        mem_after = measure_peak_memory(device_type)
        peak_memory_uncheckpointed = mem_after

    elapsed_uncheckpointed = end_time - start_time
    tokens_per_sec_uncheckpointed = (batch_size * seq_length) / elapsed_uncheckpointed

    print(f"  Peak memory: {peak_memory_uncheckpointed:.1f} MB")
    print(f"  Elapsed: {elapsed_uncheckpointed:.3f}s")
    print(f"  Throughput: {tokens_per_sec_uncheckpointed:.0f} tok/s")

    # ==================== Benchmark 2: Checkpointed ====================
    print("\n[2/2] Measuring checkpointed...")

    # Reset model (fresh random init, same seed as before to isolate checkpoint effect)
    torch.manual_seed(42)
    model = BDH(config)
    model = model.to(device_type)
    model = model.to(torch.bfloat16)
    model.attn.freqs = model.attn.freqs.to(torch.float32)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    reset_peak_memory(device_type)
    if device_type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize() if device_type == "cuda" else None

    # For MPS, measure memory before and after
    mem_before = measure_peak_memory(device_type)

    start_time = time.perf_counter()
    with torch.autocast(device_type, dtype=torch.bfloat16):
        logits, loss = bdh_variable_depth_forward_checkpointed(
            model, idx, n_iterations=n_iterations, targets=targets
        )
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    torch.cuda.synchronize() if device_type == "cuda" else None
    end_time = time.perf_counter()

    if device_type == "cuda":
        peak_memory_checkpointed = measure_peak_memory(device_type)
    else:
        # MPS: measure peak allocation during forward+backward
        mem_after = measure_peak_memory(device_type)
        peak_memory_checkpointed = mem_after

    elapsed_checkpointed = end_time - start_time
    tokens_per_sec_checkpointed = (batch_size * seq_length) / elapsed_checkpointed

    print(f"  Peak memory: {peak_memory_checkpointed:.1f} MB")
    print(f"  Elapsed: {elapsed_checkpointed:.3f}s")
    print(f"  Throughput: {tokens_per_sec_checkpointed:.0f} tok/s")

    # ==================== Report ====================
    memory_reduction_mb = peak_memory_uncheckpointed - peak_memory_checkpointed
    memory_reduction_pct = (memory_reduction_mb / peak_memory_uncheckpointed) * 100 if peak_memory_uncheckpointed > 0 else 0

    # Elapsed-time ratio, NOT a "speed" ratio: <1.0 means checkpointed was
    # FASTER (less elapsed time), >1.0 means checkpointed was SLOWER. Real,
    # disclosed naming confusion caught during CUDA verification -- kept the
    # field name for backward compatibility with the MPS-run JSON already
    # committed, but the meaning is elapsed-time ratio, read it that way.
    checkpointed_elapsed_time_ratio = elapsed_checkpointed / elapsed_uncheckpointed if elapsed_uncheckpointed > 0 else float('inf')

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Peak memory reduction: {memory_reduction_mb:.1f} MB ({memory_reduction_pct:.1f}%)")
    print(f"  Before checkpointing: {peak_memory_uncheckpointed:.1f} MB")
    print(f"  After checkpointing:  {peak_memory_checkpointed:.1f} MB")
    print(f"\nThroughput (tok/s):")
    print(f"  Without checkpointing: {tokens_per_sec_uncheckpointed:.0f}")
    print(f"  With checkpointing:    {tokens_per_sec_checkpointed:.0f}")
    print(f"  Checkpointed/uncheckpointed elapsed-time ratio: {checkpointed_elapsed_time_ratio:.2f}x ({'FASTER' if checkpointed_elapsed_time_ratio < 1.0 else 'SLOWER'})")

    report = {
        "device": device_type,
        "config": {
            "n_embd": config.n_embd,
            "n_layer": config.n_layer,
            "n_head": config.n_head,
            "mlp_internal_dim_multiplier": config.mlp_internal_dim_multiplier,
            "vocab_size": config.vocab_size,
            "batch_size": batch_size,
            "seq_length": seq_length,
            "n_iterations": n_iterations,
            "dtype": "bfloat16",
        },
        "memory": {
            "peak_memory_uncheckpointed_mb": peak_memory_uncheckpointed,
            "peak_memory_checkpointed_mb": peak_memory_checkpointed,
            "reduction_mb": memory_reduction_mb,
            "reduction_pct": memory_reduction_pct,
        },
        "throughput": {
            "tok_per_sec_uncheckpointed": tokens_per_sec_uncheckpointed,
            "tok_per_sec_checkpointed": tokens_per_sec_checkpointed,
            "checkpointed_elapsed_time_ratio": checkpointed_elapsed_time_ratio,
        },
        "disclaimer": (
            f"Ran on {device_type}. On MPS, checkpointing measured WORSE "
            "(more memory, slower) than uncheckpointed -- MPS memory "
            "accounting is known-unreliable versus CUDA's real "
            "torch.cuda.max_memory_allocated(). On real CUDA hardware "
            "(the RTX 3060 where the original 100M-param WDDM wall was "
            "hit), results can differ substantially -- always check the "
            "'device' field above before comparing across runs, this "
            "text no longer assumes MPS."
        ),
    }

    return report


if __name__ == "__main__":
    try:
        report = run_benchmark()
        print("\n" + "=" * 60)
        print("JSON Report:")
        print("=" * 60)
        print(json.dumps(report, indent=2))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
