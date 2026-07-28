"""
Phase 4: Metal kernel optimization.

Benchmark phases 1-5 of Metal kernel implementation path.
"""

import mlx.core as mx
import numpy as np
from typing import Dict
import time

from hz0.metal_gdn2.kernels.metal_optimization import (
    MetalKernelOptimizer,
    MetalKernelImplementationPath,
)


def benchmark_phase1_reference() -> Dict[str, float]:
    """Phase 1: Reference MLX ops baseline."""
    print("Phase 1: MLX Reference Ops")
    print("-" * 40)

    # Create test tensors
    B, T, H, Dk, Dv = 1, 128, 2, 64, 64
    queries = mx.array(np.random.randn(B, T, H, Dk).astype(np.float32) * 0.1)
    keys = mx.array(np.random.randn(B, T, H, Dk).astype(np.float32) * 0.1)
    values = mx.array(np.random.randn(B, T, H, Dv).astype(np.float32) * 0.1)
    decays = mx.array(
        np.clip(0.95 + 0.02 * np.random.randn(B, T, H, Dk), 0.9, 1.0).astype(np.float32)
    )
    erases = mx.array(np.clip(np.random.randn(B, T, H, Dk), 0, 1).astype(np.float32))
    writes = mx.array(np.clip(np.random.randn(B, T, H, Dv), 0, 1).astype(np.float32))
    state = mx.zeros((B, H, Dv, Dk))

    # Benchmark forward
    forward_metrics = MetalKernelOptimizer.benchmark_forward(
        queries, keys, values, decays, erases, writes, state, num_iters=5
    )

    print(f"Forward latency: {forward_metrics['forward_latency_ms']:.2f} ms")
    print(f"Forward throughput: {forward_metrics['throughput_tokens_per_sec']:.0f} tok/s")

    # Benchmark backward
    backward_metrics = MetalKernelOptimizer.benchmark_backward(
        queries, keys, values, decays, erases, writes, state, num_iters=3
    )

    print(f"Backward latency: {backward_metrics['backward_latency_ms']:.2f} ms")

    # Benchmark decode
    decode_metrics = MetalKernelOptimizer.profile_decode(
        queries[:, 0],
        keys[:, 0],
        values[:, 0],
        decays[:, 0],
        erases[:, 0],
        writes[:, 0],
        state,
        num_iters=50,
    )

    print(f"Decode latency: {decode_metrics['decode_latency_ms']:.2f} ms")
    print(f"Decode throughput: {decode_metrics['tokens_per_second']:.1f} tok/s")

    return {
        "phase": 1,
        "name": "MLX Reference",
        **forward_metrics,
        **backward_metrics,
        **decode_metrics,
    }


def run_phase4_benchmarks() -> Dict:
    """Run Phase 4 Metal kernel optimization benchmarks."""
    print("=" * 60)
    print("PHASE 4: METAL KERNEL OPTIMIZATION")
    print("=" * 60)
    print()

    print("Current Implementation:")
    print(f"  {MetalKernelImplementationPath.current_phase()}")
    print()

    print("Optimization Roadmap:")
    print(MetalKernelImplementationPath.implementation_roadmap())
    print()

    print("=" * 60)
    print("BENCHMARKING")
    print("=" * 60)
    print()

    # Benchmark phase 1 (reference)
    phase1_results = benchmark_phase1_reference()

    print()
    print("=" * 60)
    print("PHASE 4 SUMMARY")
    print("=" * 60)
    print()

    print("Success Criteria:")
    for criterion, description in MetalKernelSuccessCriteria.criteria().items():
        print(f"  ✓ {criterion}: {description}")

    print()
    print("Next Steps:")
    print("  1. Implement mx.fast.metal_kernel wrapper (Phase 2)")
    print("  2. Profile wrapper overhead vs reference")
    print("  3. Fuse decay/erase/write operations (Phase 3)")
    print("  4. Implement chunk-parallel forward (Phase 4)")
    print("  5. Register-resident state optimization (Phase 5)")

    return {
        "phase1_reference": phase1_results,
        "roadmap": MetalKernelImplementationPath.implementation_roadmap(),
        "success_criteria": MetalKernelSuccessCriteria.criteria(),
    }


class MetalKernelSuccessCriteria:
    """Gate 4 success criteria."""

    @staticmethod
    def criteria() -> Dict[str, str]:
        return {
            "decode_speedup": "Decode 2× faster than MLX reference",
            "training_speedup": "Training 2× faster than MLX reference",
            "memory_efficiency": "Peak memory <24GB for 110M model",
            "numerical_stability": "Match MLX to 1e-4 tolerance",
            "scaling": "Maintain speedup at seq_len 128-2048",
        }


if __name__ == "__main__":
    results = run_phase4_benchmarks()
