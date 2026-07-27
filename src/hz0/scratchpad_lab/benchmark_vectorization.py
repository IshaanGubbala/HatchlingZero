"""
Phase 7 benchmark: Loop vs vectorized scratchpad performance.

Target: 3-5x speedup from vectorized implementation.
"""

import mlx.core as mx
import numpy as np
import time
from typing import Dict

from hz0.scratchpad_lab.tiny_memory_model import TinyMemoryModel
from hz0.scratchpad_lab.vectorized_scratchpad import VectorizedScratchpad


def benchmark_loop_based(
    batch_size: int = 2,
    seq_len: int = 256,
    num_iters: int = 10,
) -> float:
    """Benchmark loop-based (current) implementation."""
    model = TinyMemoryModel(vocab_size=256, model_dim=64, num_layers=1, num_slots=8, slot_dim=32)
    x = mx.random.normal((batch_size, seq_len, 64))

    times = []
    for _ in range(num_iters):
        state = model._get_initial_state(batch_size)
        start = time.time()
        logits, _, _ = model(mx.array(np.random.randint(0, 256, (batch_size, seq_len)), dtype=mx.int32))
        mx.eval(logits)
        times.append(time.time() - start)

    return np.mean(times[1:])  # Skip warmup


def benchmark_vectorized(
    batch_size: int = 2,
    seq_len: int = 256,
    num_iters: int = 10,
) -> float:
    """Benchmark vectorized implementation."""
    model = VectorizedScratchpad(model_dim=64, num_slots=8, slot_dim=32)
    x = mx.random.normal((batch_size, seq_len, 64))

    times = []
    for _ in range(num_iters):
        state = mx.zeros((batch_size, 8, 32))
        start = time.time()
        output, new_state, _ = model.forward_vectorized(x, state)
        mx.eval(output, new_state)
        times.append(time.time() - start)

    return np.mean(times[1:])


def run_vectorization_benchmark():
    """Run full benchmark suite."""
    print("=" * 70)
    print("PHASE 7: VECTORIZATION BENCHMARK")
    print("=" * 70)
    print()

    configs = [
        {"batch_size": 1, "seq_len": 128, "num_iters": 10},
        {"batch_size": 2, "seq_len": 256, "num_iters": 10},
        {"batch_size": 4, "seq_len": 512, "num_iters": 5},
    ]

    results = []

    for config in configs:
        print(f"Benchmark: B={config['batch_size']}, T={config['seq_len']}")

        loop_time = benchmark_loop_based(**config)
        vec_time = benchmark_vectorized(**config)
        speedup = loop_time / vec_time

        results.append({
            "config": config,
            "loop_time_ms": loop_time * 1000,
            "vec_time_ms": vec_time * 1000,
            "speedup": speedup,
        })

        print(f"  Loop-based:    {loop_time*1000:.2f} ms")
        print(f"  Vectorized:    {vec_time*1000:.2f} ms")
        print(f"  Speedup:       {speedup:.1f}x")
        print()

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    avg_speedup = np.mean([r["speedup"] for r in results])
    print(f"Average speedup: {avg_speedup:.1f}x")
    print(f"Target: 3-5x")
    if avg_speedup >= 3:
        print("✓ PASSED")
    else:
        print(f"✗ Below target (got {avg_speedup:.1f}x)")


if __name__ == "__main__":
    run_vectorization_benchmark()
