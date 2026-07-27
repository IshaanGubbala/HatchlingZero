"""
Phase 14a-2: Benchmark streaming decode speedup.

Compare:
- Full-sequence per-token (old, wasteful)
- Streaming with state caching (new, efficient)
"""

import mlx.core as mx
import numpy as np
import time

from hz0.model_port.mlx_gdn2_lm import create_hz_36m_mlx


def benchmark_full_sequence_decode():
    """Decode using full-sequence forward (old, wasteful)."""
    model = create_hz_36m_mlx()
    num_tokens = 32

    # Accumulate tokens
    tokens = [np.random.randint(0, 32768) for _ in range(num_tokens)]
    accumulated = []

    print("Old approach: Full-sequence per token")
    print("-" * 80)

    # Warmup
    seq = mx.array([accumulated + [tokens[0]]], dtype=mx.int32)
    _ = model(seq)

    # Benchmark
    times = []
    for i, token in enumerate(tokens):
        accumulated.append(token)
        seq = mx.array([accumulated], dtype=mx.int32)

        start = time.perf_counter()
        logits, _ = model(seq)
        mx.eval(logits)
        times.append(time.perf_counter() - start)

    avg_time = np.mean(times)
    throughput = 1.0 / avg_time if avg_time > 0 else 0

    print(f"Time per token: {avg_time * 1000:.2f}ms")
    print(f"Throughput: {throughput:.0f} tok/s")

    return throughput


def benchmark_streaming_decode():
    """Decode using streaming with state caching (new, efficient)."""
    model = create_hz_36m_mlx()
    num_tokens = 32

    print("\nNew approach: Streaming with state caching")
    print("-" * 80)

    # Warmup
    logits, layer_states, kv_caches = model.decode_step(42)
    mx.eval(logits)

    # Benchmark
    times = []
    for i in range(num_tokens):
        token_id = np.random.randint(0, 32768)

        start = time.perf_counter()
        logits, layer_states, kv_caches = model.decode_step(token_id, layer_states, kv_caches)
        mx.eval(logits)
        times.append(time.perf_counter() - start)

    avg_time = np.mean(times)
    throughput = 1.0 / avg_time if avg_time > 0 else 0

    print(f"Time per token: {avg_time * 1000:.2f}ms")
    print(f"Throughput: {throughput:.0f} tok/s")

    return throughput


if __name__ == "__main__":
    print("=" * 80)
    print("PHASE 14a-2: STREAMING DECODE BENCHMARK")
    print("=" * 80)
    print()

    old_throughput = benchmark_full_sequence_decode()
    new_throughput = benchmark_streaming_decode()

    print()
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Old (full-sequence): {old_throughput:.0f} tok/s")
    print(f"New (streaming): {new_throughput:.0f} tok/s")

    if new_throughput > old_throughput:
        speedup = new_throughput / old_throughput
        print(f"Speedup: {speedup:.1f}x")
    else:
        print(f"⚠ Streaming not faster (possibly due to MLX/Python overhead)")

    print()
    print("Analysis:")
    print(f"  Phase 11 baseline (full 36M model): 5 tok/s")
    print(f"  Phase 14a-2 (streaming refactored): {new_throughput:.0f} tok/s")

    if new_throughput > 20:
        print("  ✓ Speedup achieved. Phase 14a SUCCESSFUL")
        print("  Ready for Phase 14b (full validation + training equiv)")
    elif new_throughput > 5:
        print(f"  ⚠ Partial speedup ({new_throughput/5:.1f}x)")
        print("  Phase 15 (Metal backend) still needed for target (100+ tok/s)")
    else:
        print("  ✗ No improvement (Python/MLX overhead still dominates)")
