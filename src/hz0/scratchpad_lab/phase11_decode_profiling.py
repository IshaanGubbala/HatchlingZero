"""
Phase 11: Decode profiling (plan section 14, experiment 5).

Break down token latency into components:
- embedding
- QKV/gate projections
- recurrent state update
- attention layers
- MLP
- normalization
- LM head
- sampling

Goal: Identify 5x slowdown source vs transformer baseline.
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import time
from typing import Dict, Tuple

from hz0.model_port.mlx_gdn2_lm import create_hz_36m_mlx, create_hz_110m_mlx
from hz0.scratchpad_lab.phase6_hz0a_training import SimpleTransformerBaseline


class TokenLatencyProfiler:
    """Profile per-component decode latency."""

    def __init__(self, model: nn.Module, model_name: str = "hybrid", is_hybrid: bool = True):
        self.model = model
        self.model_name = model_name
        self.is_hybrid = is_hybrid
        self.timings = {}

    def profile_single_token(self, token_id: int, seq_len: int = 256) -> Dict[str, float]:
        """Profile latency for single token decode."""
        # Create dummy sequence
        seq = np.zeros(seq_len, dtype=np.int32)
        seq[-1] = token_id
        seq_mx = mx.array(seq).reshape(1, -1)

        timings = {}
        total_start = time.perf_counter()

        # Forward pass (monolithic)
        start = time.perf_counter()
        output = self.model(seq_mx)
        if self.is_hybrid:
            logits, state = output
        else:
            logits = output
        mx.eval(logits)
        elapsed = time.perf_counter() - start
        timings["forward_total"] = elapsed

        # Predict next token
        start = time.perf_counter()
        pred_idx = mx.argmax(logits[0, -1, :])
        mx.eval(pred_idx)
        elapsed = time.perf_counter() - start
        timings["argmax"] = elapsed

        total_elapsed = time.perf_counter() - total_start
        timings["total"] = total_elapsed

        return timings

    def profile_batch_decode(
        self, num_tokens: int = 32, seq_len: int = 256, num_runs: int = 5
    ) -> Dict[str, float]:
        """Profile decode throughput across multiple tokens."""
        times = []

        for run in range(num_runs):
            seq = np.random.randint(0, 32768, seq_len, dtype=np.int32)
            seq_mx = mx.array(seq).reshape(1, -1)

            start = time.perf_counter()
            for _ in range(num_tokens):
                output = self.model(seq_mx)
                if self.is_hybrid:
                    logits, state = output
                else:
                    logits = output
                pred_idx = mx.argmax(logits[0, -1, :])
                mx.eval(pred_idx)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        avg_time = np.mean(times)
        avg_per_token = avg_time / num_tokens
        throughput = 1.0 / avg_per_token if avg_per_token > 0 else 0

        return {
            "total_time_sec": avg_time,
            "time_per_token_ms": avg_per_token * 1000,
            "throughput_tokens_per_sec": throughput,
            "std_dev_ms": np.std(times) * 1000 / num_tokens,
        }

    def profile_prefill_vs_decode(
        self, prefill_len: int = 256, decode_tokens: int = 10
    ) -> Dict[str, float]:
        """Profile prefill (full sequence) vs decode (one token at a time)."""
        results = {}

        # Prefill: full sequence forward
        seq = np.random.randint(0, 32768, prefill_len, dtype=np.int32)
        seq_mx = mx.array(seq).reshape(1, -1)

        start = time.perf_counter()
        output = self.model(seq_mx)
        if self.is_hybrid:
            logits, state = output
        else:
            logits = output
        mx.eval(logits)
        prefill_time = time.perf_counter() - start

        results["prefill_time_sec"] = prefill_time
        results["prefill_throughput_tok_per_sec"] = prefill_len / prefill_time

        # Decode: one token at a time
        seq_pos = np.zeros(prefill_len, dtype=np.int32)
        seq_mx_pos = mx.array(seq_pos).reshape(1, -1)

        start = time.perf_counter()
        for _ in range(decode_tokens):
            output = self.model(seq_mx_pos)
            if self.is_hybrid:
                logits, state = output
            else:
                logits = output
            pred_idx = mx.argmax(logits[0, -1, :])
            mx.eval(pred_idx)
        decode_time = time.perf_counter() - start

        results["decode_time_sec"] = decode_time
        results["decode_throughput_tok_per_sec"] = decode_tokens / decode_time
        results["prefill_to_decode_ratio"] = prefill_time / decode_time if decode_time > 0 else 0

        return results


def run_decode_profiling():
    """Run full decode profiling suite."""
    print("=" * 80)
    print("PHASE 11: DECODE PROFILING (Plan Section 14, Exp 5)")
    print("=" * 80)
    print()

    # Profile 36M hybrid
    print("Model 1: 36M Hybrid")
    print("-" * 80)
    hybrid_36m = create_hz_36m_mlx()
    profiler_h36 = TokenLatencyProfiler(hybrid_36m, "hybrid_36m", is_hybrid=True)

    # Warmup
    seq = np.random.randint(0, 32768, 256, dtype=np.int32)
    seq_mx = mx.array(seq).reshape(1, -1)
    _ = hybrid_36m(seq_mx)

    # Profile
    h36_decode = profiler_h36.profile_batch_decode(num_tokens=32, seq_len=256, num_runs=5)
    h36_prefill = profiler_h36.profile_prefill_vs_decode(prefill_len=256, decode_tokens=10)

    print(f"Decode throughput: {h36_decode['throughput_tokens_per_sec']:.0f} tok/s")
    print(f"Time per token: {h36_decode['time_per_token_ms']:.2f} ms")
    print(f"Prefill throughput: {h36_prefill['prefill_throughput_tok_per_sec']:.0f} tok/s")
    print(f"Decode throughput (single): {h36_prefill['decode_throughput_tok_per_sec']:.0f} tok/s")
    print()

    # Profile transformer baseline
    print("Model 2: Transformer Baseline (768D)")
    print("-" * 80)
    transformer = SimpleTransformerBaseline()
    profiler_tf = TokenLatencyProfiler(transformer, "transformer", is_hybrid=False)

    # Warmup
    _ = transformer(seq_mx)

    # Profile
    tf_decode = profiler_tf.profile_batch_decode(num_tokens=32, seq_len=256, num_runs=5)
    tf_prefill = profiler_tf.profile_prefill_vs_decode(prefill_len=256, decode_tokens=10)

    print(f"Decode throughput: {tf_decode['throughput_tokens_per_sec']:.0f} tok/s")
    print(f"Time per token: {tf_decode['time_per_token_ms']:.2f} ms")
    print(f"Prefill throughput: {tf_prefill['prefill_throughput_tok_per_sec']:.0f} tok/s")
    print(f"Decode throughput (single): {tf_prefill['decode_throughput_tok_per_sec']:.0f} tok/s")
    print()

    # Profile 110M hybrid
    print("Model 3: 110M Hybrid (Large)")
    print("-" * 80)
    hybrid_110m = create_hz_110m_mlx()
    profiler_h110 = TokenLatencyProfiler(hybrid_110m, "hybrid_110m", is_hybrid=True)

    # Warmup
    _ = hybrid_110m(seq_mx)

    # Profile
    h110_decode = profiler_h110.profile_batch_decode(num_tokens=16, seq_len=256, num_runs=3)
    h110_prefill = profiler_h110.profile_prefill_vs_decode(prefill_len=256, decode_tokens=5)

    print(f"Decode throughput: {h110_decode['throughput_tokens_per_sec']:.0f} tok/s")
    print(f"Time per token: {h110_decode['time_per_token_ms']:.2f} ms")
    print(f"Prefill throughput: {h110_prefill['prefill_throughput_tok_per_sec']:.0f} tok/s")
    print(f"Decode throughput (single): {h110_prefill['decode_throughput_tok_per_sec']:.0f} tok/s")
    print()

    # Comparison
    print("=" * 80)
    print("DECODE PERFORMANCE COMPARISON")
    print("=" * 80)
    print()
    print(f"{'Model':<25} {'Decode tok/s':<15} {'Per-token ms':<15} {'vs Transformer':<15}")
    print("-" * 80)

    tf_throughput = tf_decode["throughput_tokens_per_sec"]
    print(f"{'Transformer 768D':<25} {tf_throughput:>10.0f}      {tf_decode['time_per_token_ms']:>10.2f}    baseline")
    print(
        f"{'36M Hybrid':<25} {h36_decode['throughput_tokens_per_sec']:>10.0f}      {h36_decode['time_per_token_ms']:>10.2f}    {tf_throughput/h36_decode['throughput_tokens_per_sec']:.1f}x"
    )
    print(
        f"{'110M Hybrid':<25} {h110_decode['throughput_tokens_per_sec']:>10.0f}      {h110_decode['time_per_token_ms']:>10.2f}    {tf_throughput/h110_decode['throughput_tokens_per_sec']:.1f}x"
    )

    print()
    print("=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    print()

    h36_ratio = tf_throughput / h36_decode["throughput_tokens_per_sec"]
    h110_ratio = tf_throughput / h110_decode["throughput_tokens_per_sec"]

    print(f"36M Hybrid decode slowdown: {h36_ratio:.1f}x")
    print(f"110M Hybrid decode slowdown: {h110_ratio:.1f}x")
    print()

    print("Prefill vs Decode:")
    print(f"  36M: prefill {h36_prefill['prefill_throughput_tok_per_sec']:.0f} tok/s vs decode {h36_prefill['decode_throughput_tok_per_sec']:.0f} tok/s")
    print(f"  Transformer: prefill {tf_prefill['prefill_throughput_tok_per_sec']:.0f} tok/s vs decode {tf_prefill['decode_throughput_tok_per_sec']:.0f} tok/s")
    print()

    if h36_ratio > 2:
        print(f"⚠ Slowdown significant ({h36_ratio:.1f}x). Candidates:")
        print("  - Recurrent state accumulation (RNN loop per token)")
        print("  - Metal graph launches (overhead per token)")
        print("  - Python loop overhead (interpreted dispatch)")
        print("  - State copies (device memory transfers)")
        print("  - Attention layers (if dense attention per token)")
    else:
        print(f"✓ Slowdown acceptable ({h36_ratio:.1f}x)")

    print()
    print("NEXT STEPS")
    print("-" * 80)
    print("1. Profile with MLX Metal logs to isolate kernel vs Python overhead")
    print("2. Test streaming mode (fixed state size per token)")
    print("3. Compare chunked decode vs token-by-token")
    print("4. Benchmark with longer sequences (context window effect)")


if __name__ == "__main__":
    run_decode_profiling()
