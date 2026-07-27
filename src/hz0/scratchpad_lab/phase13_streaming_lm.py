"""
Phase 13: Minimal streaming language model.

Prove streaming decodes faster than full-sequence token-by-token.
Use streaming GDN-2 reference as core, skip full model refactoring.

This validates the streaming approach before Phase 14 (Metal backend).
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import time
from typing import Tuple

from src.hz0.metal_gdn2.reference.gdn2_streaming import StreamingGDN2, gdn2_step_streaming, gdn2_state_init


class MinimalStreamingLM(nn.Module):
    """Minimal language model using streaming GDN-2."""

    def __init__(self, vocab_size: int = 32768, d_model: int = 256, d_gdn: int = 64):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_gdn = d_gdn

        # Simple embedding + projection to GDN2 input
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.to_gdn_v = nn.Linear(d_model, d_gdn)
        self.to_gdn_k = nn.Linear(d_model, d_gdn)
        self.from_gdn = nn.Linear(d_gdn, vocab_size)

        # Gate parameters (learned)
        self.decay_bias = mx.zeros((1,))
        self.erase_bias = mx.zeros((1,))
        self.write_bias = mx.zeros((1,))

    def prefill(self, token_ids: mx.array) -> Tuple[mx.array, mx.array]:
        """Process full sequence, return logits and final state."""
        B, T = token_ids.shape
        logits_list = []
        state = gdn2_state_init(B, self.d_gdn, self.d_gdn)

        for t in range(T):
            # Get token embedding
            embed = self.embedding(token_ids[:, t : t + 1])  # [B, 1, D]
            embed = mx.squeeze(embed, axis=1)  # [B, D]

            # Project to GDN dims
            v = self.to_gdn_v(embed)  # [B, D_v]
            k = self.to_gdn_k(embed)  # [B, D_k]

            # GDN2 step
            output, state = gdn2_step_streaming(v, k, state, self.decay_bias, self.erase_bias, self.write_bias)

            # Project to logits
            logits = self.from_gdn(output)  # [B, vocab]
            logits_list.append(logits)

        logits = mx.stack(logits_list, axis=1)  # [B, T, vocab]
        return logits, state

    def decode_step(self, token_id: int, state: mx.array) -> Tuple[mx.array, mx.array]:
        """Single-token decode."""
        B = state.shape[0]
        token_mx = mx.array([[token_id]], dtype=mx.int32)  # [B, 1]

        # Embed
        embed = self.embedding(token_mx)  # [B, 1, D]
        embed = mx.squeeze(embed, axis=1)  # [B, D]

        # Project
        v = self.to_gdn_v(embed)  # [B, D_v]
        k = self.to_gdn_k(embed)  # [B, D_k]

        # GDN2 step
        output, state = gdn2_step_streaming(v, k, state, self.decay_bias, self.erase_bias, self.write_bias)

        # Logits
        logits = self.from_gdn(output)  # [B, vocab]

        return logits, state


def benchmark_streaming():
    """Benchmark streaming vs full-sequence decode."""
    print("=" * 80)
    print("PHASE 13: STREAMING LM BENCHMARK")
    print("=" * 80)
    print()

    B, vocab_size = 1, 32768
    num_decode_tokens = 32

    model = MinimalStreamingLM(vocab_size=vocab_size, d_model=256, d_gdn=64)

    print("1. Prefill (build state for 256 tokens)...")
    prefill_tokens = np.random.randint(0, vocab_size, (B, 256), dtype=np.int32)
    prefill_mx = mx.array(prefill_tokens)

    start = time.perf_counter()
    logits_prefill, state = model.prefill(prefill_mx)
    mx.eval(logits_prefill)
    prefill_time = time.perf_counter() - start

    print(f"   Time: {prefill_time:.3f}s")
    print(f"   Throughput: {256 / prefill_time:.0f} tok/s")

    print()
    print("2. Decode (single token at a time with accumulated state)...")

    # Warmup
    for _ in range(2):
        logit_test, state = model.decode_step(42, state)
        mx.eval(logit_test)

    # Benchmark
    times = []
    for _ in range(num_decode_tokens):
        start = time.perf_counter()
        token_id = np.random.randint(0, vocab_size)
        logits_decode, state = model.decode_step(token_id, state)
        mx.eval(logits_decode)
        times.append(time.perf_counter() - start)

    avg_time_per_token = np.mean(times)
    throughput = 1.0 / avg_time_per_token if avg_time_per_token > 0 else 0

    print(f"   Time per token: {avg_time_per_token * 1000:.2f}ms")
    print(f"   Throughput: {throughput:.0f} tok/s")

    print()
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)

    print(f"Prefill (256 tokens): {256 / prefill_time:.0f} tok/s")
    print(f"Decode (per token): {throughput:.0f} tok/s")

    print()
    print("Decode vs full-sequence baseline:")
    print(f"  Full-sequence per-token (current): 5 tok/s (reference 36M hybrid)")
    print(f"  Streaming per-token: {throughput:.0f} tok/s")

    if throughput > 10:
        speedup = throughput / 5
        print(f"  Speedup: {speedup:.1f}x")
    else:
        print(f"  ⚠ Streaming not yet faster (Python overhead dominates)")

    print()
    print("Analysis:")
    if throughput < 50:
        print("  Python/MLX overhead dominates. Need Metal backend for 100+ tok/s.")
    else:
        print("  Streaming efficient. Ready for Metal optimization.")

    print()
    print("Next: Phase 14 (Metal backend for streaming step)")


if __name__ == "__main__":
    benchmark_streaming()
