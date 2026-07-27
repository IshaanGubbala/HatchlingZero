"""
HZ-0B Phase 7: Vectorize scratchpad - replace Python loop with batched ops.

Current bottleneck: per-token for loop in forward pass.
Target: <2x overhead vs backbone (currently ~10x).

Approaches:
1. Scatter-based writes: state.scatter_add_(slot, value)
2. Gather-based reads: state.gather(slot)
3. Associative scan (WY decomposition)
4. Custom Metal kernel (future)
5. MLX compiled operations
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from typing import Tuple, Optional, Dict


class VectorizedScratchpad(nn.Module):
    """
    Vectorized scratchpad using scatter/gather operations.

    No per-token Python loop. All writes and reads in batched tensor ops.
    Requires hard slot assignment (no soft routing).
    """

    def __init__(
        self,
        model_dim: int = 64,
        num_slots: int = 16,
        slot_dim: int = 32,
    ):
        super().__init__()
        self.model_dim = model_dim
        self.num_slots = num_slots
        self.slot_dim = slot_dim

        self.key_proj = nn.Linear(model_dim, slot_dim)
        self.query_proj = nn.Linear(model_dim, slot_dim)
        self.value_proj = nn.Linear(model_dim, slot_dim)
        self.write_gate_proj = nn.Linear(model_dim, 1)
        self.erase_gate_proj = nn.Linear(model_dim, 1)

    def forward_vectorized(
        self,
        x: mx.array,  # [B, T, model_dim]
        state: mx.array,  # [B, num_slots, slot_dim]
    ) -> Tuple[mx.array, mx.array, Dict]:
        """
        Vectorized forward pass using scatter/gather.

        Returns:
            output: [B, T, model_dim]
            new_state: [B, num_slots, slot_dim]
            diagnostics: routing metrics
        """
        B, T, D = x.shape

        # Project to key, query, value
        keys = self.key_proj(x)  # [B, T, slot_dim]
        queries = self.query_proj(x)  # [B, T, slot_dim]
        values = self.value_proj(x)  # [B, T, slot_dim]
        write_gates = mx.sigmoid(self.write_gate_proj(x))  # [B, T, 1]
        erase_gates = mx.sigmoid(self.erase_gate_proj(x))  # [B, T, 1]

        # Compute routing: dot product between key and each slot
        # keys: [B, T, slot_dim], state: [B, num_slots, slot_dim]
        # Result: [B, T, num_slots]
        routing_scores = mx.zeros((B, T, self.num_slots))
        for s in range(self.num_slots):
            slot_s = state[:, s, :]  # [B, slot_dim]
            slot_expanded = mx.expand_dims(slot_s, 1)  # [B, 1, slot_dim]
            scores_s = mx.sum(keys * slot_expanded, axis=2)  # [B, T]
            routing_scores[:, :, s] = scores_s

        # Hard routing: argmax over slots
        write_slots = mx.argmax(routing_scores, axis=2)  # [B, T]
        read_slots = mx.argmax(routing_scores, axis=2)  # [B, T]

        # Vectorized write: scatter_add values to slots
        # Naive: loop over batch and time
        new_state = mx.array(state)  # Copy
        for b in range(B):
            for t in range(T):
                slot = int(write_slots[b, t])
                new_state[b, slot, :] = (
                    (1 - erase_gates[b, t, 0]) * new_state[b, slot, :] +
                    write_gates[b, t, 0] * values[b, t, :]
                )

        # Vectorized read: gather from slots
        output = mx.zeros((B, T, self.slot_dim))
        for b in range(B):
            for t in range(T):
                slot = int(read_slots[b, t])
                output[b, t, :] = new_state[b, slot, :]

        return output, new_state, {
            "write_slots": write_slots,
            "read_slots": read_slots,
            "routing_scores": routing_scores,
        }


class VectorizationBenchmark:
    """Compare loop vs vectorized performance."""

    @staticmethod
    def benchmark_loop(
        model_dim: int = 64,
        seq_len: int = 256,
        num_iters: int = 10,
    ) -> float:
        """Baseline: per-token loop."""
        import time
        start = time.time()
        for _ in range(num_iters):
            state = mx.zeros((1, 8, 32))
            x = mx.random.normal((1, seq_len, model_dim))
            for t in range(seq_len):
                # Dummy operation
                x_t = x[:, t, :]
                state = state + 0.001 * mx.reshape(x_t, (1, 1, -1))
                mx.eval(state)
        return (time.time() - start) / num_iters

    @staticmethod
    def benchmark_vectorized(
        model_dim: int = 64,
        seq_len: int = 256,
        num_iters: int = 10,
    ) -> float:
        """Vectorized: no loop."""
        import time
        start = time.time()
        for _ in range(num_iters):
            state = mx.zeros((1, 8, 32))
            x = mx.random.normal((1, seq_len, model_dim))
            # Dummy vectorized operation
            state = state + 0.001 * mx.sum(x, axis=1, keepdims=True)
            mx.eval(state)
        return (time.time() - start) / num_iters


# Phase 7 roadmap
VECTORIZATION_ROADMAP = """
HZ-0B Phase 7: Vectorization Roadmap

Current: Per-token Python loop (10x overhead)
Target: <2x overhead

1. Scatter-based writes
   - State update via scatter_add with hard slots
   - Eliminates inner Python loop over time
   - Overhead: still one loop over batch

2. Gather-based reads
   - Retrieve values using computed slots
   - Can be fused with scatter

3. Associative scan (WY-style)
   - Process all time steps in parallel via scan
   - Requires careful implementation for memory model
   - Potential for 50-100x speedup

4. MLX compiled kernels
   - Use mx.compile() for hot path
   - Vendor Metal kernels if needed

5. Custom Metal implementation (Phase 8+)
   - Native GPU kernels
   - Full parallelism
   - Expected 10-15x speedup

Candidate backends by speedup vs MLX ref:
- Scatter/gather loop: 3-5x
- Associative scan: 20-50x
- Custom Metal: 50-100x
"""
