"""Debug memory overwrite mechanism.

Trace what happens during overwrite to understand the issue.
"""

import mlx.core as mx
import mlx.nn as nn
from hz0.validation.phase3_memory_redesign import ImprovedScratchpadMemory


def debug_overwrite():
    """Trace memory state during overwrite."""
    print("="*70)
    print("Debug: Memory Overwrite Mechanism")
    print("="*70)

    memory = ImprovedScratchpadMemory(model_dim=64, num_slots=8, slot_dim=64)
    B, D = 1, 64

    memory_state = memory.initialize_state(B)
    print(f"\nInitial state shape: {memory_state.shape}")
    print(f"Initial state mean: {float(mx.mean(mx.abs(memory_state))):.6f}")

    # Write v1 (ones * 1.0)
    key1 = mx.ones((B, D)) * 0.1
    val1 = mx.ones((B, D)) * 1.0

    print(f"\n--- Write v1 (value = 1.0) ---")
    memory_state = memory.write(key1, val1, memory_state)
    print(f"State after write v1: mean={float(mx.mean(mx.abs(memory_state))):.6f}, max={float(mx.max(mx.abs(memory_state))):.6f}")

    read1, attn1 = memory.read(key1, memory_state)
    read1_out = memory.output_proj(read1)
    dist1 = float(mx.mean(mx.abs(read1_out - val1)))
    print(f"Read after v1: distance={dist1:.6f}")
    print(f"Attention weights: {float(mx.mean(attn1)):.6f} (mean)")

    # Overwrite with v2 (ones * 2.0) multiple times
    val2 = mx.ones((B, D)) * 2.0

    print(f"\n--- Overwrite with v2 (value = 2.0) ---")
    for iter_num in range(5):
        memory_state = memory.write(key1, val2, memory_state)
        read2, attn2 = memory.read(key1, memory_state)
        read2_out = memory.output_proj(read2)
        dist2 = float(mx.mean(mx.abs(read2_out - val2)))
        print(
            f"Iteration {iter_num+1}: "
            f"state_mean={float(mx.mean(mx.abs(memory_state))):.6f}, "
            f"distance={dist2:.6f}"
        )

    print(f"\n--- Analysis ---")
    print(f"Initial read distance (v1): {dist1:.6f}")
    print(f"Final read distance (v2):   {dist2:.6f}")
    print(f"Improvement ratio:          {dist2 / dist1 if dist1 > 0 else 0:.4f}")
    print(f"Threshold for PASS:         < 0.8 (currently {'PASS' if dist2 < dist1 * 0.8 else 'FAIL'})")

    print(f"\n--- Problem Analysis ---")
    print(f"The memory state grows unbounded (accumulation).")
    print(f"Both v1 and v2 get summed into the state, so reading returns mix.")
    print(f"Solution: Replace instead of accumulate, or stronger erase.")


if __name__ == "__main__":
    debug_overwrite()
