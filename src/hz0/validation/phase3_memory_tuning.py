"""Tune memory overwrite gate scaling.

Experiment with erase/write ratios to maximize overwrite capability.
"""

import mlx.core as mx
import mlx.nn as nn
from src.hz0.validation.phase3_memory_redesign import ImprovedScratchpadMemory


def test_overwrite_scaling(erase_scale: float, write_scale: float) -> float:
    """Test overwrite with given erase/write scales.

    Returns: improvement ratio (< 1.0 is better)
    """
    # Monkey-patch the write method to use custom scaling
    class TunedMemory(ImprovedScratchpadMemory):
        def write(self, key, value, memory_state):
            k = self.key_proj(key)
            v = self.value_proj(value)
            write_gates = mx.sigmoid(self.write_gate_proj(key))
            erase_gates = mx.sigmoid(self.erase_gate_proj(key))

            erase_effect = mx.expand_dims(erase_gates, axis=2)
            memory_state = memory_state * (1.0 - erase_effect * erase_scale)

            write_effect = mx.expand_dims(write_gates, axis=2) * mx.expand_dims(v, axis=1)
            memory_state = memory_state + write_effect * write_scale

            memory_state = mx.clip(memory_state, -10.0, 10.0)
            return memory_state

    memory = TunedMemory(model_dim=64, num_slots=8, slot_dim=64)
    B, D = 1, 64

    memory_state = memory.initialize_state(B)

    # Write v1
    key1 = mx.ones((B, D)) * 0.1
    val1 = mx.ones((B, D)) * 1.0
    memory_state = memory.write(key1, val1, memory_state)

    read1, _ = memory.read(key1, memory_state)
    dist1 = float(mx.mean(mx.abs(read1 - val1)))

    # Overwrite with v2
    val2 = mx.ones((B, D)) * 2.0
    for _ in range(5):
        memory_state = memory.write(key1, val2, memory_state)

    read2, _ = memory.read(key1, memory_state)
    dist2 = float(mx.mean(mx.abs(read2 - val2)))

    # Return improvement (lower is better)
    return dist2 / dist1 if dist1 > 0 else 0.0


def main():
    """Test various erase/write combinations."""
    print("="*70)
    print("Memory Overwrite Gate Tuning")
    print("="*70)

    print("\nTesting erase_scale × write_scale combinations:")
    print("-" * 70)
    print(f"{'Erase':<10} {'Write':<10} {'Ratio':<10} {'Status':<20}")
    print("-" * 70)

    best_ratio = float("inf")
    best_config = None

    erase_scales = [0.3, 0.5, 0.7, 1.0, 1.5]
    write_scales = [0.5, 0.7, 1.0, 1.5, 2.0, 3.0]

    for erase_scale in erase_scales:
        for write_scale in write_scales:
            ratio = test_overwrite_scaling(erase_scale, write_scale)

            status = "✗ FAIL" if ratio > 0.8 else "⚠ OK" if ratio > 0.5 else "✓ PASS"
            print(f"{erase_scale:<10.1f} {write_scale:<10.1f} {ratio:<10.4f} {status}")

            if ratio < best_ratio and ratio <= 0.5:
                best_ratio = ratio
                best_config = (erase_scale, write_scale)

    print("-" * 70)
    print(f"\n✓ Best configuration found:")
    if best_config:
        erase, write = best_config
        print(f"  Erase scale: {erase:.1f}")
        print(f"  Write scale: {write:.1f}")
        print(f"  Improvement ratio: {best_ratio:.4f}")
        print(f"\nRecommended update to phase3_memory_redesign.py:")
        print(f"  Line 104: memory_state * (1.0 - erase_effect * {erase:.1f})")
        print(f"  Line 109: write_effect * {write:.1f}")
    else:
        print("  No optimal config found. Baseline (0.5, 0.5) may be best.")


if __name__ == "__main__":
    main()
