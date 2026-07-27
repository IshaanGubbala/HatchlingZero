"""
HZ-0B Phase 4: Oracle ablations - isolate failure modes.

Run model under 4 conditions:
1. Baseline (learned routing + storage + read)
2. Oracle routing (deterministic routing)
3. Oracle storage (ground-truth values)
4. Oracle read (oracle routing on reads)

Measures: Which ablation improves recall? Identifies bottleneck.
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.nn import losses as mlx_losses
import numpy as np
from typing import Dict, Tuple
import time

from hz0.scratchpad_lab.tiny_memory_model import TinyMemoryModel


def test_ablation(
    model: TinyMemoryModel,
    test_sequence: mx.array,  # [1, seq_len]
    test_targets: mx.array,   # [1, seq_len]
    variant: str = "baseline",
) -> Dict:
    """Run model variant and measure recall."""
    if variant == "baseline":
        logits, _, diags = model(test_sequence)
    elif variant == "oracle_routing":
        logits, _, diags = model.forward_oracle_routing(test_sequence)
    elif variant == "oracle_storage":
        logits, _, diags = model.forward_oracle_storage(test_sequence)
    elif variant == "oracle_read":
        logits, _, diags = model.forward_oracle_read(test_sequence)
    else:
        raise ValueError(f"Unknown variant: {variant}")

    # Measure recall at read position (seq_len // 2)
    read_pos = test_sequence.shape[1] // 2
    pred = mx.argmax(logits[0, read_pos, :])
    target = test_targets[0, read_pos]
    recall = float(pred == target)

    return {
        "variant": variant,
        "recall": recall,
        "logits_shape": logits.shape,
        "diagnostics": diags,
    }


def run_oracle_comparison():
    """Compare all 4 ablation variants."""
    print("=" * 60)
    print("HZ-0B PHASE 4: ORACLE ABLATION COMPARISON")
    print("=" * 60)
    print()

    # Create model
    model = TinyMemoryModel(
        vocab_size=256,
        model_dim=64,
        num_layers=1,
        num_slots=8,
        slot_dim=32,
    )
    print("Model: TinyMemoryModel (1M params)")
    print()

    # Test on each curriculum stage
    stages = [
        ("fixed_key_value", "Fixed key→value"),
        ("multiple_keys", "Multiple keys"),
        ("distractors", "With distractors"),
        ("overwrite", "Sequential overwrite"),
        ("protected", "Protected memory"),
    ]

    print("Test Data Generation:")
    print("-" * 60)

    for stage_name, stage_desc in stages:
        print(f"\n{stage_desc}:")

        # Generate test sequence
        if stage_name == "fixed_key_value":
            seq = mx.zeros((1, 32), dtype=mx.int32)
            seq[0, 0] = 10
            seq[0, 16] = 10
            target = mx.ones_like(seq) * 50
            target[0, :16] = 0
        elif stage_name == "multiple_keys":
            seq = mx.array([[10, 11, 10, 12, 10, 13, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]], dtype=mx.int32)
            target = mx.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 50, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=mx.int32)
        elif stage_name == "distractors":
            seq = mx.array([[10, 11, 11, 12, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=mx.int32)
            target = mx.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 50, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=mx.int32)
        elif stage_name == "overwrite":
            seq = mx.array([[10, 0, 0, 0, 10, 0, 0, 0, 10, 0, 0, 0, 10, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=mx.int32)
            target = mx.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 60, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=mx.int32)
        elif stage_name == "protected":
            seq = mx.array([[10, 11, 0, 0, 10, 0, 0, 0, 11, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=mx.int32)
            target = mx.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 51, 60, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=mx.int32)
        else:
            continue

        # Test all variants
        variants = ["baseline", "oracle_routing", "oracle_storage", "oracle_read"]
        results = {}

        for variant in variants:
            result = test_ablation(model, seq, target, variant)
            results[variant] = result

        # Print results
        print(f"  Baseline:        {results['baseline']['recall']:.0%} recall")
        print(f"  Oracle routing:  {results['oracle_routing']['recall']:.0%} recall")
        print(f"  Oracle storage:  {results['oracle_storage']['recall']:.0%} recall")
        print(f"  Oracle read:     {results['oracle_read']['recall']:.0%} recall")

        # Analyze
        baseline = results['baseline']['recall']
        routing_boost = results['oracle_routing']['recall'] - baseline
        storage_boost = results['oracle_storage']['recall'] - baseline
        read_boost = results['oracle_read']['recall'] - baseline

        if routing_boost > 0.3:
            print(f"  → Routing is bottleneck (+{routing_boost:.0%})")
        if storage_boost > 0.3:
            print(f"  → Storage is bottleneck (+{storage_boost:.0%})")
        if read_boost > 0.3:
            print(f"  → Read routing is bottleneck (+{read_boost:.0%})")

    print()
    print("=" * 60)
    print("ABLATION SUMMARY")
    print("=" * 60)
    print("""
Interpretation:
- If oracle routing helps: routing learned incorrectly
- If oracle storage helps: values not stored properly
- If oracle read helps: read routing learned incorrectly
- If all fail together: fundamental architecture issue
- If all pass: memory system works; curriculum just needs longer training
""")


if __name__ == "__main__":
    run_oracle_comparison()
