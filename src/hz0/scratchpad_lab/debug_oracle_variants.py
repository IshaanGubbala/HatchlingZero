"""
Debug: Why oracle variants aren't improving over baseline.

Check: Do oracle routines actually route differently?
Are they using the hash-based oracle correctly?
"""

import mlx.core as mx
import numpy as np
from src.hz0.scratchpad_lab.tiny_memory_model import TinyMemoryModel
from src.hz0.scratchpad_lab.test_tiny_model import MemoryCurriculumStage


def debug_oracle_routing():
    """Trace oracle routing behavior."""
    print("=" * 70)
    print("DEBUG: ORACLE ROUTING")
    print("=" * 70)
    print()

    model = TinyMemoryModel(vocab_size=256, model_dim=64, num_layers=1, num_slots=8, slot_dim=32)
    stage = MemoryCurriculumStage("fixed_key_value", num_training_examples=10)

    # Generate a simple sequence
    seq, target = stage.generate_batch(batch_size=1, held_out=False)

    print(f"Sequence shape: {seq.shape}")
    print(f"Target shape: {target.shape}")
    print()

    # Run standard forward
    logits_baseline, _, _ = model(seq)
    print("✓ Baseline forward completed")

    # Run forward_oracle_routing
    try:
        logits_oracle, _, _ = model.forward_oracle_routing(seq)
        print("✓ Oracle routing forward completed")
    except Exception as e:
        print(f"✗ Oracle routing error: {e}")
        return

    # Compare predictions
    read_start = stage.seq_len // 2
    baseline_pred = mx.argmax(logits_baseline[0, read_start, :])
    oracle_pred = mx.argmax(logits_oracle[0, read_start, :])
    target_val = target[0, read_start]

    print()
    print(f"Read position: {read_start}")
    print(f"  Baseline prediction: {baseline_pred}")
    print(f"  Oracle prediction:   {oracle_pred}")
    print(f"  Target value:        {target_val}")
    print(f"  Baseline correct: {baseline_pred == target_val}")
    print(f"  Oracle correct:   {oracle_pred == target_val}")
    print()

    # Check if routings differ
    if baseline_pred == oracle_pred:
        print("⚠ Predictions identical (oracle not providing alternative routing)")
    else:
        print("✓ Predictions differ (oracle routing has alternative)")
    print()


def debug_oracle_ablations():
    """Check all oracle variants on one sequence."""
    print("=" * 70)
    print("DEBUG: ALL ORACLE VARIANTS")
    print("=" * 70)
    print()

    model = TinyMemoryModel(vocab_size=256, model_dim=64, num_layers=1, num_slots=8, slot_dim=32)
    stage = MemoryCurriculumStage("fixed_key_value", num_training_examples=10)

    seq, target = stage.generate_batch(batch_size=1, held_out=False)
    read_start = stage.seq_len // 2
    target_val = target[0, read_start]

    variants = [
        ("baseline", model.__call__),
        ("oracle_routing", model.forward_oracle_routing),
        ("oracle_storage", model.forward_oracle_storage),
        ("oracle_read", model.forward_oracle_read),
    ]

    results = []
    for name, forward_fn in variants:
        try:
            logits, _, _ = forward_fn(seq)
            pred = mx.argmax(logits[0, read_start, :])
            correct = bool(pred == target_val)
            results.append({
                "variant": name,
                "prediction": int(pred),
                "correct": correct,
                "confidence": float(mx.max(logits[0, read_start, :]))
            })
            print(f"✓ {name:20s}: pred={int(pred)}, correct={correct}, confidence={float(mx.max(logits[0, read_start, :])):.2f}")
        except Exception as e:
            print(f"✗ {name:20s}: {e}")
    print()

    # Summary
    correct_count = sum(1 for r in results if r["correct"])
    print(f"Summary: {correct_count}/{len(results)} variants correct")
    if correct_count == 0:
        print("⚠ All variants wrong - check training or oracle signal")
    elif correct_count < len(results):
        print(f"⚠ Partial correctness - oracle variants should isolate bottlenecks")


def check_oracle_methods_exist():
    """Verify oracle methods are actually defined."""
    print("=" * 70)
    print("CHECK: ORACLE METHODS EXIST")
    print("=" * 70)
    print()

    model = TinyMemoryModel(vocab_size=256, model_dim=64, num_layers=1, num_slots=8, slot_dim=32)

    methods = [
        "__call__",
        "forward_oracle_routing",
        "forward_oracle_storage",
        "forward_oracle_read",
    ]

    for method_name in methods:
        exists = hasattr(model, method_name)
        callable_method = callable(getattr(model, method_name, None))
        status = "✓" if exists and callable_method else "✗"
        print(f"{status} {method_name:30s}: exists={exists}, callable={callable_method}")


if __name__ == "__main__":
    check_oracle_methods_exist()
    print()
    debug_oracle_routing()
    print()
    debug_oracle_ablations()
