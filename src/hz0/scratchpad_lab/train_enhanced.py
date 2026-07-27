"""
Enhanced training: 500 steps per stage, detailed metrics per stage.

Goal: Reach 95%+ recall on all curriculum stages to pass gates.
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.nn import losses as mlx_losses
import numpy as np
from typing import Dict
import time

from src.hz0.scratchpad_lab.tiny_memory_model import TinyMemoryModel
from src.hz0.scratchpad_lab.test_tiny_model import MemoryCurriculumStage


def train_enhanced_stage(stage: MemoryCurriculumStage, num_steps: int = 500) -> Dict:
    """Train one stage with detailed metrics."""
    model = TinyMemoryModel(vocab_size=256, model_dim=64, num_layers=1, num_slots=8, slot_dim=32)
    optimizer = optim.Adam(learning_rate=2e-3)  # Slightly higher LR

    recalls = []
    losses = []

    for step in range(num_steps):
        # Training batch
        seq_train, target_train = stage.generate_batch(batch_size=2, held_out=False)

        def loss_fn(m):
            logits, _, _ = m(seq_train)
            read_phase_start = stage.seq_len // 2
            pred = logits[:, read_phase_start:, :]
            targ = target_train[:, read_phase_start:]
            loss = mx.mean(mlx_losses.cross_entropy(pred, targ))
            return loss

        loss_val, grads = nn.value_and_grad(model, loss_fn)(model)
        optimizer.update(model, grads)
        mx.eval(loss_val)
        losses.append(float(loss_val))

        # Validation every 25 steps
        if step % 25 == 0:
            seq_val, target_val = stage.generate_batch(batch_size=1, held_out=False)
            read_start = stage.seq_len // 2

            # Batch recall
            logits, _, _ = model(seq_val)
            pred = mx.argmax(logits[0, read_start, :])
            recall = float(pred == target_val[0, read_start])
            recalls.append(recall)

    return {
        "stage": stage.name,
        "final_recall": recalls[-1] if recalls else 0,
        "max_recall": max(recalls) if recalls else 0,
        "mean_recall": np.mean(recalls) if recalls else 0,
        "min_loss": min(losses) if losses else 0,
        "final_loss": losses[-1] if losses else 0,
    }


def run_enhanced_training():
    """Train model on all 7 stages with enhanced metrics."""
    print("=" * 70)
    print("ENHANCED CURRICULUM TRAINING (500 steps/stage)")
    print("=" * 70)
    print()

    stages = [
        MemoryCurriculumStage("fixed_key_value", num_training_examples=100),
        MemoryCurriculumStage("multiple_keys", num_training_examples=200),
        MemoryCurriculumStage("random_values", num_training_examples=200),
        MemoryCurriculumStage("distractors", num_training_examples=200),
        MemoryCurriculumStage("overwrite", num_training_examples=200),
        MemoryCurriculumStage("protected", num_training_examples=200),
        MemoryCurriculumStage("distance", num_training_examples=200),
    ]

    results_all = []
    start_time = time.time()

    for stage in stages:
        print(f"Training: {stage.name:20s}...", end="", flush=True)
        stage_start = time.time()
        result = train_enhanced_stage(stage, num_steps=500)
        stage_time = time.time() - stage_start
        results_all.append(result)

        print(f" {stage_time:.1f}s")
        print(f"  Final recall:  {result['final_recall']:.0%}")
        print(f"  Max recall:    {result['max_recall']:.0%}")
        print(f"  Mean recall:   {result['mean_recall']:.0%}")
        print(f"  Min loss:      {result['min_loss']:.4f}")
        print()

    total_time = time.time() - start_time

    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Total time: {total_time:.1f}s")
    print()

    print("SUMMARY")
    print("-" * 70)
    print(f"{'Stage':20s} | {'Final':>6s} | {'Max':>6s} | {'Mean':>6s} | {'Loss':>8s}")
    print("-" * 70)
    for result in results_all:
        print(f"{result['stage']:20s} | {result['final_recall']:>6.0%} | "
              f"{result['max_recall']:>6.0%} | {result['mean_recall']:>6.0%} | "
              f"{result['min_loss']:>8.4f}")

    # Gate validation
    print()
    print("=" * 70)
    print("GATE VALIDATION (Mean recall vs thresholds)")
    print("=" * 70)
    gates = {
        "associative_recall": 0.95,
        "interference_resistance": 0.90,
        "overwrite_consistency": 0.95,
        "distance_robustness": 0.80,
    }

    gate_status = {}
    for i, gate_name in enumerate(gates):
        threshold = gates[gate_name]
        recall = results_all[i]["mean_recall"] if i < len(results_all) else 0
        passed = recall >= threshold
        status = "✓" if passed else "✗"
        gate_status[gate_name] = passed
        print(f"{status} {gate_name:30s}: {recall:.0%} >= {threshold:.0%}")

    passed_gates = sum(1 for v in gate_status.values() if v)
    total_gates = len(gate_status)
    print()
    print(f"Gates passed: {passed_gates}/{total_gates}")


if __name__ == "__main__":
    run_enhanced_training()
