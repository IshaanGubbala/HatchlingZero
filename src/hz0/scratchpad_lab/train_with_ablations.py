"""
Train tiny model on full curriculum with oracle ablation tracking.

Measures: Which ablation shows improvement? Identifies bottleneck.
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.nn import losses as mlx_losses
import numpy as np
from typing import Dict
import time

from src.hz0.scratchpad_lab.tiny_memory_model import TinyMemoryModel
from src.hz0.scratchpad_lab.test_tiny_model import MemoryCurriculumStage, train_stage


def train_with_oracle_tracking(stage: MemoryCurriculumStage, num_steps: int = 100) -> Dict:
    """Train stage, track oracle ablations."""
    model = TinyMemoryModel(vocab_size=256, model_dim=64, num_layers=1, num_slots=8, slot_dim=32)
    optimizer = optim.Adam(learning_rate=1e-3)

    baseline_recalls = []
    oracle_routing_recalls = []
    oracle_storage_recalls = []
    oracle_read_recalls = []

    for step in range(num_steps):
        # Training batch
        seq_train, target_train = stage.generate_batch(batch_size=1, held_out=False)

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

        # Validation every 10 steps
        if step % 20 == 0:
            seq_val, target_val = stage.generate_batch(batch_size=1, held_out=False)
            read_start = stage.seq_len // 2

            # Baseline
            logits, _, _ = model(seq_val)
            pred = mx.argmax(logits[0, read_start, :])
            baseline_recalls.append(float(pred == target_val[0, read_start]))

            # Oracle routing
            logits, _, _ = model.forward_oracle_routing(seq_val)
            pred = mx.argmax(logits[0, read_start, :])
            oracle_routing_recalls.append(float(pred == target_val[0, read_start]))

            # Oracle storage
            logits, _, _ = model.forward_oracle_storage(seq_val)
            pred = mx.argmax(logits[0, read_start, :])
            oracle_storage_recalls.append(float(pred == target_val[0, read_start]))

            # Oracle read
            logits, _, _ = model.forward_oracle_read(seq_val)
            pred = mx.argmax(logits[0, read_start, :])
            oracle_read_recalls.append(float(pred == target_val[0, read_start]))

    return {
        "stage": stage.name,
        "baseline": np.mean(baseline_recalls) if baseline_recalls else 0,
        "oracle_routing": np.mean(oracle_routing_recalls) if oracle_routing_recalls else 0,
        "oracle_storage": np.mean(oracle_storage_recalls) if oracle_storage_recalls else 0,
        "oracle_read": np.mean(oracle_read_recalls) if oracle_read_recalls else 0,
    }


def run_full_training():
    """Train model on all 7 curriculum stages with oracle tracking."""
    print("=" * 70)
    print("FULL CURRICULUM TRAINING WITH ORACLE ABLATIONS")
    print("=" * 70)
    print()

    stages = [
        MemoryCurriculumStage("fixed_key_value", num_training_examples=50),
        MemoryCurriculumStage("multiple_keys", num_training_examples=100),
        MemoryCurriculumStage("random_values", num_training_examples=100),
        MemoryCurriculumStage("distractors", num_training_examples=100),
        MemoryCurriculumStage("overwrite", num_training_examples=100),
        MemoryCurriculumStage("protected", num_training_examples=100),
        MemoryCurriculumStage("distance", num_training_examples=100),
    ]

    results_all = []

    for stage in stages:
        print(f"Training: {stage.name}")
        result = train_with_oracle_tracking(stage, num_steps=100)
        results_all.append(result)
        print(f"  Baseline:       {result['baseline']:.0%}")
        print(f"  Oracle routing: {result['oracle_routing']:.0%}")
        print(f"  Oracle storage: {result['oracle_storage']:.0%}")
        print(f"  Oracle read:    {result['oracle_read']:.0%}")

        # Identify bottleneck
        baseline = result['baseline']
        routing_boost = result['oracle_routing'] - baseline
        storage_boost = result['oracle_storage'] - baseline
        read_boost = result['oracle_read'] - baseline

        if routing_boost > 0.2:
            print(f"  ✗ ROUTING bottleneck (+{routing_boost:.0%})")
        if storage_boost > 0.2:
            print(f"  ✗ STORAGE bottleneck (+{storage_boost:.0%})")
        if read_boost > 0.2:
            print(f"  ✗ READ routing bottleneck (+{read_boost:.0%})")
        if baseline > 0.8:
            print(f"  ✓ Passed!")
        print()

    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    for result in results_all:
        print(f"{result['stage']:20s}: baseline={result['baseline']:.0%}, "
              f"routing={result['oracle_routing']:.0%}, "
              f"storage={result['oracle_storage']:.0%}, "
              f"read={result['oracle_read']:.0%}")


if __name__ == "__main__":
    run_full_training()
