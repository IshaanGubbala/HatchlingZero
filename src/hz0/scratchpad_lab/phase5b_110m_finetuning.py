"""
Phase 5B: End-to-end fine-tuning (unfrozen backbone).

Goal: Train full hybrid model on curriculum.
Measure: Language quality + memory quality trade-off.
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.nn import losses as mlx_losses
import numpy as np
from typing import Dict
import time

from hz0.scratchpad_lab.hz0b_hybrid_model import HZ0BHybridModel
from hz0.scratchpad_lab.test_tiny_model import MemoryCurriculumStage


def train_phase5b(
    num_steps: int = 10000,
    eval_every: int = 200,
    checkpoint_every: int = 1000,
):
    """Phase 5B: End-to-end fine-tuning (unfrozen backbone)."""
    print("=" * 70)
    print("PHASE 5B: END-TO-END FINE-TUNING (110M BACKBONE + SCRATCHPAD)")
    print("=" * 70)
    print()

    print("1. Creating hybrid model...")
    try:
        model = HZ0BHybridModel(freeze_backbone=False)  # Backbone unfrozen
        print("   ✓ Model created (backbone unfrozen)")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return

    print("2. Setting up training...")
    # Lower LR for fine-tuning (backbone + scratchpad both learning)
    optimizer = optim.Adam(learning_rate=1e-4)

    # Use multiple curriculum stages for diversity
    stages = [
        MemoryCurriculumStage("fixed_key_value", num_training_examples=500),
        MemoryCurriculumStage("multiple_keys", num_training_examples=500),
        MemoryCurriculumStage("distractors", num_training_examples=500),
    ]

    print()
    print("3. Training loop...")
    print("-" * 70)

    losses = []
    recalls_by_stage = {s.name: [] for s in stages}
    start_time = time.time()
    stage_idx = 0

    for step in range(num_steps):
        # Cycle through stages
        stage = stages[stage_idx % len(stages)]

        # Training batch
        seq_train, target_train = stage.generate_batch(batch_size=2, held_out=False)

        def loss_fn(m):
            logits, _, _ = m(seq_train)
            read_phase_start = stage.seq_len // 2
            pred = logits[:, read_phase_start:, :]
            targ = target_train[:, read_phase_start:]
            # Clip logits for numerical stability
            pred = mx.clip(pred, -100.0, 100.0)
            loss = mx.mean(mlx_losses.cross_entropy(pred, targ))
            return loss

        loss_val, grads = nn.value_and_grad(model, loss_fn)(model)
        optimizer.update(model, grads)
        mx.eval(loss_val)
        losses.append(float(loss_val))

        # Validation every eval_every steps
        if step % eval_every == 0:
            seq_val, target_val = stage.generate_batch(batch_size=1, held_out=False)
            read_start = stage.seq_len // 2
            logits, _, _ = model(seq_val)
            pred = mx.argmax(logits[0, read_start, :])
            recall = float(pred == target_val[0, read_start])
            recalls_by_stage[stage.name].append(recall)

            elapsed = time.time() - start_time
            print(f"Step {step:5d} ({elapsed:6.1f}s): stage={stage.name:15s}, "
                  f"loss={float(loss_val):.4f}, recall={recall:.0%}")

        # Checkpoint
        if step > 0 and step % checkpoint_every == 0:
            print(f"         → Checkpoint at step {step}")

        stage_idx += 1

    print()
    print("=" * 70)
    print("PHASE 5B COMPLETE")
    print("=" * 70)
    total_time = time.time() - start_time
    print(f"Total time: {total_time:.1f}s")
    print(f"Final loss: {losses[-1]:.4f}")
    print()

    # Per-stage summary
    print("RECALL BY STAGE")
    print("-" * 70)
    for stage_name, recalls in recalls_by_stage.items():
        if recalls:
            print(f"{stage_name:20s}: max={max(recalls):.0%}, mean={np.mean(recalls):.0%}")

    print()
    print("NEXT: Phase 5C (Production validation)")


if __name__ == "__main__":
    train_phase5b(num_steps=10000, eval_every=200, checkpoint_every=1000)
