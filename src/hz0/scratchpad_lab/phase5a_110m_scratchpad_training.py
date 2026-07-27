"""
Phase 5A: Train scratchpad on frozen 110M backbone.

Goal: Validate memory layer performance on real-scale model.
Target: >90% recall gates maintained on production architecture.
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


def train_phase5a(
    num_steps: int = 5000,
    eval_every: int = 100,
    checkpoint_every: int = 500,
):
    """Phase 5A: Scratchpad training on frozen 110M backbone."""
    print("=" * 70)
    print("PHASE 5A: SCRATCHPAD TRAINING (FROZEN 110M BACKBONE)")
    print("=" * 70)
    print()

    print("1. Creating hybrid model...")
    try:
        model = HZ0BHybridModel(freeze_backbone=True)
        print("   ✓ Model created (backbone frozen)")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return

    print("2. Setting up training...")
    optimizer = optim.Adam(learning_rate=2e-3)
    stage = MemoryCurriculumStage("fixed_key_value", num_training_examples=1000)

    print()
    print("3. Training loop...")
    print("-" * 70)

    losses = []
    recalls = []
    start_time = time.time()

    for step in range(num_steps):
        # Training batch
        seq_train, target_train = stage.generate_batch(batch_size=2, held_out=False)

        def loss_fn(m):
            logits, _, _ = m(seq_train)
            read_phase_start = stage.seq_len // 2
            pred = logits[:, read_phase_start:, :]
            targ = target_train[:, read_phase_start:]
            # Clip logits for numerical stability (cross_entropy is sensitive to large values)
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
            recalls.append(recall)

            elapsed = time.time() - start_time
            print(f"Step {step:5d} ({elapsed:6.1f}s): loss={float(loss_val):.4f}, recall={recall:.0%}")

        # Checkpoint
        if step > 0 and step % checkpoint_every == 0:
            print(f"         → Checkpoint at step {step}")

    print()
    print("=" * 70)
    print("PHASE 5A COMPLETE")
    print("=" * 70)
    total_time = time.time() - start_time
    print(f"Total time: {total_time:.1f}s")
    print(f"Final loss: {losses[-1]:.4f}")
    print(f"Max recall: {max(recalls) if recalls else 0:.0%}")
    print(f"Mean recall: {np.mean(recalls) if recalls else 0:.0%}")
    print()

    # Gate validation
    print("GATE STATUS (Phase 5A)")
    print("-" * 70)
    if recalls:
        final_recall = np.mean(recalls[-10:]) if len(recalls) >= 10 else np.mean(recalls)
        print(f"Memory recall: {final_recall:.0%}")
        if final_recall >= 0.90:
            print("✓ HZ-0B gates maintained on 110M backbone")
        else:
            print(f"✗ Recall below target ({final_recall:.0%} < 90%)")
    print()

    print("NEXT: Phase 5B (End-to-end fine-tuning)")


if __name__ == "__main__":
    train_phase5a(num_steps=5000, eval_every=100, checkpoint_every=500)
