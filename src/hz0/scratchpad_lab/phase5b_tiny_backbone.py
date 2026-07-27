"""
Phase 5B: End-to-end fine-tuning (scratchpad + backbone both learning).

Goal: Train hybrid model on curriculum with both components unfrozen.
Measure: Joint learning + memory gate maintenance.
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.nn import losses as mlx_losses
import numpy as np
import time

from src.hz0.scratchpad_lab.tiny_memory_model import TinyMemoryModel
from src.hz0.scratchpad_lab.test_tiny_model import MemoryCurriculumStage


class TinyHybridModel(nn.Module):
    """Hybrid: tiny backbone + scratchpad memory (both learning)."""

    def __init__(self):
        super().__init__()
        self.backbone = TinyMemoryModel(
            vocab_size=256,
            model_dim=128,
            num_layers=2,
            num_slots=32,
            slot_dim=64,
        )
        self.scratchpad = TinyMemoryModel(
            vocab_size=256,
            model_dim=128,
            num_layers=1,
            num_slots=32,
            slot_dim=64,
        )
        self.gate_proj = nn.Linear(256, 1)

    def __call__(self, input_ids: mx.array, memory_state: mx.array = None):
        logits_backbone, _, _ = self.backbone(input_ids)
        if memory_state is None:
            batch_size = input_ids.shape[0]
            memory_state = self.scratchpad._get_initial_state(batch_size)
        logits_memory, new_memory, _ = self.scratchpad(input_ids, memory_state)
        gate_logits = self.gate_proj(logits_backbone)
        gate = mx.sigmoid(gate_logits)
        logits_fused = gate * logits_memory + (1 - gate) * logits_backbone
        return logits_fused, new_memory


def train_phase5b(num_steps: int = 5000, eval_every: int = 200):
    """Phase 5B: End-to-end fine-tuning."""
    print("=" * 70)
    print("PHASE 5B: END-TO-END FINE-TUNING (TINY BACKBONE)")
    print("=" * 70)
    print()

    print("Creating model...")
    model = TinyHybridModel()
    optimizer = optim.Adam(learning_rate=1e-4)  # Lower LR for joint learning

    # Cycle through curriculum stages
    stages = [
        MemoryCurriculumStage("fixed_key_value", num_training_examples=200),
        MemoryCurriculumStage("multiple_keys", num_training_examples=200),
        MemoryCurriculumStage("distractors", num_training_examples=200),
    ]

    print("Training...")
    print("-" * 70)

    losses = []
    recalls_by_stage = {s.name: [] for s in stages}
    start = time.time()

    for step in range(num_steps):
        stage = stages[step % len(stages)]
        seq, tgt = stage.generate_batch(batch_size=2, held_out=False)

        def loss_fn(m):
            logits, _ = m(seq)
            read_start = stage.seq_len // 2
            pred = logits[:, read_start:, :]
            targ = tgt[:, read_start:]
            return mx.mean(mlx_losses.cross_entropy(pred, targ))

        loss_val, grads = nn.value_and_grad(model, loss_fn)(model)
        optimizer.update(model, grads)
        mx.eval(loss_val)
        losses.append(float(loss_val))

        if step % eval_every == 0:
            seq_v, tgt_v = stage.generate_batch(batch_size=1, held_out=False)
            logits, _ = model(seq_v)
            read_start = stage.seq_len // 2
            pred = mx.argmax(logits[0, read_start, :])
            recall = float(pred == tgt_v[0, read_start])
            recalls_by_stage[stage.name].append(recall)

            elapsed = time.time() - start
            print(f"Step {step:4d} ({elapsed:6.1f}s): {stage.name:15s} loss={float(loss_val):.4f}, recall={recall:.0%}")

    print()
    print("=" * 70)
    print("PHASE 5B COMPLETE")
    print("=" * 70)
    print(f"Total time: {time.time() - start:.1f}s")
    print(f"Final loss: {losses[-1]:.4f}")
    print()

    print("RECALL BY STAGE")
    print("-" * 70)
    for stage_name, recalls in recalls_by_stage.items():
        if recalls:
            print(f"{stage_name:20s}: max={max(recalls):.0%}, mean={np.mean(recalls):.0%}")

    print()
    print("GATE VALIDATION")
    print("-" * 70)
    avg_recall = np.mean([np.mean(r) for r in recalls_by_stage.values() if r])
    print(f"Average recall: {avg_recall:.0%}")
    if avg_recall >= 0.90:
        print("✓ End-to-end fine-tuning maintained gates")
    else:
        print(f"⚠ Recall dropped to {avg_recall:.0%}")

    print()
    print("NEXT: Phase 5C (Production validation)")


if __name__ == "__main__":
    train_phase5b(num_steps=5000, eval_every=200)
