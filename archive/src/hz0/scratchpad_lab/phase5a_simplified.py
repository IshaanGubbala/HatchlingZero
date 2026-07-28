"""
Phase 5A Simplified: Test scratchpad learning on dummy backbone.

Bypass broken GDN2 model. Use random logits + learned fusion.
Goal: Validate scratchpad learns independent of backbone quality.
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.nn import losses as mlx_losses
import numpy as np
import time

from hz0.scratchpad_lab.tiny_memory_model import TinyMemoryModel
from hz0.scratchpad_lab.test_tiny_model import MemoryCurriculumStage


class SimplifiedHybridModel(nn.Module):
    """Hybrid: dummy backbone + real scratchpad."""

    def __init__(self, vocab_size: int = 256):
        super().__init__()
        self.vocab_size = vocab_size
        self.backbone_linear = nn.Linear(128, vocab_size)  # Dummy backbone
        self.scratchpad = TinyMemoryModel(
            vocab_size=vocab_size,
            model_dim=64,
            num_layers=1,
            num_slots=16,
            slot_dim=32,
        )
        self.gate_proj = nn.Linear(vocab_size, 1)

    def __call__(self, input_ids: mx.array, memory_state: mx.array = None):
        batch_size, seq_len = input_ids.shape

        # Dummy backbone: random projection
        random_features = mx.random.normal((batch_size, seq_len, 128))
        logits_backbone = self.backbone_linear(random_features)  # [B, T, vocab]

        # Scratchpad
        if memory_state is None:
            memory_state = self.scratchpad._get_initial_state(batch_size)
        logits_memory, new_memory, _ = self.scratchpad(input_ids, memory_state)

        # Learned fusion gate
        gate_logits = self.gate_proj(logits_backbone)  # [B, T, 1]
        gate = mx.sigmoid(gate_logits)

        # Fuse (both logits same shape now)
        logits_fused = gate * logits_memory + (1 - gate) * logits_backbone

        return logits_fused, new_memory


def train_phase5a_simplified(num_steps: int = 1000):
    """Train scratchpad on dummy backbone."""
    print("=" * 70)
    print("PHASE 5A SIMPLIFIED: SCRATCHPAD ON DUMMY BACKBONE")
    print("=" * 70)
    print()

    print("Creating model...")
    model = SimplifiedHybridModel()
    optimizer = optim.Adam(learning_rate=2e-3)
    stage = MemoryCurriculumStage("fixed_key_value", num_training_examples=100)

    print("Training...")
    print("-" * 70)

    losses = []
    recalls = []
    start = time.time()

    for step in range(num_steps):
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

        if step % 50 == 0:
            seq_v, tgt_v = stage.generate_batch(batch_size=1, held_out=False)
            logits, _ = model(seq_v)
            read_start = stage.seq_len // 2
            pred = mx.argmax(logits[0, read_start, :])
            recall = float(pred == tgt_v[0, read_start])
            recalls.append(recall)

            elapsed = time.time() - start
            print(f"Step {step:4d} ({elapsed:6.1f}s): loss={float(loss_val):.4f}, recall={recall:.0%}")

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Final loss: {losses[-1]:.4f}")
    print(f"Mean recall: {np.mean(recalls):.0%}")
    print()

    if np.mean(recalls) >= 0.8:
        print("✓ Scratchpad learning works on dummy backbone")
        print("  → Backbone model (GDN2) needs debugging")
    else:
        print("✗ Scratchpad not learning")
        print("  → Architecture issue")


if __name__ == "__main__":
    train_phase5a_simplified(num_steps=1000)
