"""
Phase 5A: Scratchpad training with fixed GDN-2 backbone.

Now that GDN-2 NaN is fixed, validate memory layer on real backbone.
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.nn import losses as mlx_losses
import numpy as np
import time

from src.hz0.scratchpad_lab.tiny_memory_model import TinyMemoryModel
from src.hz0.model_port.mlx_gdn2_lm import create_hz_36m_mlx
from src.hz0.scratchpad_lab.test_tiny_model import MemoryCurriculumStage


class HybridWithGDN2(nn.Module):
    """GDN-2 backbone + scratchpad memory."""

    def __init__(self):
        super().__init__()
        self.backbone = create_hz_36m_mlx()
        self.scratchpad = TinyMemoryModel(
            vocab_size=32768,
            model_dim=576,
            num_layers=1,
            num_slots=32,
            slot_dim=128,
        )
        self.gate_proj = nn.Linear(32768, 1)

    def __call__(self, input_ids: mx.array, memory_state: mx.array = None):
        # Backbone
        logits_backbone, _ = self.backbone(input_ids)

        # Scratchpad
        if memory_state is None:
            batch_size = input_ids.shape[0]
            memory_state = self.scratchpad._get_initial_state(batch_size)
        logits_memory, new_memory, _ = self.scratchpad(input_ids, memory_state)

        # Learned fusion gate
        gate_logits = self.gate_proj(logits_backbone)
        gate = mx.sigmoid(gate_logits)

        # Fuse
        logits_fused = gate * logits_memory + (1 - gate) * logits_backbone

        return logits_fused, new_memory


def train_phase5a_gdn2(num_steps: int = 500, eval_every: int = 50):
    """Phase 5A with GDN-2 backbone."""
    print("=" * 70)
    print("PHASE 5A: GDN-2 BACKBONE + SCRATCHPAD")
    print("=" * 70)
    print()

    print("1. Creating hybrid model...")
    try:
        model = HybridWithGDN2()
        print("   ✓ Model created")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return

    print("2. Setting up training...")
    optimizer = optim.Adam(learning_rate=5e-4)
    stage = MemoryCurriculumStage("fixed_key_value", num_training_examples=100)

    print("3. Training...")
    print("-" * 70)

    losses = []
    recalls = []
    start = time.time()

    for step in range(num_steps):
        seq, tgt = stage.generate_batch(batch_size=1, held_out=False)

        def loss_fn(m):
            logits, _ = m(seq)
            read_start = stage.seq_len // 2
            pred = logits[:, read_start:, :]
            targ = tgt[:, read_start:]
            # Clip logits for stability
            pred = mx.clip(pred, -100.0, 100.0)
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
            recalls.append(recall)

            elapsed = time.time() - start
            print(f"Step {step:4d} ({elapsed:6.1f}s): loss={float(loss_val):.4f}, recall={recall:.0%}")

    print()
    print("=" * 70)
    print("PHASE 5A COMPLETE")
    print("=" * 70)
    print(f"Total time: {time.time() - start:.1f}s")
    print(f"Final loss: {losses[-1]:.4f}")
    print(f"Mean recall: {np.mean(recalls):.0%}")
    print()

    if np.mean(recalls) >= 0.80:
        print("✓ Gates maintained on GDN-2 backbone")
    else:
        print(f"✗ Recall below threshold ({np.mean(recalls):.0%})")


if __name__ == "__main__":
    train_phase5a_gdn2(num_steps=500, eval_every=50)
