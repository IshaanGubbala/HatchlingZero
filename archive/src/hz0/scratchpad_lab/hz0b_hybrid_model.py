"""
HZ-0B hybrid model: 110M backbone + scratchpad memory.

Learned gating fusion for flexible routing between backbone and memory.
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Tuple, Optional, Dict

from hz0.model_port.mlx_gdn2_lm import create_hz_36m_mlx
from hz0.scratchpad_lab.tiny_memory_model import TinyMemoryModel


class HZ0BHybridModel(nn.Module):
    """Full HZ-0B hybrid: backbone + scratchpad with learned fusion."""

    def __init__(
        self,
        backbone_model_dim: int = 768,
        scratchpad_model_dim: int = 768,
        vocab_size: int = 32768,
        num_slots: int = 64,
        slot_dim: int = 128,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.backbone = create_hz_36m_mlx()
        self.backbone_frozen = freeze_backbone

        # Scratchpad memory layer
        self.scratchpad = TinyMemoryModel(
            vocab_size=vocab_size,
            model_dim=scratchpad_model_dim,
            num_layers=1,
            num_slots=num_slots,
            slot_dim=slot_dim,
        )

        # Learned gating fusion: vocab_size → 1
        self.fusion_gate_proj = nn.Linear(vocab_size, 1)

    def __call__(
        self,
        input_ids: mx.array,
        memory_state: mx.array = None,
    ) -> Tuple[mx.array, mx.array, Dict]:
        """
        Forward pass with fusion.

        Args:
            input_ids: [batch, seq_len]
            memory_state: [batch, num_slots, slot_dim] or None

        Returns:
            logits: [batch, seq_len, vocab_size]
            memory_state: Updated memory state
            diagnostics: Routing + gate metrics
        """
        # Backbone forward
        logits_backbone, _ = self.backbone(input_ids)  # [B, T, vocab]

        # Initialize memory state if needed
        if memory_state is None:
            batch_size = input_ids.shape[0]
            memory_state = self.scratchpad._get_initial_state(batch_size)

        # Scratchpad forward
        logits_memory, new_memory, scratchpad_diag = self.scratchpad(
            input_ids, memory_state
        )

        # Learned fusion gate: weighted logits confidence
        batch_size, seq_len, vocab_size = logits_backbone.shape
        # Use learned parameter to weight backbone vs scratchpad
        # For now: simple learned scalar gate per sequence position
        gate_logits = self.fusion_gate_proj(logits_backbone)  # [B, T, 1]
        gate = mx.sigmoid(gate_logits)  # [B, T, 1]

        # Fused logits: gate * scratchpad + (1-gate) * backbone
        logits_fused = gate * logits_memory + (1 - gate) * logits_backbone

        return logits_fused, new_memory, {
            "backbone": logits_backbone,
            "memory": logits_memory,
            "fused": logits_fused,
            "gate": gate,
            "scratchpad_diag": scratchpad_diag,
        }

    def freeze_backbone(self, freeze: bool = True):
        """Freeze/unfreeze backbone weights."""
        for param in self.backbone.parameters():
            if freeze:
                param.requires_grad = False
            else:
                param.requires_grad = True


class HZ0BTrainer:
    """Training loop for HZ-0B hybrid model."""

    def __init__(self, model: HZ0BHybridModel, learning_rate: float = 1e-3):
        self.model = model
        self.optimizer = nn.optimizers.Adam(learning_rate=learning_rate)

    def train_step(self, batch_ids, batch_targets):
        """Single training step."""
        def loss_fn(m):
            logits, _, _ = m(batch_ids)
            loss = mx.mean(nn.losses.cross_entropy(logits, batch_targets))
            return loss

        loss_val, grads = nn.value_and_grad(self.model, loss_fn)(self.model)
        self.optimizer.update(self.model, grads)
        mx.eval(loss_val)
        return float(loss_val)

    def eval_step(self, batch_ids, batch_targets):
        """Evaluation step."""
        logits, _, _ = self.model(batch_ids)
        loss = mx.mean(nn.losses.cross_entropy(logits, batch_targets))
        preds = mx.argmax(logits, axis=-1)
        accuracy = mx.mean((preds == batch_targets).astype(mx.float32))
        return float(loss), float(accuracy)


# Phase A: Train scratchpad only (backbone frozen)
def train_phase_a(
    model: HZ0BHybridModel,
    train_data,  # Iterator of (batch_ids, batch_targets)
    num_steps: int = 1000,
    eval_every: int = 100,
):
    """Phase A: Scratchpad training on frozen backbone."""
    model.freeze_backbone(True)
    trainer = HZ0BTrainer(model, learning_rate=2e-3)

    print("=" * 70)
    print("PHASE A: TRAIN SCRATCHPAD (FROZEN BACKBONE)")
    print("=" * 70)
    print()

    for step, (batch_ids, batch_targets) in enumerate(train_data):
        if step >= num_steps:
            break

        loss = trainer.train_step(batch_ids, batch_targets)

        if step % eval_every == 0:
            print(f"Step {step:5d}: loss={loss:.4f}")

    print()
    print("Phase A complete!")


# Phase B: End-to-end fine-tuning (backbone unfrozen)
def train_phase_b(
    model: HZ0BHybridModel,
    train_data,
    num_steps: int = 5000,
    eval_every: int = 500,
):
    """Phase B: End-to-end fine-tuning."""
    model.freeze_backbone(False)
    trainer = HZ0BTrainer(model, learning_rate=5e-4)

    print("=" * 70)
    print("PHASE B: END-TO-END FINE-TUNING")
    print("=" * 70)
    print()

    for step, (batch_ids, batch_targets) in enumerate(train_data):
        if step >= num_steps:
            break

        loss = trainer.train_step(batch_ids, batch_targets)

        if step % eval_every == 0:
            print(f"Step {step:5d}: loss={loss:.4f}")

    print()
    print("Phase B complete!")


if __name__ == "__main__":
    # Quick test
    print("Creating HZ-0B hybrid model...")
    model = HZ0BHybridModel()
    print("✓ Model created")

    print("Testing forward pass...")
    import numpy as np
    batch_ids = mx.array(np.random.randint(0, 32768, (1, 128)), dtype=mx.int32)
    logits, state, diag = model(batch_ids)
    print(f"✓ Forward pass: logits {logits.shape}, state {state.shape}")

    print()
    print("Ready for Phase A & B training on 110M backbone!")
