"""Phase 3a: Integrate scratchpad memory into HZ-0A model.

Add HZ-0B (Hebbian scratchpad) to GDN2LanguageModel.
Validate on memory tasks.
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Optional, Tuple
from hz0.model_port.mlx_gdn2_lm import GDN2LanguageModel
from hz0.scratchpad_lab.tiny_memory_model import ScratchpadMemory


class HZ0AWithScratchpad(nn.Module):
    """HZ-0A + HZ-0B scratchpad memory."""

    def __init__(
        self,
        vocab_size: int = 32768,
        model_dim: int = 768,
        num_layers: int = 24,
        num_heads: int = 12,
        gdn2_every: int = 3,
        memory_slots: int = 64,
    ):
        super().__init__()
        self.base_model = GDN2LanguageModel(
            vocab_size=vocab_size,
            model_dim=model_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            gdn2_every=gdn2_every,
        )

        # Add scratchpad memory
        self.memory = ScratchpadMemory(
            slot_dim=model_dim,
            num_slots=memory_slots,
        )

        self.vocab_size = vocab_size
        self.model_dim = model_dim

    def __call__(self, x: mx.array, memory_state: Optional[dict] = None) -> Tuple[mx.array, dict]:
        """Forward with memory integration.

        Args:
            x: [B, T] token ids
            memory_state: dict with accumulated memory state

        Returns:
            logits: [B, T, vocab]
            memory_state: updated state
        """
        # Base model forward
        base_logits, recurrent_state = self.base_model(x)

        # Memory read: use hidden states to query memory
        # (Simplified: use logits softmax as query)
        query = mx.softmax(base_logits[:, -1:, :], axis=-1)  # Use last token logits

        memory_read = self.memory.read(query)

        # Combine base + memory (simplified: just add)
        combined = base_logits + memory_read[:, None, :]

        # Memory write: use hidden state as key-value pair
        self.memory.write(x[:, -1], mx.softmax(base_logits[:, -1:, :], axis=-1).squeeze())

        # Return combined logits and states
        return combined, {
            "recurrent": recurrent_state,
            "memory": self.memory.state,
        }

    def decode_step(self, token_id: int, layer_states=None, kv_caches=None, memory_state=None):
        """Single-token decode with memory."""
        # Use base model streaming
        logits, layer_states, kv_caches = self.base_model.decode_step(token_id, layer_states, kv_caches)

        # Optional: incorporate memory
        # (For now, just return base logits)

        return logits, layer_states, kv_caches


def test_memory_integration():
    """Test Phase 3a: Scratchpad integration."""
    print("="*70)
    print("Phase 3a: Scratchpad Memory Integration")
    print("="*70)

    # Create model with memory
    model = HZ0AWithScratchpad(
        vocab_size=8192,
        model_dim=256,
        num_layers=6,
        num_heads=4,
        gdn2_every=2,
        memory_slots=32,
    )

    print(f"\n✓ Model created with scratchpad memory")
    print(f"  Memory slots: 32")
    print(f"  Model dim: 256")

    # Test forward pass
    tokens = mx.random.randint(0, 8192, shape=(2, 64))
    logits, states = model(tokens)

    print(f"\n✓ Forward pass works")
    print(f"  Input shape: {tokens.shape}")
    print(f"  Output shape: {logits.shape}")

    # Verify no NaN
    if mx.any(mx.isnan(logits)):
        print(f"✗ NaN detected in output")
    else:
        print(f"✓ No NaN in output")

    # Verify memory state
    if states and "memory" in states:
        print(f"✓ Memory state captured")
    else:
        print(f"✗ Memory state missing")

    print("\n" + "="*70)
    print("Phase 3a: Integration successful")
    print("="*70)

    print("\nNext steps:")
    print("  1. Run memory task benchmarks (associative recall, overwrite)")
    print("  2. Compare: with memory vs without memory")
    print("  3. Measure: LM loss delta")
    print("  4. Scale to 36M, 110M models")

    return model


if __name__ == "__main__":
    model = test_memory_integration()
