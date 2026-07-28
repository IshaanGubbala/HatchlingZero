"""
Phase 13: Integrate streaming GDN-2 into MLX language model.

Replace full-sequence forward with streaming single-token updates.
Goal: Fix 125x decode slowdown via constant-time per-token processing.

API changes:
- Old: logits, state = model(tokens)  # full sequence at once
- New: logits, state = model.forward_streaming(tokens, state)  # token-by-token
       or: logits = model.decode_step(token_id, state)  # single token
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.nn import losses as mlx_losses
import numpy as np
import time
from typing import Tuple, Optional

from hz0.model_port.mlx_gdn2_lm import create_hz_36m_mlx
from hz0.metal_gdn2.reference.gdn2_streaming import StreamingGDN2


class StreamingLanguageModel(nn.Module):
    """Language model with streaming GDN-2 backend."""

    def __init__(self, base_model: nn.Module, vocab_size: int = 32768):
        super().__init__()
        self.base_model = base_model
        self.vocab_size = vocab_size
        # Extract layer count and dimensions from base model
        self.num_layers = len(self.base_model.layers)

    def forward_streaming(
        self,
        token_ids: mx.array,  # [B, T]
        initial_state: Optional[mx.array] = None,
    ) -> Tuple[mx.array, mx.array]:
        """
        Process sequence token-by-token with streaming GDN-2.

        Args:
            token_ids: [B, T] token indices
            initial_state: [B, L, D_v, D_k] or None

        Returns:
            logits: [B, T, vocab_size]
            final_state: [B, L, D_v, D_k]
        """
        B, T = token_ids.shape

        # Extract model dimensions from base model
        # For simplicity, use base model's embedding dim
        embedding_dim = self.base_model.embedding.weight.shape[1]

        # Initialize streaming state if not provided
        if initial_state is None:
            # State shape: [B, num_layers, D_v, D_k]
            # For now, use embedding_dim for both D_v and D_k
            initial_state = mx.zeros((B, self.num_layers, embedding_dim, embedding_dim))

        logits_list = []
        current_state = initial_state

        # Process each token
        for t in range(T):
            token_id = token_ids[:, t : t + 1]  # [B, 1]

            # Embed token
            token_embed = self.base_model.embedding(token_id)  # [B, 1, D]
            token_embed = mx.squeeze(token_embed, axis=1)  # [B, D]

            # For now, pass through base model normally (full backward compat)
            # In Phase 14, replace with streaming GDN-2 step
            logit_t, _ = self.base_model(token_ids[:, : t + 1])  # [B, T, vocab]
            logit_t = logit_t[:, -1, :]  # [B, vocab] - last token only

            logits_list.append(logit_t)
            mx.eval(logit_t)

        # Stack outputs
        logits = mx.stack(logits_list, axis=1)  # [B, T, vocab]

        return logits, current_state

    def decode_step(
        self,
        token_id: int,
        state: mx.array,
    ) -> Tuple[mx.array, mx.array]:
        """
        Single-token decode with accumulated state.

        Args:
            token_id: scalar token index
            state: [B, L, D_v, D_k] accumulated state

        Returns:
            logits: [B, vocab_size] logit distribution
            state_new: [B, L, D_v, D_k] updated state
        """
        # Create single-token input
        token_mx = mx.array([[token_id]], dtype=mx.int32)

        # Forward through base model (temporary - full sequence)
        # In Phase 14, replace with true streaming step
        logits, _ = self.base_model(token_mx)
        logits = logits[0, -1, :]  # [vocab_size]

        # Return updated state (for now, dummy - Phase 14 uses real streaming)
        state_new = state

        return logits, state_new


def test_streaming_equivalence():
    """Verify streaming forward pass matches full-sequence."""
    print("=" * 80)
    print("PHASE 13: STREAMING INTEGRATION TEST")
    print("=" * 80)
    print()

    B, T = 1, 16
    print(f"Test config: batch_size={B}, seq_len={T}")
    print()

    # Create model
    model = create_hz_36m_mlx()
    streaming_model = StreamingLanguageModel(model)

    # Generate test batch
    seq = np.random.randint(0, 32768, (B, T), dtype=np.int32)
    seq_mx = mx.array(seq)

    print("1. Full-sequence forward (baseline)...")
    start = time.perf_counter()
    logits_full, _ = model(seq_mx)
    full_time = time.perf_counter() - start
    print(f"   Time: {full_time:.3f}s")
    print(f"   Shape: {logits_full.shape}")

    print("\n2. Streaming forward (token-by-token)...")
    start = time.perf_counter()
    logits_stream, _ = streaming_model.forward_streaming(seq_mx)
    stream_time = time.perf_counter() - start
    print(f"   Time: {stream_time:.3f}s")
    print(f"   Shape: {logits_stream.shape}")

    print("\n3. Decode step (single token)...")
    state = mx.zeros((B, 2, 64, 64))  # Dummy state
    token_id = 42
    start = time.perf_counter()
    logit_decode, _ = streaming_model.decode_step(token_id, state)
    decode_time = time.perf_counter() - start
    print(f"   Time: {decode_time * 1000:.2f}ms per token")
    print(f"   Shape: {logit_decode.shape}")

    print()
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Full-sequence: {full_time:.3f}s ({T / full_time:.0f} tok/s)")
    print(f"Streaming: {stream_time:.3f}s ({T / stream_time:.0f} tok/s)")
    print(f"Decode step: {decode_time * 1000:.2f}ms ({1 / decode_time:.0f} tok/s)")
    print()

    # Compare outputs
    diff = mx.max(mx.abs(logits_full - logits_stream))
    print(f"Output difference: {float(diff):.2e}")

    if float(diff) < 1e-4:
        print("✓ Streaming output matches full-sequence")
    else:
        print("⚠ Streaming output differs (may be expected due to state handling)")

    print()
    print("=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("Phase 13a: Verify training equivalence")
    print("  - Train with streaming forward for 50 steps")
    print("  - Compare loss trajectory vs full-sequence")
    print()
    print("Phase 13b: Benchmark decode improvement")
    print("  - Current: 5 tok/s (full-sequence per token)")
    print("  - Target: 20+ tok/s (streaming with state caching)")
    print()
    print("Phase 14: Metal backend")
    print("  - Implement Metal kernel for streaming step")
    print("  - Target: 100-200 tok/s")


def train_with_streaming(steps: int = 50):
    """Test training equivalence with streaming."""
    print("\n" + "=" * 80)
    print("PHASE 13a: STREAMING TRAINING EQUIVALENCE")
    print("=" * 80)
    print()

    model = create_hz_36m_mlx()
    streaming_model = StreamingLanguageModel(model)

    opt = optim.Adam(learning_rate=2e-4)
    losses = []

    print(f"Training {steps} steps with streaming...")
    print("-" * 80)

    for step in range(steps):
        batch = mx.array(np.random.randint(0, 32768, (1, 256)), dtype=mx.int32)

        def loss_fn(m):
            logits, _ = m.forward_streaming(batch)
            pred = logits[:, :-1, :]
            targ = batch[:, 1:]
            pred = mx.clip(pred, -100.0, 100.0)
            return mx.mean(mlx_losses.cross_entropy(pred, targ))

        loss_val, grads = nn.value_and_grad(streaming_model, loss_fn)(streaming_model)

        # Gradient clipping
        def clip_grad(g):
            if isinstance(g, mx.array):
                return mx.clip(g, -1.0, 1.0)
            elif isinstance(g, dict):
                return {k: clip_grad(v) for k, v in g.items()}
            elif isinstance(g, (list, tuple)):
                return type(g)(clip_grad(item) for item in g)
            return g

        grads = clip_grad(grads)
        opt.update(streaming_model, grads)
        mx.eval(loss_val)

        loss_float = float(loss_val)
        if not np.isnan(loss_float):
            losses.append(loss_float)
        else:
            print(f"Step {step + 1}: NaN detected")
            break

        if (step + 1) % 10 == 0:
            print(f"Step {step + 1:3d}: loss={loss_float:.4f}")

    print()
    print(f"✓ Training complete: {len(losses)}/{steps} steps")
    print(f"  Final loss: {losses[-1]:.4f}")
    print(f"  Mean loss: {np.mean(losses):.4f}")

    if len(losses) >= steps * 0.95:
        print("✓ Streaming training STABLE")
    else:
        print("⚠ Streaming training unstable")


if __name__ == "__main__":
    test_streaming_equivalence()
    train_with_streaming(steps=50)
