"""
Metal forward kernel for GDN-2 recurrent operations.

Adapts MLX-LM's Metal GatedDeltaNet structure for trainable variants.
Grid layout:
  - One grid plane per (batch, head)
  - One grid row per value channel
  - 32 SIMD lanes cooperate over key dimension
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Tuple, Optional


def gdn2_forward_metal(
    queries: mx.array,    # [B, T, H, Dk]
    keys: mx.array,       # [B, T, H, Dk]
    values: mx.array,     # [B, T, H, Dv]
    decays: mx.array,     # [B, T, H, Dk]
    erases: mx.array,     # [B, T, H, Dk]
    writes: mx.array,     # [B, T, H, Dv]
    initial_state: Optional[mx.array] = None,  # [B, H, Dv, Dk]
    chunk_size: int = 64,  # For future chunkwise formulation
) -> Tuple[mx.array, mx.array]:
    """
    GDN-2 forward pass via Metal kernel.

    For now, falls back to MLX ops during kernel development.
    Production version will compile to native Metal.

    Returns:
        outputs: [B, T, H, Dv]
        final_state: [B, H, Dv, Dk]
    """
    # Placeholder: use MLX reference until Metal kernel ready
    from src.hz0.metal_gdn2.reference.gdn2_mlx import gdn2_sequence_ops

    return gdn2_sequence_ops(
        queries, keys, values, decays, erases, writes, initial_state
    )


class GDN2MetalModule(nn.Module):
    """
    GDN-2 layer with Metal backend.

    Wraps Metal forward kernel in a trainable module.
    Manages state persistence for streaming inference.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        num_heads: int = 1,
        chunk_size: int = 64,
    ):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim or dim
        self.num_heads = num_heads
        self.chunk_size = chunk_size

        head_dim = self.hidden_dim // num_heads
        self.head_dim = head_dim

        # Projection layers
        self.to_qkv = nn.Linear(dim, 3 * self.hidden_dim)
        self.to_decay_erase_write = nn.Linear(dim, self.hidden_dim * 3)
        self.to_out = nn.Linear(self.hidden_dim, dim)

        # State buffer for streaming
        self._state = None

    def __call__(
        self,
        x: mx.array,  # [B, T, D]
        state: Optional[mx.array] = None,
    ) -> Tuple[mx.array, mx.array]:
        """
        Forward pass.

        Args:
            x: [B, T, D] input
            state: [B, H, Dv, Dk] recurrent state (optional)

        Returns:
            output: [B, T, D]
            state: [B, H, Dv, Dk] updated state
        """
        B, T, D = x.shape

        # Project to Q/K/V
        qkv = self.to_qkv(x)  # [B, T, 3*hidden]
        qkv = mx.reshape(qkv, (B, T, 3, self.hidden_dim))
        q, k, v = mx.split(qkv, 3, axis=2)
        q = mx.squeeze(q, axis=2)  # [B, T, hidden]
        k = mx.squeeze(k, axis=2)
        v = mx.squeeze(v, axis=2)

        # Reshape to [B, T, H, Dk]
        q = mx.reshape(q, (B, T, self.num_heads, self.head_dim))
        k = mx.reshape(k, (B, T, self.num_heads, self.head_dim))
        v = mx.reshape(v, (B, T, self.num_heads, self.head_dim))

        # Decay, erase, write projections
        decay_erase_write = self.to_decay_erase_write(x)  # [B, T, 3*hidden]
        decay_erase_write = mx.reshape(
            decay_erase_write, (B, T, 3, self.hidden_dim)
        )
        d, e, w = mx.split(decay_erase_write, 3, axis=2)
        d = mx.squeeze(d, axis=2)  # [B, T, hidden]
        e = mx.squeeze(e, axis=2)
        w = mx.squeeze(w, axis=2)

        # Reshape and normalize
        d = mx.reshape(d, (B, T, self.num_heads, self.head_dim))
        d = mx.sigmoid(d)  # Decay in (0, 1)

        e = mx.reshape(e, (B, T, self.num_heads, self.head_dim))
        e = mx.sigmoid(e)  # Erase in (0, 1)

        w = mx.reshape(w, (B, T, self.num_heads, self.head_dim))
        w = mx.sigmoid(w)  # Write gate in (0, 1)

        # Run GDN-2 forward
        output, state = gdn2_forward_metal(
            q, k, v, d, e, w, initial_state=state, chunk_size=self.chunk_size
        )

        # Reshape output back to [B, T, hidden]
        output = mx.reshape(output, (B, T, self.hidden_dim))

        # Final projection
        output = self.to_out(output)  # [B, T, D]

        return output, state

    def streaming_step(
        self,
        x: mx.array,  # [B, D]
        state: Optional[mx.array] = None,
    ) -> Tuple[mx.array, mx.array]:
        """
        Single-token streaming step.
        Stateless if state passed explicitly, else uses _state buffer.

        Args:
            x: [B, D] or [B, 1, D] token embedding
            state: [B, H, Dv, Dk] or None

        Returns:
            output: [B, D]
            state: [B, H, Dv, Dk]
        """
        # Handle shape: if [B, D] add T dimension, if [B, T, D] keep it
        if len(x.shape) == 2:
            x = mx.expand_dims(x, axis=1)  # [B, 1, D]
        elif len(x.shape) == 3 and x.shape[1] != 1:
            x = mx.expand_dims(x, axis=1)  # Handle edge case

        output, state = self(x, state=state)  # Returns [B, 1, D], state
        output = mx.squeeze(output, axis=1)  # [B, D]

        self._state = state
        return output, state

    def reset_state(self):
        """Clear streaming state."""
        self._state = None
