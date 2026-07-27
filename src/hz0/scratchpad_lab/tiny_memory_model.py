"""
HZ-0B Phase 1: Tiny scratchpad laboratory model for controlled memory testing.

1-5M parameters, 64-128 model_dim, 1-3 layers, 16-64 sequence length.
Dedicated to understanding whether routing + storage + readout work
in isolation before integrating with 110M backbone.
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Tuple, Optional, Dict
import numpy as np


class TinyMemoryModel(nn.Module):
    """
    Minimal model for scratchpad memory testing.

    Architecture:
    - Token embedding (vocab_size → model_dim)
    - 1-2 small recurrent or attention blocks
    - Slot-addressed scratchpad (num_slots × slot_dim)
    - Simple prediction head

    Total ~1-5M parameters.
    """

    def __init__(
        self,
        vocab_size: int = 256,
        model_dim: int = 64,
        num_layers: int = 1,
        num_slots: int = 16,
        slot_dim: int = 32,
        num_heads: int = 2,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.model_dim = model_dim
        self.num_layers = num_layers
        self.num_slots = num_slots
        self.slot_dim = slot_dim

        # Token embedding
        self.token_embed = nn.Embedding(vocab_size, model_dim)

        # Recurrent/attention layers
        self.layers = [
            nn.MultiHeadAttention(
                dims=model_dim,
                num_heads=num_heads,
                bias=True,
            ) if i % 2 == 0 else
            nn.Linear(model_dim, model_dim)
            for i in range(num_layers)
        ]

        # Scratchpad components
        self.key_proj = nn.Linear(model_dim, slot_dim)
        self.query_proj = nn.Linear(model_dim, slot_dim)
        self.value_proj = nn.Linear(model_dim, slot_dim)
        self.write_gate_proj = nn.Linear(model_dim, 1)
        self.erase_gate_proj = nn.Linear(model_dim, 1)

        # Readout
        self.readout_proj = nn.Linear(slot_dim, model_dim)

        # Output head
        self.output_head = nn.Linear(model_dim, vocab_size)

        # Scratchpad state (registers)
        self.state_dim = (num_slots, slot_dim)

    def _get_initial_state(self, batch_size: int = 1) -> mx.array:
        """Initialize scratchpad state."""
        return mx.zeros((batch_size, self.num_slots, self.slot_dim))

    def _forward_layer(self, x: mx.array, layer_idx: int) -> mx.array:
        """Forward through one layer (attention or linear)."""
        layer = self.layers[layer_idx]
        if isinstance(layer, nn.MultiHeadAttention):
            # Self-attention: Q,K,V all from input
            return layer(x, x)
        else:
            # Linear layer
            return layer(x)

    def forward(
        self,
        input_ids: mx.array,
        state: Optional[mx.array] = None,
    ) -> Tuple[mx.array, mx.array, Dict]:
        """
        Forward pass with scratchpad memory.

        Args:
            input_ids: [batch, seq_len]
            state: [batch, num_slots, slot_dim] or None

        Returns:
            logits: [batch, seq_len, vocab_size]
            new_state: [batch, num_slots, slot_dim]
            diagnostics: routing + recall metrics
        """
        batch_size, seq_len = input_ids.shape

        if state is None:
            state = self._get_initial_state(batch_size)

        # Embed tokens
        x = self.token_embed(input_ids)  # [B, T, D]

        # Process through layers
        for i in range(self.num_layers):
            x = self._forward_layer(x, i)

        # Scratchpad processing
        logits_list = []
        diagnostics = {
            "write_slots": [],
            "read_slots": [],
            "slot_matches": [],
            "key_values": [],
            "retrieved_values": [],
        }

        for t in range(seq_len):
            x_t = x[:, t, :]  # [B, D]

            # Projections for memory operations
            key_t = self.key_proj(x_t)  # [B, slot_dim]
            query_t = self.query_proj(x_t)
            value_t = self.value_proj(x_t)

            # Compute soft routing scores
            key_expanded = mx.expand_dims(key_t, 1)  # [B, 1, slot_dim]
            scores = mx.einsum("bsd,bkd->bk", key_expanded, mx.expand_dims(mx.ones((batch_size, self.num_slots, self.slot_dim)) * state, 1))  # [B, num_slots]

            # Simplified: just match against each slot's L2 norm
            slot_norms = mx.sqrt(mx.sum(state ** 2, axis=2) + 1e-8)  # [B, num_slots]
            key_norm = mx.sqrt(mx.sum(key_t ** 2, axis=1, keepdims=True) + 1e-8)  # [B, 1]
            scores = mx.exp(-mx.abs(slot_norms - key_norm) / (self.slot_dim ** 0.5))  # [B, num_slots]

            # Hard routing (argmax)
            write_slot = mx.argmax(scores, axis=1)  # [B]
            read_slot = mx.argmax(scores, axis=1)  # [B]

            # Gates
            write_gate = mx.sigmoid(self.write_gate_proj(x_t))  # [B, 1]
            erase_gate = mx.sigmoid(self.erase_gate_proj(x_t))  # [B, 1]

            # Write: state[slot] = erase * state[slot] + write * value
            for b in range(batch_size):
                slot = int(write_slot[b])
                state_b = state[b]
                state_b[slot] = (1 - erase_gate[b]) * state_b[slot] + write_gate[b] * value_t[b]

            # Read: retrieve value from slot
            retrieved = mx.zeros_like(value_t)
            for b in range(batch_size):
                slot = int(read_slot[b])
                retrieved[b] = state[b][slot]

            # Readout + output
            readout = self.readout_proj(retrieved)  # [B, D]
            residual = x_t + readout  # Residual connection
            logit_t = self.output_head(residual)  # [B, vocab]
            logits_list.append(logit_t)

            # Record diagnostics
            match = (write_slot == read_slot).astype(mx.float32)
            diagnostics["write_slots"].append(write_slot)
            diagnostics["read_slots"].append(read_slot)
            diagnostics["slot_matches"].append(match)
            diagnostics["key_values"].append((key_t, value_t))
            diagnostics["retrieved_values"].append(retrieved)

        logits = mx.stack(logits_list, axis=1)  # [B, T, vocab]

        return logits, state, diagnostics


def create_tiny_memory_model(
    vocab_size: int = 256,
    model_dim: int = 64,
    num_layers: int = 1,
    num_slots: int = 16,
    slot_dim: int = 32,
) -> TinyMemoryModel:
    """Factory for tiny memory model."""
    return TinyMemoryModel(
        vocab_size=vocab_size,
        model_dim=model_dim,
        num_layers=num_layers,
        num_slots=num_slots,
        slot_dim=slot_dim,
    )
