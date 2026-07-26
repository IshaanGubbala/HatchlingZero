from __future__ import annotations

import math

import torch
from torch import nn


def recurrent_state_scan_with_initial_state(
    g_state: torch.Tensor,
    update: torch.Tensor,
    initial_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sequential recurrence with explicit state carry for chunked execution."""
    if g_state.ndim != 3 or update.ndim != 3:
        raise ValueError("Expected [batch, seq, dim] recurrence tensors.")
    if g_state.shape != update.shape:
        raise ValueError("Gate and update tensors must share the same shape.")
    if initial_state.shape != g_state.shape[:1] + g_state.shape[2:]:
        raise ValueError("initial_state must have shape [batch, dim].")

    outputs = []
    state = initial_state
    for t in range(g_state.size(1)):
        state = g_state[:, t] * state + update[:, t]
        outputs.append(state)
    return torch.stack(outputs, dim=1), state


def recurrent_state_scan(g_state: torch.Tensor, update: torch.Tensor) -> torch.Tensor:
    """Vectorized associative scan for state_t = a_t * state_{t-1} + b_t."""
    a = g_state.clone()
    b = update.clone()
    seq = a.size(1)
    offset = 1
    while offset < seq:
        curr_a = a[:, offset:].clone()
        prev_a = a[:, :-offset].clone()
        prev_b = b[:, :-offset].clone()
        a[:, offset:] = curr_a * prev_a
        b[:, offset:] = b[:, offset:] + curr_a * prev_b
        offset *= 2
    return b


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * norm * self.weight


class RecurrentMixerBlock(nn.Module):
    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.in_proj = nn.Linear(d_model, 4 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        u, g_in, g_state, candidate = self.in_proj(x).chunk(4, dim=-1)
        g_in = torch.sigmoid(g_in)
        g_state = torch.sigmoid(g_state)
        candidate = torch.tanh(candidate)
        state = recurrent_state_scan(g_state, g_in * candidate)
        mixed = state + u
        mixed = self.out_proj(mixed)
        return residual + self.dropout(mixed)


class GDN2ReferenceMixerBlock(nn.Module):
    """Local torch mixer with separated decay / erase / write gates.

    This is still a dense PyTorch fallback, but it more closely matches the
    revised HZ-0A architectural target than the earlier single-update-gate
    mixer. It is intended as a Mac-native reference path before a dedicated
    MLX/Metal backend exists.
    """

    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.norm = RMSNorm(d_model)
        self.in_proj = nn.Linear(d_model, 5 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.forward_with_state(x)
        return y

    def forward_with_state(
        self,
        x: torch.Tensor,
        initial_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        residual = x
        x = self.norm(x)
        u, decay_logits, erase_logits, write_logits, candidate = self.in_proj(x).chunk(5, dim=-1)
        decay = torch.sigmoid(decay_logits)
        erase = torch.sigmoid(erase_logits)
        write = torch.sigmoid(write_logits)
        candidate = torch.tanh(candidate)
        gate = decay * (1.0 - erase)
        update = write * candidate
        if initial_state is None:
            state = recurrent_state_scan(gate, update)
            final_state = state[:, -1]
        else:
            state, final_state = recurrent_state_scan_with_initial_state(gate, update, initial_state)
        mixed = self.out_proj(state + u)
        return residual + self.dropout(mixed), final_state


class AnchorAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        batch, seq, dim = x.shape
        head_dim = dim // self.n_heads
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        def reshape(t: torch.Tensor) -> torch.Tensor:
            return t.view(batch, seq, self.n_heads, head_dim).transpose(1, 2)

        q = reshape(q)
        k = reshape(k)
        v = reshape(v)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
        mask = torch.triu(torch.ones(seq, seq, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch, seq, dim)
        out = self.out_proj(out)
        return residual + self.dropout(out)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.up = nn.Linear(d_model, d_ff)
        self.gate = nn.Linear(d_model, d_ff)
        self.down = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = torch.nn.functional.silu(self.gate(x)) * self.up(x)
        x = self.down(x)
        return residual + self.dropout(x)
