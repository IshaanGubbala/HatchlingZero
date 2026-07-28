"""Clean MLX HZ-0A model matching the locked A1 topology."""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from reference.hz0a_mlx_metal import native_gdn2_forward_differentiable


class GDN2(nn.Module):
    def __init__(self, dim: int, heads: int, native_metal: bool = False):
        super().__init__()
        self.dim, self.heads, self.head_dim = dim, heads, dim // heads
        self.native_metal = native_metal
        self.qkv = nn.Linear(dim, 3 * dim)
        self.gates = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        self.gates.bias = mx.concatenate([mx.full((dim,), 4.59512), mx.full((2 * dim,), -4.59512)])

    def __call__(self, x, state=None):
        bsz, steps, _ = x.shape
        q, k, v = mx.split(self.qkv(x).reshape(bsz, steps, 3, self.heads, self.head_dim), 3, axis=2)
        q, k, v = (mx.squeeze(item, axis=2) for item in (q, k, v))
        d, e, w = mx.split(self.gates(x).reshape(bsz, steps, 3, self.heads, self.head_dim), 3, axis=2)
        d, e, w = (mx.squeeze(item, axis=2) for item in (d, e, w))
        if state is None:
            state = mx.zeros((bsz, self.heads, self.head_dim, self.head_dim), dtype=x.dtype)
        if self.native_metal:
            output, state = native_gdn2_forward_differentiable(q, k, v, d, e, w, state)
            return self.out(output.reshape(bsz, steps, self.dim)), state
        d, e, w = (mx.sigmoid(item) for item in (d, e, w))
        outputs = []
        for t in range(steps):
            state = d[:, t, :, :, None] * (1 - e[:, t, :, :, None]) * state + w[:, t, :, :, None] * v[:, t, :, :, None] * k[:, t, :, None, :]
            outputs.append(mx.sum(state * q[:, t, :, None, :], axis=-1))
        return self.out(mx.stack(outputs, axis=1).reshape(bsz, steps, self.dim)), state


class CausalAttention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.dim, self.heads, self.head_dim = dim, heads, dim // heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)

    def __call__(self, x, cache=None):
        bsz, steps, _ = x.shape
        q, k, v = mx.split(self.qkv(x).reshape(bsz, steps, 3, self.heads, self.head_dim), 3, axis=2)
        q, k, v = (mx.squeeze(item, axis=2).transpose(0, 2, 1, 3) for item in (q, k, v))
        if cache is not None:
            k = mx.concatenate([cache[0], k], axis=2)
            v = mx.concatenate([cache[1], v], axis=2)
        scores = mx.matmul(q, k.transpose(0, 1, 3, 2)) / mx.sqrt(mx.array(self.head_dim, dtype=mx.float32))
        total_steps = k.shape[2]
        past_steps = total_steps - steps
        mask = mx.triu(mx.full((steps, total_steps), -1e9), past_steps + 1)
        weights = mx.softmax(scores + mask[None, None], axis=-1)
        out = mx.matmul(weights, v).transpose(0, 2, 1, 3).reshape(bsz, steps, self.dim)
        return self.out(out), (k, v)


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, d_ff: int, attention: bool, native_metal: bool = False):
        super().__init__()
        self.attention = attention
        self.norm1, self.norm2 = nn.RMSNorm(dim), nn.RMSNorm(dim)
        self.mixer = CausalAttention(dim, heads) if attention else GDN2(dim, heads, native_metal)
        self.gate, self.up, self.down = nn.Linear(dim, d_ff), nn.Linear(dim, d_ff), nn.Linear(d_ff, dim)

    def __call__(self, x, state=None):
        if self.attention:
            mixed, next_state = self.mixer(self.norm1(x), state)
        else:
            mixed, next_state = self.mixer(self.norm1(x), state)
        x = x + mixed
        mlp = self.down(nn.silu(self.gate(self.norm2(x))) * self.up(self.norm2(x)))
        return x + mlp, next_state


class HZ0AMlxModel(nn.Module):
    def __init__(self, vocab_size: int, dim: int, layers: int, heads: int, d_ff: int, attention_indices: tuple[int, ...], native_metal: bool = False):
        super().__init__()
        self.vocab_size, self.dim, self.heads = vocab_size, dim, heads
        self.embedding = nn.Embedding(vocab_size, dim)
        self.blocks = [Block(dim, heads, d_ff, index in attention_indices, native_metal) for index in range(layers)]
        self.final_norm = nn.RMSNorm(dim)

    def __call__(self, token_ids, states=None):
        x = self.embedding(token_ids)
        if states is None:
            states = [None] * len(self.blocks)
        next_states = []
        for block, state in zip(self.blocks, states):
            x, state = block(x, state)
            next_states.append(state)
        return mx.matmul(self.final_norm(x), self.embedding.weight.T), next_states


def from_spec(path: str | Path) -> HZ0AMlxModel:
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    return HZ0AMlxModel(spec["vocab_size"], spec["d_model"], spec["num_layers"], spec["num_heads"], spec["d_ff"], tuple(spec["attention_layer_indices"]))
