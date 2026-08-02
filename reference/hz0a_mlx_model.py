"""Clean MLX HZ-0A model matching the locked A1 topology."""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.utils import checkpoint

from reference.hz0a_mlx_metal import native_gdn2_fix_forward_differentiable, native_gdn2_forward_differentiable


class GDN2(nn.Module):
    def __init__(self, dim: int, heads: int, native_metal: bool = False):
        super().__init__()
        self.dim, self.heads, self.head_dim = dim, heads, dim // heads
        self.native_metal = native_metal
        # Single combined projection instead of separate qkv/gates Linears:
        # mathematically identical (a Linear applied to concatenated weight
        # rows equals two separate Linears on the same input, just computed
        # as one matmul/dispatch instead of two) -- cuts one full
        # (dim, 6*dim) matmul's dispatch overhead per GDN2 block per call.
        # Matches the layout hz0a-pmetal-tensor's Rust Gdn2Block already
        # used (in_proj: dim -> heads*(4*d_k+2*d_v)); the Python/MLX path
        # had drifted to the less efficient two-projection form.
        self.in_proj = nn.Linear(dim, 6 * dim)
        self.out = nn.Linear(dim, dim)
        self.in_proj.bias = mx.concatenate([mx.zeros((3 * dim,)), mx.full((dim,), 4.59512), mx.full((2 * dim,), -4.59512)])

    def __call__(self, x, state=None):
        bsz, steps, _ = x.shape
        q, k, v, d, e, w = mx.split(self.in_proj(x).reshape(bsz, steps, 6, self.heads, self.head_dim), 6, axis=2)
        q, k, v, d, e, w = (mx.squeeze(item, axis=2) for item in (q, k, v, d, e, w))
        if state is None:
            state = mx.zeros((bsz, self.heads, self.head_dim, self.head_dim), dtype=x.dtype)
        if self.native_metal:
            output, state = native_gdn2_forward_differentiable(q, k, v, d, e, w, state)
            return self.out(output.reshape(bsz, steps, self.dim)), state
        d, e, w = (mx.sigmoid(item) for item in (d, e, w))
        outputs = []
        for t in range(steps):
            state = d[:, t, :, None, :] * (1 - e[:, t, :, None, :]) * state + w[:, t, :, :, None] * v[:, t, :, :, None] * k[:, t, :, None, :]
            outputs.append(mx.sum(state * q[:, t, :, None, :], axis=-1))
        return self.out(mx.stack(outputs, axis=1).reshape(bsz, steps, self.dim)), state


class GDN2Fix(nn.Module):
    """Opt-in exact vector-gated GDN-2 reference path.

    This deliberately remains separate from ``GDN2`` until the Metal kernel
    and its VJP have matched this implementation on every tensor.
    """

    def __init__(self, dim: int, heads: int, native_metal: bool = False):
        super().__init__()
        self.dim, self.heads, self.head_dim = dim, heads, dim // heads
        self.native_metal = native_metal
        self.in_proj = nn.Linear(dim, 6 * dim)
        self.out = nn.Linear(dim, dim)
        self.decay_a = mx.full((1,), -6.13)
        self.in_proj.bias = mx.concatenate([
            mx.zeros((3 * dim,)),
            mx.full((dim,), 4.59512),
            mx.full((2 * dim,), -4.59512),
        ])

    def __call__(self, x, state=None):
        bsz, steps, _ = x.shape
        projected = self.in_proj(x).reshape(bsz, steps, 6, self.heads, self.head_dim)
        q, k, v, decay, erase, write = mx.split(projected, 6, axis=2)
        q, k, v, decay, erase, write = (mx.squeeze(item, axis=2) for item in (q, k, v, decay, erase, write))
        raw_q, raw_k, raw_decay, raw_erase, raw_write = q, k, decay, erase, write
        q = q / mx.maximum(mx.linalg.norm(q.astype(mx.float32), axis=-1, keepdims=True), 1e-6)
        k = k / mx.maximum(mx.linalg.norm(k.astype(mx.float32), axis=-1, keepdims=True), 1e-6)
        decay_rate = mx.exp(self.decay_a).reshape(1, 1, 1, 1)
        decay_fp32 = decay.astype(mx.float32)
        softplus_decay = mx.maximum(decay_fp32, 0) + mx.log1p(mx.exp(-mx.abs(decay_fp32)))
        alpha = mx.exp(-decay_rate * softplus_decay).astype(x.dtype)
        erase = mx.sigmoid(erase.astype(mx.float32)).astype(x.dtype)
        write = mx.sigmoid(write.astype(mx.float32)).astype(x.dtype)
        if state is None:
            state = mx.zeros((bsz, self.heads, self.head_dim, self.head_dim), dtype=x.dtype)
        if self.native_metal:
            mixed, state = native_gdn2_fix_forward_differentiable(raw_q, raw_k, v, raw_decay, raw_erase, raw_write, state)
            return self.out(mixed.reshape(bsz, steps, self.dim)), state
        outputs = []
        for t in range(steps):
            decayed = state * alpha[:, t, :, None, :]
            old_value = mx.sum(decayed * (erase[:, t] * k[:, t])[:, :, None, :], axis=-1)
            residual = write[:, t] * v[:, t] - old_value
            state = decayed + residual[:, :, :, None] * k[:, t, :, None, :]
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
    def __init__(self, dim: int, heads: int, d_ff: int, attention: bool, native_metal: bool = False, mixer: str = "gdn2"):
        super().__init__()
        self.attention = attention
        self.norm1, self.norm2 = nn.RMSNorm(dim), nn.RMSNorm(dim)
        if attention:
            self.mixer = CausalAttention(dim, heads)
        elif mixer == "gdn2_fix":
            self.mixer = GDN2Fix(dim, heads, native_metal)
        else:
            self.mixer = GDN2(dim, heads, native_metal)
        self.gate, self.up, self.down = nn.Linear(dim, d_ff), nn.Linear(dim, d_ff), nn.Linear(d_ff, dim)

    def __call__(self, x, state=None):
        mixed, next_state = self.mixer(self.norm1(x), state)
        x = x + mixed
        normed2 = self.norm2(x)
        mlp = self.down(nn.silu(self.gate(normed2)) * self.up(normed2))
        return x + mlp, next_state


class HZ0AMlxModel(nn.Module):
    def __init__(self, vocab_size: int, dim: int, layers: int, heads: int, d_ff: int, attention_indices: tuple[int, ...], native_metal: bool = False, checkpoint_blocks: bool = False, mixer: str = "gdn2"):
        super().__init__()
        self.vocab_size, self.dim, self.heads = vocab_size, dim, heads
        self.checkpoint_blocks = checkpoint_blocks
        self.embedding = nn.Embedding(vocab_size, dim)
        self.blocks = [Block(dim, heads, d_ff, index in attention_indices, native_metal, mixer) for index in range(layers)]
        self._checkpointed_blocks = [checkpoint(block) for block in self.blocks] if checkpoint_blocks else self.blocks
        self.final_norm = nn.RMSNorm(dim)

    def __call__(self, token_ids, states=None):
        x = self.embedding(token_ids)
        if states is None:
            states = [None] * len(self.blocks)
        next_states = []
        for block, state in zip(self._checkpointed_blocks, states):
            x, state = block(x, state)
            next_states.append(state)
        return mx.matmul(self.final_norm(x), self.embedding.weight.T), next_states


def from_spec(path: str | Path) -> HZ0AMlxModel:
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    return HZ0AMlxModel(spec["vocab_size"], spec["d_model"], spec["num_layers"], spec["num_heads"], spec["d_ff"], tuple(spec["attention_layer_indices"]))
