from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    clipped = np.clip(x, -30.0, 30.0)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        return 1.0 / (1.0 + np.exp(-clipped))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    work = x.astype(np.float64, copy=False)
    shifted = work - np.max(work, axis=axis, keepdims=True)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def init_state(batch_size: int, num_heads: int, d_v: int, d_k: int, dtype=np.float32) -> np.ndarray:
    return np.zeros((batch_size, num_heads, d_v, d_k), dtype=dtype)


def gdn2_step(
    q_t: np.ndarray,
    k_t: np.ndarray,
    v_t: np.ndarray,
    decay_logits_t: np.ndarray,
    erase_logits_t: np.ndarray,
    write_logits_t: np.ndarray,
    prev_state: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    decay = sigmoid(decay_logits_t).astype(prev_state.dtype)
    erase = sigmoid(erase_logits_t).astype(prev_state.dtype)
    write = sigmoid(write_logits_t).astype(prev_state.dtype)

    decay_b = decay[:, :, None, :]
    erase_b = erase[:, :, None, :]
    write_b = write[:, :, :, None]
    outer = v_t[:, :, :, None] * k_t[:, :, None, :]
    update = write_b * outer
    next_state = decay_b * ((1.0 - erase_b) * prev_state) + update
    readout = np.einsum("bhvk,bhk->bhv", next_state, q_t)
    return readout.astype(prev_state.dtype), next_state.astype(prev_state.dtype)


def gdn2_scan(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    decay_logits: np.ndarray,
    erase_logits: np.ndarray,
    write_logits: np.ndarray,
    initial_state: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    bsz, steps, num_heads, d_v = v.shape
    d_k = q.shape[-1]
    state = initial_state.copy() if initial_state is not None else init_state(bsz, num_heads, d_v, d_k, dtype=v.dtype)
    outputs = []
    for t in range(steps):
        out_t, state = gdn2_step(
            q[:, t],
            k[:, t],
            v[:, t],
            decay_logits[:, t],
            erase_logits[:, t],
            write_logits[:, t],
            state,
        )
        outputs.append(out_t)
    return np.stack(outputs, axis=1), state


def gdn2_chunk_scan(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    decay_logits: np.ndarray,
    erase_logits: np.ndarray,
    write_logits: np.ndarray,
    chunk_size: int,
    initial_state: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    state = initial_state
    chunks = []
    for start in range(0, q.shape[1], chunk_size):
        end = min(start + chunk_size, q.shape[1])
        out, state = gdn2_scan(
            q[:, start:end],
            k[:, start:end],
            v[:, start:end],
            decay_logits[:, start:end],
            erase_logits[:, start:end],
            write_logits[:, start:end],
            initial_state=state,
        )
        chunks.append(out)
    return np.concatenate(chunks, axis=1), state


@dataclass
class ReferenceLinear:
    weight: np.ndarray
    bias: np.ndarray

    @classmethod
    def init(cls, rng: np.random.Generator, in_dim: int, out_dim: int, std: float = 0.02) -> "ReferenceLinear":
        weight = rng.normal(0.0, std, size=(in_dim, out_dim)).astype(np.float32)
        bias = np.zeros((out_dim,), dtype=np.float32)
        return cls(weight=weight, bias=bias)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if not np.isfinite(x).all():
            raise FloatingPointError("ReferenceLinear received non-finite input")
        if not np.isfinite(self.weight).all() or not np.isfinite(self.bias).all():
            raise FloatingPointError("ReferenceLinear parameters must remain finite")
        out = np.einsum("...i,ij->...j", x.astype(np.float64), self.weight.astype(np.float64))
        out = out + self.bias.astype(np.float64)
        if not np.isfinite(out).all():
            raise FloatingPointError("ReferenceLinear produced non-finite output")
        return out.astype(np.float32)


@dataclass
class RMSNorm:
    scale: np.ndarray
    eps: float = 1e-6

    @classmethod
    def init(cls, dim: int) -> "RMSNorm":
        return cls(scale=np.ones((dim,), dtype=np.float32))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        # Accumulate the norm in float64 so large recurrent residuals do not
        # overflow before normalization brings them back to a usable scale.
        x64 = x.astype(np.float64, copy=False)
        rms = np.sqrt(np.mean(np.square(x64), axis=-1, keepdims=True) + self.eps)
        return ((x64 / rms) * self.scale.astype(np.float64)).astype(np.float32)


@dataclass
class SwiGLU:
    gate_proj: ReferenceLinear
    up_proj: ReferenceLinear
    down_proj: ReferenceLinear

    @classmethod
    def init(cls, rng: np.random.Generator, d_model: int, d_ff: int) -> "SwiGLU":
        return cls(
            gate_proj=ReferenceLinear.init(rng, d_model, d_ff),
            up_proj=ReferenceLinear.init(rng, d_model, d_ff),
            down_proj=ReferenceLinear.init(rng, d_ff, d_model),
        )

    def __call__(self, x: np.ndarray) -> np.ndarray:
        gate = self.gate_proj(x)
        gate = gate * sigmoid(gate)
        up = self.up_proj(x)
        return self.down_proj(gate * up)


@dataclass
class GDN2ReferenceMixer:
    num_heads: int
    d_model: int
    d_k: int
    d_v: int
    in_proj: ReferenceLinear
    out_proj: ReferenceLinear

    @classmethod
    def init(cls, rng: np.random.Generator, d_model: int, num_heads: int, d_k: int, d_v: int) -> "GDN2ReferenceMixer":
        out_dim = num_heads * (4 * d_k + 2 * d_v)
        mixer = cls(
            num_heads=num_heads,
            d_model=d_model,
            d_k=d_k,
            d_v=d_v,
            in_proj=ReferenceLinear.init(rng, d_model, out_dim),
            out_proj=ReferenceLinear.init(rng, num_heads * d_v, d_model),
        )
        gate_bias = 4.59512
        decay_start = num_heads * (2 * d_k + d_v)
        erase_start = decay_start + num_heads * d_k
        write_start = erase_start + num_heads * d_k
        mixer.in_proj.bias[decay_start:erase_start] = gate_bias
        mixer.in_proj.bias[erase_start:write_start] = -gate_bias
        mixer.in_proj.bias[write_start:] = -gate_bias
        return mixer

    def _split(self, x: np.ndarray) -> tuple[np.ndarray, ...]:
        bsz, steps, _ = x.shape
        proj = self.in_proj(x).reshape(bsz, steps, self.num_heads, 4 * self.d_k + 2 * self.d_v)
        cursor = 0
        q = proj[..., cursor:cursor + self.d_k]
        cursor += self.d_k
        k = proj[..., cursor:cursor + self.d_k]
        cursor += self.d_k
        v = proj[..., cursor:cursor + self.d_v]
        cursor += self.d_v
        decay = proj[..., cursor:cursor + self.d_k]
        cursor += self.d_k
        erase = proj[..., cursor:cursor + self.d_k]
        cursor += self.d_k
        write = proj[..., cursor:cursor + self.d_v]
        return q, k, v, decay, erase, write

    def __call__(self, x: np.ndarray, state: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        q, k, v, decay, erase, write = self._split(x)
        readout, next_state = gdn2_scan(q, k, v, decay, erase, write, initial_state=state)
        readout = readout.reshape(x.shape[0], x.shape[1], self.num_heads * self.d_v)
        return self.out_proj(readout), next_state


@dataclass
class CausalSelfAttention:
    num_heads: int
    d_model: int
    q_proj: ReferenceLinear
    k_proj: ReferenceLinear
    v_proj: ReferenceLinear
    out_proj: ReferenceLinear

    @classmethod
    def init(cls, rng: np.random.Generator, d_model: int, num_heads: int) -> "CausalSelfAttention":
        return cls(
            num_heads=num_heads,
            d_model=d_model,
            q_proj=ReferenceLinear.init(rng, d_model, d_model),
            k_proj=ReferenceLinear.init(rng, d_model, d_model),
            v_proj=ReferenceLinear.init(rng, d_model, d_model),
            out_proj=ReferenceLinear.init(rng, d_model, d_model),
        )

    def __call__(self, x: np.ndarray) -> np.ndarray:
        bsz, steps, dim = x.shape
        head_dim = dim // self.num_heads
        # Keep attention intermediates in float64: recurrent residuals can be
        # large during tiny-reference stress runs even when outputs remain finite.
        q = self.q_proj(x).astype(np.float64).reshape(bsz, steps, self.num_heads, head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).astype(np.float64).reshape(bsz, steps, self.num_heads, head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).astype(np.float64).reshape(bsz, steps, self.num_heads, head_dim).transpose(0, 2, 1, 3)
        q = q / np.sqrt(np.mean(np.square(q), axis=-1, keepdims=True) + 1e-6)
        k = k / np.sqrt(np.mean(np.square(k), axis=-1, keepdims=True) + 1e-6)
        with np.errstate(over="ignore", invalid="ignore"):
            scores = np.matmul(q, np.swapaxes(k, -1, -2)) / np.sqrt(head_dim)
        mask = np.triu(np.ones((steps, steps), dtype=bool), k=1)
        scores = np.where(mask[None, None], -1e9, scores)
        weights = softmax(scores, axis=-1)
        out = np.matmul(weights, v).transpose(0, 2, 1, 3).reshape(bsz, steps, dim)
        return self.out_proj(out.astype(np.float32))


@dataclass
class HZ0ABlock:
    norm1: RMSNorm
    norm2: RMSNorm
    mixer: GDN2ReferenceMixer | CausalSelfAttention
    mlp: SwiGLU
    is_attention: bool

    @classmethod
    def init(
        cls,
        rng: np.random.Generator,
        d_model: int,
        num_heads: int,
        d_k: int,
        d_v: int,
        d_ff: int,
        is_attention: bool,
    ) -> "HZ0ABlock":
        mixer: GDN2ReferenceMixer | CausalSelfAttention
        if is_attention:
            mixer = CausalSelfAttention.init(rng, d_model, num_heads)
        else:
            mixer = GDN2ReferenceMixer.init(rng, d_model, num_heads, d_k, d_v)
        return cls(
            norm1=RMSNorm.init(d_model),
            norm2=RMSNorm.init(d_model),
            mixer=mixer,
            mlp=SwiGLU.init(rng, d_model, d_ff),
            is_attention=is_attention,
        )

    def __call__(self, x: np.ndarray, state: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
        x_norm = self.norm1(x)
        if self.is_attention:
            mixed = self.mixer(x_norm)
            next_state = None
        else:
            mixed, next_state = self.mixer(x_norm, state)
        x = x + mixed
        x = x + self.mlp(self.norm2(x))
        return x, next_state


@dataclass
class TinyHZ0AModel:
    vocab_size: int
    d_model: int
    num_heads: int
    d_k: int
    d_v: int
    embedding: np.ndarray
    blocks: list[HZ0ABlock]
    final_norm: RMSNorm

    @classmethod
    def init(
        cls,
        rng_seed: int,
        vocab_size: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_k: int,
        d_v: int,
        d_ff: int,
        attention_layer_indices: Iterable[int],
    ) -> "TinyHZ0AModel":
        rng = np.random.default_rng(rng_seed)
        embedding = rng.normal(0.0, 0.02, size=(vocab_size, d_model)).astype(np.float32)
        blocks = [
            HZ0ABlock.init(rng, d_model, num_heads, d_k, d_v, d_ff, is_attention=(idx in set(attention_layer_indices)))
            for idx in range(num_layers)
        ]
        return cls(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            d_k=d_k,
            d_v=d_v,
            embedding=embedding,
            blocks=blocks,
            final_norm=RMSNorm.init(d_model),
        )

    def init_states(self, batch_size: int) -> list[np.ndarray | None]:
        return [
            init_state(batch_size, self.num_heads, self.d_v, self.d_k)
            if not block.is_attention
            else None
            for block in self.blocks
        ]

    def __call__(self, token_ids: np.ndarray, states: list[np.ndarray | None] | None = None) -> tuple[np.ndarray, list[np.ndarray | None]]:
        x = self.embedding[token_ids]
        states = self.init_states(token_ids.shape[0]) if states is None else [s.copy() if s is not None else None for s in states]
        next_states: list[np.ndarray | None] = []
        for block, state in zip(self.blocks, states):
            x, next_state = block(x, state)
            next_states.append(next_state)
        x = self.final_norm(x)
        logits = np.einsum("btd,vd->btv", x, self.embedding)
        return logits, next_states


def cross_entropy_loss(logits: np.ndarray, targets: np.ndarray) -> float:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    log_probs = shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))
    gathered = np.take_along_axis(log_probs, targets[..., None], axis=-1)[..., 0]
    return float(-np.mean(gathered))
