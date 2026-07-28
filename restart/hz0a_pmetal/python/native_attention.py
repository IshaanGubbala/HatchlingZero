"""Unfused causal attention with explicit NumPy backward."""

from __future__ import annotations

import numpy as np

from restart.hz0a_pmetal.python.native_layers import NativeLinear


class NativeCausalAttention:
    def __init__(self, name: str, dim: int, heads: int, rng: np.random.Generator):
        if dim % heads:
            raise ValueError("attention dimension must divide evenly by heads")
        self.dim, self.heads, self.head_dim = dim, heads, dim // heads
        self.qkv = NativeLinear(name + ".qkv", dim, 3 * dim, rng)
        self.out = NativeLinear(name + ".out", dim, dim, rng)
        self._cache = None

    def parameters(self):
        return self.qkv.parameters() + self.out.parameters()

    def forward(self, x: np.ndarray):
        bsz, steps, _ = x.shape
        packed = self.qkv.forward(x).reshape(bsz, steps, self.heads, 3 * self.head_dim)
        q, k, v = np.split(packed, 3, axis=-1)
        scale = self.head_dim ** -0.5
        scores = np.einsum("bthd,bshd->bhts", q, k) * scale
        scores = np.where(np.triu(np.ones((steps, steps), dtype=bool), 1)[None, None], -1e9, scores)
        weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
        weights /= weights.sum(axis=-1, keepdims=True)
        mixed = np.einsum("bhts,bshd->bthd", weights, v).reshape(bsz, steps, self.dim)
        output = self.out.forward(mixed)
        self._cache = (x, q, k, v, weights, scale)
        return output

    def backward(self, grad_output: np.ndarray):
        x, q, k, v, weights, scale = self._cache
        grad_mixed = self.out.backward(grad_output).reshape(q.shape)
        grad_weights = np.einsum("bthd,bshd->bhts", grad_mixed, v)
        grad_v = np.einsum("bhts,bthd->bshd", weights, grad_mixed)
        grad_scores = weights * (grad_weights - np.sum(grad_weights * weights, axis=-1, keepdims=True))
        grad_scores *= np.tril(np.ones(grad_scores.shape[-2:], dtype=np.float32))[None, None]
        grad_q = np.einsum("bhts,bshd->bthd", grad_scores, k) * scale
        grad_k = np.einsum("bhts,bthd->bshd", grad_scores, q) * scale
        grad_packed = np.concatenate([grad_q, grad_k, grad_v], axis=-1).reshape(x.shape[0], x.shape[1], -1)
        return self.qkv.backward(grad_packed)
