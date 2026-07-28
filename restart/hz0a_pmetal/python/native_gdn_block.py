"""Trainable native GDN-2 recurrent block using the explicit PMetal operator."""

from __future__ import annotations

import numpy as np

from restart.hz0a_pmetal.python.native_layers import NativeLinear, NativeParameter
from restart.hz0a_pmetal.python.pmetal_reference import Gdn2ForwardInputs, gdn2_backward, gdn2_forward


class NativeGDN2Block:
    def __init__(self, name: str, dim: int, heads: int, d_k: int, d_v: int, rng: np.random.Generator):
        self.dim, self.heads, self.d_k, self.d_v = dim, heads, d_k, d_v
        width = heads * (4 * d_k + 2 * d_v)
        self.in_proj = NativeLinear(name + ".in_proj", dim, width, rng)
        self.out_proj = NativeLinear(name + ".out_proj", heads * d_v, dim, rng)
        start = heads * (2 * d_k + d_v)
        self.in_proj.bias.data[start:start + heads * d_k] = 4.59512
        self.in_proj.bias.data[start + heads * d_k:start + 2 * heads * d_k] = -4.59512
        self.in_proj.bias.data[start + 2 * heads * d_k:] = -4.59512
        self._cache = None

    def parameters(self):
        return self.in_proj.parameters() + self.out_proj.parameters()

    def forward(self, x: np.ndarray, initial_state: np.ndarray | None = None):
        bsz, steps, _ = x.shape
        projected = self.in_proj.forward(x).reshape(bsz, steps, self.heads, 4 * self.d_k + 2 * self.d_v)
        q = projected[..., :self.d_k]
        k = projected[..., self.d_k:2 * self.d_k]
        v = projected[..., 2 * self.d_k:2 * self.d_k + self.d_v]
        offset = 2 * self.d_k + self.d_v
        decay = projected[..., offset:offset + self.d_k]
        erase = projected[..., offset + self.d_k:offset + 2 * self.d_k]
        write = projected[..., offset + 2 * self.d_k:]
        state = initial_state if initial_state is not None else np.zeros((bsz, self.heads, self.d_v, self.d_k), dtype=np.float32)
        result = gdn2_forward(Gdn2ForwardInputs(q, k, v, decay, erase, write, state))
        output = self.out_proj.forward(result.outputs.reshape(bsz, steps, self.heads * self.d_v))
        self._cache = (x, projected, result)
        return output, result.final_state

    def backward(self, grad_output: np.ndarray, grad_final_state: np.ndarray | None = None):
        x, projected, result = self._cache
        grad_mixed = self.out_proj.backward(grad_output).reshape(projected.shape[0], projected.shape[1], self.heads, self.d_v)
        if grad_final_state is None:
            grad_final_state = np.zeros_like(result.final_state)
        gradients = gdn2_backward(grad_mixed, grad_final_state, result.backward_cache).gradients
        grad_projected = np.concatenate([gradients["q"], gradients["k"], gradients["v"], gradients["decay_logits"], gradients["erase_logits"], gradients["write_logits"]], axis=-1)
        grad_input = self.in_proj.backward(grad_projected.reshape(x.shape[0], x.shape[1], -1))
        return grad_input, gradients["initial_state"]
