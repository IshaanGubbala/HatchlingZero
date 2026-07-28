"""Tiny complete native HZ-0A graph assembled from manual-backward modules."""

from __future__ import annotations

import numpy as np

from restart.hz0a_pmetal.python.native_attention import NativeCausalAttention
from restart.hz0a_pmetal.python.native_blocks import NativeRMSNorm, NativeSwiGLU
from restart.hz0a_pmetal.python.native_gdn_block import NativeGDN2Block
from restart.hz0a_pmetal.python.native_layers import NativeEmbedding, NativeTiedLMHead, cross_entropy_backward, cross_entropy_forward


class NativeBlock:
    def __init__(self, name, dim, heads, d_k, d_v, d_ff, attention, rng):
        self.attention = attention
        self.norm1 = NativeRMSNorm(name + ".norm1", dim)
        self.mixer = NativeCausalAttention(name + ".attention", dim, heads, rng) if attention else NativeGDN2Block(name + ".gdn2", dim, heads, d_k, d_v, rng)
        self.norm2 = NativeRMSNorm(name + ".norm2", dim)
        self.mlp = NativeSwiGLU(name + ".mlp", dim, d_ff, rng)
        self._cache = None

    def parameters(self):
        return self.norm1.parameters() + self.mixer.parameters() + self.norm2.parameters() + self.mlp.parameters()

    def forward(self, x, state=None):
        normalized = self.norm1.forward(x)
        if self.attention:
            mixed, next_state = self.mixer.forward(normalized), None
        else:
            mixed, next_state = self.mixer.forward(normalized, state)
        residual = x + mixed
        mlp_out = self.mlp.forward(self.norm2.forward(residual))
        output = residual + mlp_out
        self._cache = (x, residual, next_state)
        return output, next_state

    def backward(self, grad_output, grad_state=None):
        x, residual, _ = self._cache
        grad_residual = grad_output + self.norm2.backward(self.mlp.backward(grad_output))
        if self.attention:
            grad_normed = self.mixer.backward(grad_residual)
            return grad_residual + self.norm1.backward(grad_normed), None
        grad_normed, grad_initial = self.mixer.backward(grad_residual, grad_state)
        return grad_residual + self.norm1.backward(grad_normed), grad_initial


class NativeTinyHZ0AModel:
    def __init__(self, vocab_size, dim, layers, heads, d_k, d_v, d_ff, attention_indices, seed=0):
        rng = np.random.default_rng(seed)
        self.embedding = NativeEmbedding("embedding", vocab_size, dim, rng)
        self.blocks = [NativeBlock(f"blocks.{index}", dim, heads, d_k, d_v, d_ff, index in set(attention_indices), rng) for index in range(layers)]
        self.final_norm = NativeRMSNorm("final_norm", dim)
        self.lm_head = NativeTiedLMHead(self.embedding)
        self._states = None

    def parameters(self):
        return self.embedding.parameters() + [parameter for block in self.blocks for parameter in block.parameters()] + self.final_norm.parameters()

    def init_states(self, batch_size, heads, d_v, d_k):
        return [None if block.attention else np.zeros((batch_size, heads, d_v, d_k), dtype=np.float32) for block in self.blocks]

    def forward(self, token_ids, states=None):
        x = self.embedding.forward(token_ids)
        if states is None:
            states = [None] * len(self.blocks)
        next_states = []
        for block, state in zip(self.blocks, states):
            x, state = block.forward(x, state)
            next_states.append(state)
        hidden = self.final_norm.forward(x)
        return self.lm_head.forward(hidden), next_states

    def loss_and_backward(self, token_ids, targets, states=None):
        logits, next_states = self.forward(token_ids, states)
        loss, cache = cross_entropy_forward(logits, targets)
        grad_hidden = self.final_norm.backward(self.lm_head.backward(cross_entropy_backward(cache)))
        grad_state = [None] * len(self.blocks)
        for index in reversed(range(len(self.blocks))):
            grad_hidden, grad_state[index] = self.blocks[index].backward(grad_hidden, None)
        self.embedding.backward(grad_hidden)
        return loss, next_states
