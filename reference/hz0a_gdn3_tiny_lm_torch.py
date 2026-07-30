"""Torch/CUDA port of `reference/hz0a_gdn3_tiny_lm.py` -- same tiny,
standalone, all-GDN-family (no attention) model, either GDN2 (current,
reusing `reference/hz0a_torch_model.py`'s own `GDN2Mixer`, unmodified) or
the GDN-3 candidate (`reference/hz0a_gdn3_candidate_mixer_torch.py`).
Device-agnostic.
"""
from __future__ import annotations

import torch
from torch import nn

from reference.hz0a_gdn3_candidate_mixer_torch import GDN3CandidateMixerTorch
from reference.hz0a_torch_model import GDN2Mixer, HZ0AConfig, RMSNorm, SwiGLU


class TinyBlockTorch(nn.Module):
    def __init__(self, dim: int, heads: int, d_ff: int, use_candidate: bool):
        super().__init__()
        self.norm1, self.norm2 = RMSNorm(dim), RMSNorm(dim)
        self.heads, self.head_dim = heads, dim // heads
        if use_candidate:
            self.mixer = GDN3CandidateMixerTorch(dim, heads)
        else:
            config = HZ0AConfig(vocab_size=1, d_model=dim, num_layers=1, num_heads=heads, d_k=self.head_dim, d_v=self.head_dim, d_ff=d_ff, attention_layer_indices=())
            self.mixer = GDN2Mixer(config)
        self.mlp = SwiGLU(dim, d_ff)

    def forward(self, x, state):
        if state is None:
            # GDN2Mixer (unlike GDN3CandidateMixerTorch) requires a
            # pre-initialized state -- it no longer defaults None to zeros
            # internally (see reference/hz0a_torch_model.py's HZ0AModel.
            # init_states, which this mirrors). Initializing here rather
            # than relying on either mixer's own internal handling keeps
            # both arms of the comparison on the same explicit path.
            state = torch.zeros(x.shape[0], self.heads, self.head_dim, self.head_dim, device=x.device, dtype=x.dtype)
        mixed, next_state = self.mixer(self.norm1(x), state)
        x = x + mixed
        return x + self.mlp(self.norm2(x)), next_state


class TinyGDNLMTorch(nn.Module):
    def __init__(self, vocab_size: int, dim: int, layers: int, heads: int, d_ff: int, use_candidate: bool):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList(TinyBlockTorch(dim, heads, d_ff, use_candidate) for _ in range(layers))
        self.final_norm = RMSNorm(dim)

    def forward(self, token_ids, states=None):
        x = self.embedding(token_ids)
        states = [None] * len(self.blocks) if states is None else states
        next_states = []
        for block, state in zip(self.blocks, states):
            x, state = block(x, state)
            next_states.append(state)
        return torch.einsum("btd,vd->btv", self.final_norm(x), self.embedding.weight), next_states
