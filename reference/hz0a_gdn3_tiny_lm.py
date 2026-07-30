"""Tiny, standalone language model for comparing GDN2 (HZ-0A's current
recurrence) against the GDN-3 candidate on REAL language-modeling loss --
the one question `docs/restart/hz0a_gdn3_overwrite_benchmark_results.md`
explicitly left open (that benchmark tested the mechanism in isolation on
a synthetic write/overwrite task, not on real token sequences).

All GDN-family layers, no attention -- isolates the recurrence's own
effect on real LM loss rather than mixing in an identical (unaffected)
attention component in both arms. `reference/hz0a_mlx_model.py` is
imported for its `GDN2` class (unmodified, used as-is) and is otherwise
untouched -- this is a new, separate model, not an edit to HZ-0A's
locked one.
"""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from reference.hz0a_gdn3_candidate_mixer import GDN3CandidateMixer
from reference.hz0a_mlx_model import GDN2


class TinyBlock(nn.Module):
    def __init__(self, dim: int, heads: int, d_ff: int, use_candidate: bool):
        super().__init__()
        self.norm1, self.norm2 = nn.RMSNorm(dim), nn.RMSNorm(dim)
        self.mixer = GDN3CandidateMixer(dim, heads) if use_candidate else GDN2(dim, heads, native_metal=False)
        self.gate, self.up, self.down = nn.Linear(dim, d_ff), nn.Linear(dim, d_ff), nn.Linear(d_ff, dim)

    def __call__(self, x, state=None):
        mixed, next_state = self.mixer(self.norm1(x), state)
        x = x + mixed
        normed2 = self.norm2(x)
        return x + self.down(nn.silu(self.gate(normed2)) * self.up(normed2)), next_state


class TinyGDNLM(nn.Module):
    def __init__(self, vocab_size: int, dim: int, layers: int, heads: int, d_ff: int, use_candidate: bool):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.blocks = [TinyBlock(dim, heads, d_ff, use_candidate) for _ in range(layers)]
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
