"""A frozen, randomly-initialized (never trained) small causal
transformer, for B11 torch/CUDA experiments that need a "backbone" but
can't use the real HZ-0A checkpoint (MLX/Metal-only, does not transfer
to the Windows/CUDA machine -- `docs/rtx3060_windows_setup.md`).

**This is explicitly NOT a substitute for HZ-0A.** It is a much smaller,
untrained stand-in whose only job is to give a memory/adapter mechanism
something nonlinear and context-dependent to read from (its own
attention already lets later positions depend on earlier tokens, exactly
like a real transformer's hidden states would, just without any learned
language knowledge). Any result produced against this backbone answers
"does the mechanism work at all, robustly, at scale" -- a real,
useful, but DIFFERENT question from B11's main real-checkpoint result
(`docs/restart/hz0b_b11_evaluation_results.md`), which answers "does it
work against the actual frozen HZ-0A." Both are named explicitly wherever
this module's output is reported, never conflated.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SyntheticFrozenBackbone(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_layers: int, num_heads: int, seed: int = 0):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.embed = nn.Embedding(vocab_size, d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=d_model * 4, batch_first=True, dropout=0.0)
        self.layers = nn.TransformerEncoder(layer, num_layers=num_layers)
        with torch.no_grad():
            for p in self.parameters():
                p.copy_(torch.empty_like(p).normal_(generator=generator, std=0.02))
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """token_ids: [batch, seq] -> hidden: [batch, seq, d_model]."""
        seq_len = token_ids.shape[1]
        causal_mask = torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=token_ids.device), diagonal=1)
        hidden = self.embed(token_ids)
        return self.layers(hidden, mask=causal_mask, is_causal=True)
