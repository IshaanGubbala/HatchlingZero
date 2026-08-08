"""Parameter-matched causal transformer for the HZ-0A comparison protocol."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from torch import nn


class MatchedTransformerConfig:
    def __init__(self, values: dict):
        self.__dict__.update(values)

    @classmethod
    def from_json(cls, path: str | Path) -> "MatchedTransformerConfig":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))


class BiasFreeRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + self.eps) * self.weight


class MatchedTransformerBlock(nn.Module):
    def __init__(self, config: MatchedTransformerConfig):
        super().__init__()
        d, ff = config.d_model, config.d_ff
        self.norm1, self.norm2 = BiasFreeRMSNorm(d), BiasFreeRMSNorm(d)
        self.qkv, self.attn_out = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.gate, self.up, self.down = nn.Linear(d, ff), nn.Linear(d, ff), nn.Linear(ff, d)
        self.heads, self.head_dim = config.num_heads, config.head_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, steps, dim = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(bsz, steps, self.heads, 3 * self.head_dim).chunk(3, dim=-1)
        q, k, v = (item.transpose(1, 2) for item in (q, k, v))
        mixed = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.attn_out(mixed.transpose(1, 2).reshape(bsz, steps, dim))
        y = self.norm2(x)
        return x + self.down(torch.nn.functional.silu(self.gate(y)) * self.up(y))


class MatchedTransformerLM(nn.Module):
    def __init__(self, config: MatchedTransformerConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        # Same fix as reference/hz0a_torch_model.py's HZ0AModel (found 2026-08-01,
        # see that file's comment for the full derivation): PyTorch's nn.Embedding
        # default init is N(0, 1), ~28x larger than MLX's own `sqrt(1/dims)`
        # default -- left uncorrected here, it produces the same ~500+ initial
        # loss / 300-1000+ early gradient norms as the HZ0AModel bug did, not a
        # depth-scaling quirk of this architecture as originally assumed.
        nn.init.normal_(self.embedding.weight, std=math.sqrt(1.0 / config.d_model))
        self.blocks = nn.ModuleList(MatchedTransformerBlock(config) for _ in range(config.num_layers))
        self.final_norm = BiasFreeRMSNorm(config.d_model)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(token_ids)
        for block in self.blocks:
            x = block(x)
        return torch.einsum("btd,vd->btv", self.final_norm(x), self.embedding.weight)


def parameter_count(config: MatchedTransformerConfig) -> int:
    return sum(parameter.numel() for parameter in MatchedTransformerLM(config).parameters())
