from __future__ import annotations

import math

import torch
from torch import nn

from .blocks import FeedForward, RMSNorm


class TransformerAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        batch, seq, dim = x.shape
        head_dim = dim // self.n_heads
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        def reshape(t: torch.Tensor) -> torch.Tensor:
            return t.view(batch, seq, self.n_heads, head_dim).transpose(1, 2)

        q = reshape(q)
        k = reshape(k)
        v = reshape(v)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
        mask = torch.triu(torch.ones(seq, seq, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch, seq, dim)
        out = self.out_proj(out)
        return residual + self.dropout(out)


class TransformerLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.attn = TransformerAttention(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attn(x)
        return self.ffn(x)


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        max_seq_len: int,
        **_: object,
    ) -> None:
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.layers = nn.ModuleList(
            [TransformerLayer(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout) for _ in range(n_layers)]
        )
        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        _, seq = tokens.shape
        positions = torch.arange(seq, device=tokens.device)
        x = self.token_emb(tokens) + self.pos_emb(positions)[None, :, :]
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.lm_head(x)
