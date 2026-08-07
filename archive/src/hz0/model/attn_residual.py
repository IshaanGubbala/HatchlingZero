from __future__ import annotations

import math

import torch
from torch import nn


class AttentionResidual(nn.Module):
    """Depth-wise attention over previous layer representations, in place of
    the standard "carry forward the immediately previous layer's output"
    residual chaining.

    Standard PreNorm residual chaining feeds layer l only x_{l-1}. This
    module instead computes a learned, content-dependent combination over
    ALL previous layer outputs (including the embedding) at each sequence
    position independently:

        x_l_input = sum_{i<l} alpha_{l,i}(x) * h_i

    where alpha is a softmax attention distribution over depth (not
    sequence position) computed from a query derived from the most recent
    representation and keys derived from every prior representation. This
    is a per-layer module (each layer gets its own query/key projections,
    matching the layer-indexed alpha_{l,i} in the formula) -- it replaces
    what gets fed INTO the next layer's mixer/attention/FFN stack, not
    those sub-blocks' own internal residual connections, which are
    untouched.

    `rank`: the query/key projection dimension. `None` (or equal to
    `d_model`) is the full-rank variant; smaller values are the
    "low-rank AttnRes" variant -- only the ROUTING (query/key) side is
    low-rank, the VALUES stay full `d_model` width, so representational
    capacity of what gets passed forward is not reduced.

    `n_heads`: independent depth-attention distributions, each reading a
    distinct `d_model // n_heads` slice of the VALUE (matching standard
    multi-head attention's value-splitting), each with its own
    `rank // n_heads`-wide query/key subspace. `n_heads=1` is the
    single-head variant.
    """

    def __init__(self, d_model: int, rank: int | None = None, n_heads: int = 1) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        self.n_heads = n_heads
        self.value_head_dim = d_model // n_heads
        self.rank = rank if rank is not None else d_model
        if self.rank % n_heads != 0:
            raise ValueError(f"rank={self.rank} must be divisible by n_heads={n_heads}")
        self.qk_head_dim = self.rank // n_heads
        self.query = nn.Linear(d_model, self.rank, bias=False)
        self.key = nn.Linear(d_model, self.rank, bias=False)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        """`history`: `[batch, seq, depth, d_model]`, depth = number of
        prior layer outputs available (including the embedding), ordered
        oldest-first -- `history[:, :, -1]` is the most recent
        representation. Returns `[batch, seq, d_model]`, the depth-mixed
        representation to feed into the next layer."""
        batch, seq, depth, d_model = history.shape
        current = history[:, :, -1]

        q = self.query(current).view(batch, seq, self.n_heads, self.qk_head_dim)
        k = self.key(history).view(batch, seq, depth, self.n_heads, self.qk_head_dim)
        scores = torch.einsum("bshe,bslhe->bshl", q, k) / math.sqrt(self.qk_head_dim)
        weights = torch.softmax(scores, dim=-1)

        v = history.view(batch, seq, depth, self.n_heads, self.value_head_dim)
        pooled = torch.einsum("bshl,bslhd->bshd", weights, v)
        return pooled.reshape(batch, seq, d_model)
