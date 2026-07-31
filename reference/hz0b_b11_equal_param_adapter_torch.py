"""Torch port of `reference/hz0b_b11_equal_param_adapter.py` (the
"equal-parameter, no memory state at all" baseline), for the synthetic-
backbone B11 comparison run on CUDA. Same architecture and equation as
the MLX version -- per-position residual feed-forward transform, no
cross-position information flow of any kind."""
from __future__ import annotations

import torch
import torch.nn as nn


class EqualParamAdapter(nn.Module):
    def __init__(self, d_model: int, hidden: int, seed: int = 0):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.w1 = nn.Parameter(torch.empty(d_model, hidden).normal_(std=(2.0 / d_model) ** 0.5, generator=generator))
        self.b1 = nn.Parameter(torch.zeros(hidden))
        self.w2 = nn.Parameter(torch.empty(hidden, d_model).normal_(std=(2.0 / hidden) ** 0.5, generator=generator))
        self.b2 = nn.Parameter(torch.zeros(d_model))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        h = torch.relu(hidden @ self.w1 + self.b1)
        return hidden + (h @ self.w2 + self.b2)


def param_count(d_model: int, hidden: int) -> int:
    return d_model * hidden + hidden + hidden * d_model + d_model
