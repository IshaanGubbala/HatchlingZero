"""Composition-encoder ablation, plans/Hatchling world.md L3/L4-logic
diagnostic (2026-09-04): held-out UNSEEN (size, color) combinations
plateau at 30-60%, well above chance (25%) but far below the ~100%
ceiling every grounding/selection task otherwise reaches. One disclosed
candidate cause, never tested: `HZLanguageModel.object_encoder`
concatenates type/color/size/position one-hots into a single vector and
mixes them with ONE shared `nn.Linear` -- nothing in that layer is
structurally prevented from entangling color and size arbitrarily, so
there is no inductive bias pushing the model toward a representation
where "red" and "small" contribute independently. The standard fix in
the compositional-generalization literature is an explicitly ADDITIVE/
factorized representation: give each attribute its own embedding table
and SUM them, so composing two properties is structurally just adding
two independent vectors, not something a shared linear layer has to
learn to keep separable on its own.

Both encoders here have the identical `forward(type_idx, color_idx,
size_idx, position_idx) -> (B, N_obj, D)` signature so a controlled
ablation script can swap one for the other while leaving everything
else (mem, ws, sel_rq/sel_rk, the whole S/H reasoning pathway) exactly
as validated.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConcatLinearEncoder(nn.Module):
    """Control: reproduces HZLanguageModel.encode_objects/object_encoder
    exactly. Concatenate one-hots, mix with one shared Linear -- no
    structural separation between attributes."""

    def __init__(self, d_model: int):
        super().__init__()
        self.linear = nn.Linear(4 + 4 + 2 + 2, d_model, bias=False)

    def forward(self, type_idx: torch.Tensor, color_idx: torch.Tensor,
                size_idx: torch.Tensor, position_idx: torch.Tensor) -> torch.Tensor:
        feat = torch.cat([
            F.one_hot(type_idx, 4).float(), F.one_hot(color_idx, 4).float(),
            F.one_hot(size_idx, 2).float(), F.one_hot(position_idx, 2).float(),
        ], dim=-1)
        return self.linear(feat)


class FactorizedSumEncoder(nn.Module):
    """Test: each attribute gets its own independent embedding table;
    the object's representation is the SUM of the four -- composing
    "small" and "red" is structurally just vector addition, not
    something a shared linear layer has to learn to keep separable."""

    def __init__(self, d_model: int):
        super().__init__()
        self.type_embed = nn.Embedding(4, d_model)
        self.color_embed = nn.Embedding(4, d_model)
        self.size_embed = nn.Embedding(2, d_model)
        self.position_embed = nn.Embedding(2, d_model)

    def forward(self, type_idx: torch.Tensor, color_idx: torch.Tensor,
                size_idx: torch.Tensor, position_idx: torch.Tensor) -> torch.Tensor:
        return (self.type_embed(type_idx) + self.color_embed(color_idx)
                + self.size_embed(size_idx) + self.position_embed(position_idx))


ENCODER_VARIANTS = {
    "concat_linear": ConcatLinearEncoder,
    "factorized_sum": FactorizedSumEncoder,
}
