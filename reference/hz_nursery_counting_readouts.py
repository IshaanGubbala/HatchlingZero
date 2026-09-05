"""Counting-readout ablation, plans/Hatchling world.md L4 diagnostic
(2026-09-04): L4's counting-verification task plateaus at 65-72% TRAIN
accuracy even after 5000 steps -- a real capacity ceiling, not a
generalization gap. Before touching HZCQReasoningWorkspace's recurrence
at all, the cheap and correct diagnostic is: is H actually carrying
enough information to count, and the READOUT (mean-pooling H down to
one vector, then one linear head) is just the wrong shape for an
aggregation task? These four readouts all consume the EXACT SAME
(x_objects, H) produced by HZLanguageModel.encode_and_reason -- the
backbone (token_embed, mem, ws, object_encoder) is frozen and identical
across all four, so any accuracy difference is attributable ONLY to the
readout, not to H's internal computation. Matches PAPER-0's own
warning, now explicitly invoked by the user: don't assume a failure in
the final answer means H itself failed.

Each readout is `forward(x_objects, H) -> logit (B,)`.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MeanPoolReadout(nn.Module):
    """The CONTROL condition -- structurally identical to
    HZLanguageModel.verify_count_forward's existing head (mean over H's
    workspace slots, one linear classifier), but with its own freshly
    initialized weights so every variant gets an equal head-only
    training budget in the ablation."""

    def __init__(self, d_model: int):
        super().__init__()
        self.head = nn.Linear(d_model, 1, bias=True)

    def forward(self, x_objects: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        pooled = H.mean(dim=1)
        return self.head(pooled).squeeze(-1)


class SumPoolReadout(nn.Module):
    """Same shape as MeanPoolReadout, but SUMS instead of averaging over
    H's slots. Cheapest possible test of the specific hypothesis that
    mean-normalization is what erases cardinality information: mean
    pooling makes the magnitude of the pooled vector roughly
    INDEPENDENT of how many slots "voted" for something, which is
    exactly the signal a count needs."""

    def __init__(self, d_model: int):
        super().__init__()
        self.head = nn.Linear(d_model, 1, bias=True)

    def forward(self, x_objects: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        pooled = H.sum(dim=1)
        return self.head(pooled).squeeze(-1)


class AttnPoolReadout(nn.Module):
    """Replaces the FIXED uniform mean over H's slots with a LEARNED
    pooling query -- lets training decide which workspace slots matter
    for this task instead of averaging all of them uniformly. Still a
    single pooled vector into one linear head, so this isolates
    "does learned pooling beat fixed pooling" without yet testing
    per-object aggregation (that's PredicateSumReadout)."""

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.head = nn.Linear(d_model, 1, bias=True)

    def forward(self, x_objects: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        B = H.shape[0]
        q = self.query.expand(B, 1, self.d_model)
        scores = torch.matmul(q, self.key(H).transpose(-1, -2)) / (self.d_model ** 0.5)  # (B, 1, M_H)
        attn = F.softmax(scores, dim=-1)
        pooled = torch.matmul(attn, self.value(H)).squeeze(1)  # (B, D)
        return self.head(pooled).squeeze(-1)


class PredicateSumReadout(nn.Module):
    """Structurally different from the other three: scores EACH OBJECT
    individually for "does this object match the queried property"
    (a per-object predicate, via attention between a pooled-H query and
    each object's own encoding), turns that into a soft match
    probability per object, and SUMS those probabilities into a
    differentiable count estimate -- literally
    \\sum_i P(\\text{object}_i \\text{ matches}) -- before a final small
    head compares that estimate against whatever the pooled reasoning
    state carries about the stated number. If this variant alone fixes
    counting, the bottleneck was "aggregate need a per-item score summed
    up," not H's capacity to represent the count at all."""

    def __init__(self, d_model: int):
        super().__init__()
        self.predicate_q = nn.Linear(d_model, d_model, bias=False)
        self.predicate_k = nn.Linear(d_model, d_model, bias=False)
        self.head = nn.Linear(d_model + 1, 1, bias=True)

    def forward(self, x_objects: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        pooled_h = H.mean(dim=1)  # (B, D) -- still carries "what number was stated"
        q = self.predicate_q(pooled_h).unsqueeze(1)  # (B, 1, D)
        k = self.predicate_k(x_objects)  # (B, N_obj, D)
        match_logits = torch.matmul(q, k.transpose(-1, -2)).squeeze(1) / (x_objects.shape[-1] ** 0.5)  # (B, N_obj)
        match_probs = torch.sigmoid(match_logits)  # (B, N_obj) -- soft per-object indicator
        predicted_count = match_probs.sum(dim=-1, keepdim=True)  # (B, 1) -- differentiable \sum_i P(object_i matches)
        return self.head(torch.cat([predicted_count, pooled_h], dim=-1)).squeeze(-1)


READOUT_VARIANTS = {
    "mean_pool": MeanPoolReadout,
    "sum_pool": SumPoolReadout,
    "attn_pool": AttnPoolReadout,
    "predicate_sum": PredicateSumReadout,
}
