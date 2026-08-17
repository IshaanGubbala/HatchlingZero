"""Exact BDH forward with sequence-only RoPE terms hoisted across depth.

BDH shares one attention module across every dynamical round. The positions,
frequency buffer, and sequence length are therefore identical in every round;
only the sparse query changes. The upstream oracle rebuilds phases, cosine,
and sine inside each attention call. This executor computes those invariant
terms once while preserving the oracle as an independent reference.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from reference.hz0h_bdh_torch import BDH


def _rope_with_cos_sin(
    values: torch.Tensor,
    phases_cos: torch.Tensor,
    phases_sin: torch.Tensor,
) -> torch.Tensor:
    rotated = torch.stack(
        (-values[..., 1::2], values[..., ::2]),
        dim=-1,
    ).view_as(values)
    return (values * phases_cos).to(values.dtype) + (
        rotated * phases_sin
    ).to(values.dtype)


def bdh_hoisted_rope_forward(
    model: BDH,
    idx: torch.Tensor,
    targets: torch.Tensor | None = None,
):
    """Run exact BDH math while constructing RoPE trig terms only once."""
    config = model.config
    batch_size, sequence_length = idx.shape
    dim = config.n_embd
    heads = config.n_head
    neurons_per_head = dim * config.mlp_internal_dim_multiplier // heads

    positions = torch.arange(
        sequence_length,
        device=model.attn.freqs.device,
        dtype=model.attn.freqs.dtype,
    ).view(1, 1, -1, 1)
    phases = positions * model.attn.freqs
    phases_cos, phases_sin = model.attn.phases_cos_sin(phases)

    x = model.ln(model.embed(idx).unsqueeze(1))
    for _ in range(config.n_layer):
        x_sparse = F.relu(x @ model._w(model.encoder))
        queries = _rope_with_cos_sin(x_sparse, phases_cos, phases_sin)
        scores = (queries @ queries.mT).tril(diagonal=-1)
        attended = model.ln(scores @ x)

        y_sparse = F.relu(attended @ model._w(model.encoder_v))
        gated = model.drop(x_sparse * y_sparse)
        projected = (
            gated.transpose(1, 2).reshape(
                batch_size,
                1,
                sequence_length,
                neurons_per_head * heads,
            )
            @ model._w(model.decoder)
        )
        x = model.ln(x + model.ln(projected))

    logits = x.view(batch_size, sequence_length, dim) @ model.lm_head
    loss = None
    if targets is not None:
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
        )
    return logits, loss
