"""Symmetry-reduced BDH attention backward, with the value-broadcast copy
removed (`reference/hz0h_bdh_symmetric_backward_torch.py`, commit d83db47).

That version's backward does:

    value_heads = value.expand(batch, heads, sequence, dim).reshape(
        batch * heads, sequence, dim
    )

`value` has shape ``(B, 1, T, D)`` and is the SAME tensor for every head.
``.expand()`` makes a zero-copy view (stride 0 on the head dim), but the
following ``.reshape()`` cannot return a view over a stride-0 dimension, so
it materializes a real ``(B, H, T, D)`` copy -- literally ``H`` duplicate
copies of the same ``(T, D)`` matrix in memory -- purely so the flattened
tensor can be passed to ``torch.bmm``, which requires exact batch-dim
matches. This is the identical class of bug found earlier in this project
in the wide-GEMM encoder path: an implicit broadcast forced into a real
copy by reshaping across a non-contiguous expanded dimension.

``torch.matmul`` (unlike ``torch.bmm``) broadcasts leading batch dimensions
natively -- ``(B, H, T, D) @ (B, 1, D, T)`` runs directly with no
materialized duplicate of ``value``. This module is otherwise identical:
same forward, same exact ``dQ = (dS + dS.T) @ Q`` algebraic identity, same
saved tensors. It removes only the reshape/bmm plumbing.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from reference.hz0h_bdh_bmm_encoder_v_torch import bmm_encoder_v_step
from reference.hz0h_bdh_torch import Attention, BDH
from reference.hz0h_bdh_wide_gemm_encoder_torch import (
    bdh_wide_gemm_encoder_step,
    wide_encoder_view,
)


class _BDHSymmetricBroadcastBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        if q.ndim != 4 or value.ndim != 4:
            raise ValueError("q and value must have shape (B, heads, T, width)")
        batch, heads, sequence, _ = q.shape
        if value.shape[:3] != (batch, 1, sequence):
            raise ValueError("BDH value must have shape (B, 1, T, D)")

        scores = (q @ q.mT).tril(diagonal=-1)
        output = scores @ value
        ctx.save_for_backward(q, value, scores)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        q, value, scores = ctx.saved_tensors

        # (B,H,T,D) @ (B,1,D,T) broadcasts over the head dim in place --
        # no materialized per-head copy of value.
        dscore = torch.matmul(grad_output, value.transpose(-1, -2)).tril(diagonal=-1)
        symmetric_dscore = dscore + dscore.transpose(-1, -2)
        dq = torch.matmul(symmetric_dscore, q)

        dvalue = torch.matmul(scores.transpose(-1, -2), grad_output).sum(dim=1, keepdim=True)
        return dq, dvalue


def bdh_symmetric_broadcast_backward_attention(
    q: torch.Tensor,
    value: torch.Tensor,
    freqs: torch.Tensor,
) -> torch.Tensor:
    """Apply the faithful BDH RoPE and exact symmetry-reduced attention."""
    sequence = q.shape[2]
    phases = torch.arange(
        sequence,
        device=freqs.device,
        dtype=freqs.dtype,
    ).view(1, 1, sequence, 1) * freqs
    rotated = Attention.rope(phases, q)
    return _BDHSymmetricBroadcastBackward.apply(rotated, value)


def bdh_symmetric_broadcast_backward_forward(
    model: BDH,
    idx: torch.Tensor,
    n_iterations: int,
    targets: torch.Tensor | None = None,
):
    """Full trainable wide-GEMM BDH using the broadcast-reduced attention backward."""
    config = model.config
    batch, sequence = idx.shape
    dim = config.n_embd
    heads = config.n_head
    latent = dim * config.mlp_internal_dim_multiplier // heads

    x = model.ln(model.embed(idx).unsqueeze(1))
    encoder_wide = wide_encoder_view(model.encoder)
    for _ in range(n_iterations):
        x_sparse = F.relu(
            bdh_wide_gemm_encoder_step(x, encoder_wide, heads, latent)
        )
        y_kv = bdh_symmetric_broadcast_backward_attention(x_sparse, x, model.attn.freqs)
        y_kv = model.ln(y_kv)
        y_sparse = F.relu(bmm_encoder_v_step(y_kv, model.encoder_v))
        decoder_input = (x_sparse * y_sparse).transpose(1, 2).reshape(
            batch, 1, sequence, heads * latent
        )
        y = model.ln(decoder_input @ model.decoder)
        x = model.ln(x + y)

    logits = x.view(batch, sequence, dim) @ model.lm_head
    loss = None
    if targets is not None:
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
    return logits, loss
