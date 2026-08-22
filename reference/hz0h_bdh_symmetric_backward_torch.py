"""Exact dense BDH attention with a symmetry-reduced backward.

BDH uses the same RoPE-transformed tensor for Q and K. For

    S = tril(Q Q^T, -1),  Y = S V

the two gradient contributions that generic autograd obtains from Q's query
and key roles can be combined before multiplication:

    dQ = (dS + dS^T) Q

This uses one ``T x T`` by ``T x N`` dense GEMM instead of two. It changes no
model values, support, precision, or parameterization and keeps vendor dense
matmul rather than introducing a custom sparse or tiled kernel.
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


class _BDHSymmetricBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        if q.ndim != 4 or value.ndim != 4:
            raise ValueError("q and value must have shape (B, heads, T, width)")
        batch, heads, sequence, _ = q.shape
        if value.shape[:3] != (batch, 1, sequence):
            raise ValueError("BDH value must have shape (B, 1, T, D)")

        # Every prior benchmark of this Function fed q/value as manually
        # constructed same-dtype tensors, never through a real autocast-
        # wrapped forward. Under real autocast, LayerNorm is forced to
        # return fp32 regardless of ambient dtype (the same lesson learned
        # building FlashBDH earlier in this project), so a LayerNorm'd
        # `value` can arrive fp32 while `q` arrives bf16 -- backward (which
        # runs outside the caller's autocast scope entirely) then mixes
        # dtypes in its manual bmm calls and crashes. Disable autocast for
        # this Function's own body and explicitly normalize both inputs to
        # q's dtype so forward and backward always operate on one dtype.
        compute_dtype = q.dtype
        with torch.autocast(device_type=q.device.type, enabled=False):
            value = value.to(compute_dtype)
            scores = (q @ q.mT).tril(diagonal=-1)
            output = scores @ value
        ctx.save_for_backward(q, value, scores)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        q, value, scores = ctx.saved_tensors
        grad_output = grad_output.to(q.dtype)
        batch, heads, sequence, latent = q.shape
        dim = value.shape[-1]

        q_flat = q.reshape(batch * heads, sequence, latent)
        grad_flat = grad_output.reshape(batch * heads, sequence, dim)
        value_heads = value.expand(batch, heads, sequence, dim).reshape(
            batch * heads, sequence, dim
        )
        scores_flat = scores.reshape(batch * heads, sequence, sequence)

        dscore = torch.bmm(grad_flat, value_heads.transpose(1, 2)).tril(diagonal=-1)
        symmetric_dscore = dscore + dscore.transpose(1, 2)
        dq = torch.bmm(symmetric_dscore, q_flat).reshape_as(q)

        dvalue_heads = torch.bmm(scores_flat.transpose(1, 2), grad_flat).reshape(
            batch, heads, sequence, dim
        )
        dvalue = dvalue_heads.sum(dim=1, keepdim=True)
        return dq, dvalue


def bdh_symmetric_backward_attention(
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
    return _BDHSymmetricBackward.apply(rotated, value)


def bdh_symmetric_backward_forward(
    model: BDH,
    idx: torch.Tensor,
    n_iterations: int,
    targets: torch.Tensor | None = None,
):
    """Full trainable wide-GEMM BDH using the reduced attention backward."""
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
        y_kv = bdh_symmetric_backward_attention(x_sparse, x, model.attn.freqs)
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
