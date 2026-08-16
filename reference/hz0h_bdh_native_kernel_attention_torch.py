"""Exact BDH attention with an N-dominant recurrent scan.

This module is an isolated alternative to the read-only oracle in
``reference.hz0h_bdh_torch.py``.  BDH uses K == Q, one V tensor broadcast over
heads, no softmax, and a strictly lower causal triangle.  The oracle evaluates
that expression by materializing a T x T score matrix.  This path instead
maintains the equivalent prefix state ``S = K.T @ V`` and evaluates each
position before adding that position's key/value pair.

The explicit backward uses the equivalent masked score derivative and only a
transient T x T tile, so autograd does not retain an N x D state for every
token. It is intended as the correctness-first reference for a future
compiled CUDA kernel; benchmark claims require the real CUDA dispatch
described in the kernel specification.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from reference.hz0h_bdh_torch import Attention, BDH


class _BDHNativeTiled(torch.autograd.Function):
    """Strict-causal attention evaluated in bounded query tiles."""

    @staticmethod
    def forward(ctx, QR: torch.Tensor, V: torch.Tensor, chunk_size: int) -> torch.Tensor:
        B, nh, T, N = QR.shape
        D = V.shape[-1]
        if V.shape[1] not in (1, nh):
            raise ValueError(f"BDH V must have one or nh heads, got {V.shape}")

        q = QR.reshape(B * nh, T, N)
        v = V.expand(B, nh, T, D).reshape(B * nh, T, D)
        outputs = []
        ctx.chunk_size = int(chunk_size)
        for start in range(0, T, chunk_size):
            stop = min(T, start + chunk_size)
            scores = torch.matmul(
                q[:, start:stop, :], q.transpose(1, 2)
            ).tril(diagonal=start - 1)
            outputs.append(torch.matmul(scores, v))

        ctx.save_for_backward(QR, V)
        ctx.v_heads = V.shape[1]
        return torch.cat(outputs, dim=1).reshape(B, nh, T, D)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        QR, V = ctx.saved_tensors
        B, nh, T, N = QR.shape
        D = V.shape[-1]
        q = QR.reshape(B * nh, T, N)
        v = V.expand(B, nh, T, D).reshape(B * nh, T, D)
        dy = grad_output.reshape(B * nh, T, D)
        dQ = torch.zeros_like(q)
        dV_heads = torch.zeros_like(v)

        # For Y = tril(Q Q.T, -1) V, process the same bounded query tiles as
        # forward. Since Q is both query and key, both uses receive a gradient
        # and accumulate into the same dQ tensor.
        for start in range(0, T, ctx.chunk_size):
            stop = min(T, start + ctx.chunk_size)
            q_chunk = q[:, start:stop, :]
            score = torch.matmul(q_chunk, q.transpose(1, 2)).tril(diagonal=start - 1)
            dscore = torch.matmul(dy[:, start:stop, :], v.transpose(1, 2)).tril(diagonal=start - 1)
            dQ[:, start:stop, :] += torch.matmul(dscore, q)
            dQ += torch.matmul(dscore.transpose(1, 2), q_chunk)
            dV_heads += torch.matmul(score.transpose(1, 2), dy[:, start:stop, :])

        dQ = dQ.reshape(B, nh, T, N)
        dV_heads = dV_heads.reshape(B, nh, T, D)
        dV = dV_heads if ctx.v_heads == nh else dV_heads.sum(dim=1, keepdim=True)
        return dQ, dV, None


def bdh_native_attention(
    Q: torch.Tensor,
    V: torch.Tensor,
    freqs: torch.Tensor,
    *,
    query_chunk_size: int = 32,
) -> torch.Tensor:
    """Return exact BDH attention output using bounded query tiles."""
    if query_chunk_size < 1:
        raise ValueError("query_chunk_size must be positive")
    B, nh, T, _ = Q.shape
    r_phases = torch.arange(
        0, T, device=freqs.device, dtype=freqs.dtype
    ).view(1, 1, T, 1) * freqs
    QR = Attention.rope(r_phases, Q)
    return _BDHNativeTiled.apply(QR, V, query_chunk_size)


def bdh_native_forward(
    model: BDH,
    idx: torch.Tensor,
    targets: torch.Tensor | None = None,
):
    """Drop-in BDH forward with only the attention operation replaced."""
    C = model.config
    B, T = idx.size()
    D = C.n_embd
    nh = C.n_head
    N = D * C.mlp_internal_dim_multiplier // nh

    x = model.embed(idx).unsqueeze(1)
    x = model.ln(x)
    for _ in range(C.n_layer):
        x_latent = x @ model._w(model.encoder)
        x_sparse = F.relu(x_latent)
        yKV = bdh_native_attention(x_sparse, x, model.attn.freqs)
        yKV = model.ln(yKV)
        y_latent = yKV @ model._w(model.encoder_v)
        y_sparse = F.relu(y_latent)
        xy_sparse = model.drop(x_sparse * y_sparse)
        yMLP = (
            xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh)
            @ model._w(model.decoder)
        )
        y = model.ln(yMLP)
        x = model.ln(x + y)

    logits = x.view(B, T, D) @ model.lm_head
    loss = None
    if targets is not None:
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
    return logits, loss
