"""Packed-encoder checkpointed BDH (reference/hz0h_bdh_packed_encoder_torch.py)
with the tail of the per-round gate fused: `xy_sparse = x_sparse *
relu(y_latent)` runs as one Triton kernel
(reference/hz0h_bdh_fused_gate_torch.py) instead of a separate ReLU then
multiply. Zero change to BDH's math -- identical computation, proven
bit-exact against the unfused packed-encoder forward below.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from reference.hz0h_bdh_bmm_encoder_v_torch import bmm_encoder_v_step
from reference.hz0h_bdh_fused_gate_torch import fused_gate
from reference.hz0h_bdh_packed_encoder_torch import PackedEncoderBDH
from reference.hz0h_bdh_wide_gemm_encoder_torch import bdh_wide_gemm_encoder_step


def _bdh_packed_encoder_fused_gate_checkpoint_iteration(
    x: torch.Tensor,
    model: PackedEncoderBDH,
    B: int,
    T: int,
    D: int,
    nh: int,
    N: int,
) -> torch.Tensor:
    x_latent = bdh_wide_gemm_encoder_step(x, model.encoder_packed, nh, N)
    x_sparse = F.relu(x_latent)

    yKV = model.attn(Q=x_sparse, K=x_sparse, V=x)
    yKV = model.ln(yKV)

    y_latent = bmm_encoder_v_step(yKV, model.encoder_v)
    xy_sparse = fused_gate(x_sparse, y_latent)
    xy_sparse = model.drop(xy_sparse)

    yMLP = xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ model._w(model.decoder)
    y = model.ln(yMLP)
    x = model.ln(x + y)

    return x


def _bdh_packed_encoder_fused_gate_checkpoint_segment(
    x: torch.Tensor,
    model: PackedEncoderBDH,
    B: int,
    T: int,
    D: int,
    nh: int,
    N: int,
    segment_iterations: int,
) -> torch.Tensor:
    for _ in range(segment_iterations):
        x = _bdh_packed_encoder_fused_gate_checkpoint_iteration(x, model, B, T, D, nh, N)
    return x


def bdh_packed_encoder_fused_gate_forward_checkpointed(
    model: PackedEncoderBDH,
    idx: torch.Tensor,
    n_iterations: int,
    targets: torch.Tensor | None = None,
    checkpoint_segment_size: int = 1,
):
    if checkpoint_segment_size < 1:
        raise ValueError("checkpoint_segment_size must be at least 1")
    if n_iterations < 0:
        raise ValueError("n_iterations must be non-negative")

    C = model.config
    B, T = idx.size()
    D = C.n_embd
    nh = C.n_head
    N = D * C.mlp_internal_dim_multiplier // nh

    x = model.embed(idx).unsqueeze(1)
    x = model.ln(x)

    if not torch.is_grad_enabled():
        for _ in range(n_iterations):
            x = _bdh_packed_encoder_fused_gate_checkpoint_iteration(x, model, B, T, D, nh, N)
        logits = x.view(B, T, D) @ model.lm_head
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    iterations_remaining = n_iterations
    while iterations_remaining:
        segment_iterations = min(checkpoint_segment_size, iterations_remaining)
        x = torch.utils.checkpoint.checkpoint(
            _bdh_packed_encoder_fused_gate_checkpoint_segment,
            x,
            model,
            B,
            T,
            D,
            nh,
            N,
            segment_iterations,
            use_reentrant=False,
        )
        iterations_remaining -= segment_iterations

    logits = x.view(B, T, D) @ model.lm_head
    loss = None
    if targets is not None:
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

    return logits, loss
