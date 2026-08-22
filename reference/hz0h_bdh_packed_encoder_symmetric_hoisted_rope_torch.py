"""Packed encoder + symmetric backward (981d8e1), with RoPE's cos/sin
terms hoisted once per training step instead of recomputed once per round.

BDH shares one attention module across every recurrent round: the
sequence positions and frequency buffer are identical every round, only
the query (`x_sparse`) changes. `Attention.rope(phases, v)` recomputes
`phases_cos_sin` (a mod + cos + sin over a `(1,1,T,N)`-broadcast tensor)
from scratch on every call -- `reference/hz0h_bdh_hoisted_rope_torch.py`
proved this is real, exact, hoistable math (never modifies the oracle,
and `Attention.phases_cos_sin`/`Attention.rope` confirm the identical
formula), but that file was a standalone correctness proof against the
oracle's plain broadcast-matmul path -- it was never actually integrated
with this session's checkpointed/packed/symmetric-backward path, nor ever
benchmarked for a real measured speedup. This file is that integration.

`_rope_with_cos_sin` is bit-for-bit the same formula as
`Attention.rope`'s internals (confirmed by reading
`reference/hz0h_bdh_torch.py` directly: `phases_cos, phases_sin =
Attention.phases_cos_sin(phases); return (v*phases_cos) + (rotate_half(v)
* phases_sin)`), so precomputing `phases_cos`/`phases_sin` once and
reusing them across all `n_iterations` rounds changes nothing about the
math -- only how many times the same cos/sin values get recomputed.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from reference.hz0h_bdh_bmm_encoder_v_torch import bmm_encoder_v_step
from reference.hz0h_bdh_packed_encoder_torch import PackedEncoderBDH
from reference.hz0h_bdh_symmetric_backward_torch import _BDHSymmetricBackward
from reference.hz0h_bdh_torch import Attention
from reference.hz0h_bdh_wide_gemm_encoder_torch import bdh_wide_gemm_encoder_step


def _rope_with_cos_sin(v: torch.Tensor, phases_cos: torch.Tensor, phases_sin: torch.Tensor) -> torch.Tensor:
    v_rot = torch.stack((-v[..., 1::2], v[..., ::2]), dim=-1).view_as(v)
    return (v * phases_cos).to(v.dtype) + (v_rot * phases_sin).to(v.dtype)


def bdh_symmetric_backward_attention_hoisted(
    q: torch.Tensor,
    value: torch.Tensor,
    phases_cos: torch.Tensor,
    phases_sin: torch.Tensor,
) -> torch.Tensor:
    """Same as bdh_symmetric_backward_attention, but takes already-computed
    RoPE cos/sin terms instead of recomputing them from freqs every call."""
    rotated = _rope_with_cos_sin(q, phases_cos, phases_sin)
    return _BDHSymmetricBackward.apply(rotated, value)


def _bdh_packed_encoder_symmetric_hoisted_checkpoint_iteration(
    x: torch.Tensor,
    model: PackedEncoderBDH,
    phases_cos: torch.Tensor,
    phases_sin: torch.Tensor,
    B: int,
    T: int,
    D: int,
    nh: int,
    N: int,
) -> torch.Tensor:
    x_latent = bdh_wide_gemm_encoder_step(x, model.encoder_packed, nh, N)
    x_sparse = F.relu(x_latent)

    yKV = bdh_symmetric_backward_attention_hoisted(x_sparse, x, phases_cos, phases_sin)
    yKV = model.ln(yKV)

    y_latent = bmm_encoder_v_step(yKV, model.encoder_v)
    y_sparse = F.relu(y_latent)
    xy_sparse = x_sparse * y_sparse
    xy_sparse = model.drop(xy_sparse)

    yMLP = xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ model._w(model.decoder)
    y = model.ln(yMLP)
    x = model.ln(x + y)

    return x


def _bdh_packed_encoder_symmetric_hoisted_checkpoint_segment(
    x: torch.Tensor,
    model: PackedEncoderBDH,
    phases_cos: torch.Tensor,
    phases_sin: torch.Tensor,
    B: int,
    T: int,
    D: int,
    nh: int,
    N: int,
    segment_iterations: int,
) -> torch.Tensor:
    for _ in range(segment_iterations):
        x = _bdh_packed_encoder_symmetric_hoisted_checkpoint_iteration(x, model, phases_cos, phases_sin, B, T, D, nh, N)
    return x


def bdh_packed_encoder_symmetric_hoisted_rope_forward_checkpointed(
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

    positions = torch.arange(T, device=model.attn.freqs.device, dtype=model.attn.freqs.dtype).view(1, 1, T, 1)
    phases = positions * model.attn.freqs
    phases_cos, phases_sin = Attention.phases_cos_sin(phases)

    if not torch.is_grad_enabled():
        for _ in range(n_iterations):
            x = _bdh_packed_encoder_symmetric_hoisted_checkpoint_iteration(x, model, phases_cos, phases_sin, B, T, D, nh, N)
        logits = x.view(B, T, D) @ model.lm_head
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    iterations_remaining = n_iterations
    while iterations_remaining:
        segment_iterations = min(checkpoint_segment_size, iterations_remaining)
        x = torch.utils.checkpoint.checkpoint(
            _bdh_packed_encoder_symmetric_hoisted_checkpoint_segment,
            x,
            model,
            phases_cos,
            phases_sin,
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
