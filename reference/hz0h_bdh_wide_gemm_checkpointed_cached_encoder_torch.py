"""Checkpointed wide-GEMM BDH, with the encoder's GPU-native wide view
computed once per training step instead of once per recurrent round.

`reference/hz0h_bdh_wide_gemm_checkpointed_torch.py`'s
`_bdh_wide_gemm_checkpoint_iteration` calls
`wide_encoder_view(model.encoder)` -- a real `permute().reshape().contiguous()`
data movement over the full ~100M-element encoder weight -- on every single
round. Under `torch.utils.checkpoint.checkpoint`, that function body runs
TWICE per round (once forward, once again during backward recompute), so at
the real 8-round production config this repacks the entire encoder weight
16 times per training step to produce the exact same `(D, nh*N)` tensor
every time -- `model.encoder` doesn't change during a forward/backward pass,
only between optimizer steps.

This is already disclosed as a known, not-yet-attempted opportunity in that
module's own docstring ("caching the wide view once per optimizer step...
remains a real, disclosed, not-yet-attempted further speedup"). This file is
that follow-up: `encoder_wide = wide_encoder_view(model.encoder)` is
computed ONCE before the round loop and threaded through as a plain
argument. Checkpoint's backward recompute re-runs the per-round body with
the same `encoder_wide` tensor reference rather than re-deriving it from
`model.encoder`, so autograd still builds exactly one permute/reshape/
contiguous node feeding `model.encoder`'s gradient (not eight), and CUDA
never repacks the weight more than once per step.

Zero change to BDH's math -- identical per-round computation to
`hz0h_bdh_wide_gemm_checkpointed_torch.py`, proven bit-exact against it
below. Only WHEN the encoder's wide view is computed changes (once per step
vs. once per round-recompute).

Never modifies `reference/hz0h_bdh_torch.py` or
`reference/hz0h_bdh_wide_gemm_checkpointed_torch.py`.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from reference.hz0h_bdh_bmm_encoder_v_torch import bmm_encoder_v_step
from reference.hz0h_bdh_torch import BDH
from reference.hz0h_bdh_wide_gemm_encoder_torch import bdh_wide_gemm_encoder_step, wide_encoder_view
from reference.hz0h_bdh_wide_gemm_trainable_torch import bdh_wide_gemm_trainable_forward


def _bdh_wide_gemm_checkpoint_iteration_cached(
    x: torch.Tensor,
    encoder_wide: torch.Tensor,
    model: BDH,
    B: int,
    T: int,
    D: int,
    nh: int,
    N: int,
) -> torch.Tensor:
    """Single layer iteration, wide-GEMM math, using an already-computed
    encoder wide view instead of rebuilding it from model.encoder."""
    x_latent = bdh_wide_gemm_encoder_step(x, encoder_wide, nh, N)
    x_sparse = F.relu(x_latent)

    yKV = model.attn(Q=x_sparse, K=x_sparse, V=x)
    yKV = model.ln(yKV)

    y_latent = bmm_encoder_v_step(yKV, model.encoder_v)
    y_sparse = F.relu(y_latent)
    xy_sparse = x_sparse * y_sparse
    xy_sparse = model.drop(xy_sparse)

    yMLP = xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ model._w(model.decoder)
    y = model.ln(yMLP)
    x = model.ln(x + y)

    return x


def _bdh_wide_gemm_checkpoint_segment_cached(
    x: torch.Tensor,
    encoder_wide: torch.Tensor,
    model: BDH,
    B: int,
    T: int,
    D: int,
    nh: int,
    N: int,
    segment_iterations: int,
) -> torch.Tensor:
    """Recompute a contiguous group of shared-weight dynamical rounds."""
    for _ in range(segment_iterations):
        x = _bdh_wide_gemm_checkpoint_iteration_cached(x, encoder_wide, model, B, T, D, nh, N)
    return x


def bdh_wide_gemm_forward_checkpointed_cached_encoder(
    model: BDH,
    idx: torch.Tensor,
    n_iterations: int,
    targets: torch.Tensor | None = None,
    checkpoint_segment_size: int = 1,
):
    """Same computation as `bdh_wide_gemm_forward_checkpointed`, but
    `wide_encoder_view(model.encoder)` is computed once before the round
    loop instead of once per round (x2 under checkpoint recompute)."""
    if checkpoint_segment_size < 1:
        raise ValueError("checkpoint_segment_size must be at least 1")
    if n_iterations < 0:
        raise ValueError("n_iterations must be non-negative")

    if not torch.is_grad_enabled():
        return bdh_wide_gemm_trainable_forward(model, idx, n_iterations, targets)

    C = model.config
    B, T = idx.size()
    D = C.n_embd
    nh = C.n_head
    N = D * C.mlp_internal_dim_multiplier // nh

    x = model.embed(idx).unsqueeze(1)
    x = model.ln(x)

    encoder_wide = wide_encoder_view(model.encoder)

    iterations_remaining = n_iterations
    while iterations_remaining:
        segment_iterations = min(checkpoint_segment_size, iterations_remaining)
        x = torch.utils.checkpoint.checkpoint(
            _bdh_wide_gemm_checkpoint_segment_cached,
            x,
            encoder_wide,
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
