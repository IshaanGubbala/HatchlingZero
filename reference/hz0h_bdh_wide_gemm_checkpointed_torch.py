"""Combines two previously-separate, previously-separately-validated
memory fixes that were never tested together (a real, disclosed gap
noted in `reference/hz0h_bdh_wide_gemm_trainable_torch.py`'s own
docstring: "bdh_wide_gemm_trainable_forward has no checkpointed
variant").

Real motivation (2026-08-21, RunPod A40 dispatch): a saved-tensor audit
(`torch.autograd.graph.saved_tensors_hooks`) found raw_bdh's 37.8GB
training peak at batch=8 was 67% (25.5GB) a single bug -- the plain
`x @ model._w(model.encoder)` broadcast-matmul causes autograd to save
a FULLY EXPANDED `(B*n_head, D, N)` copy of the encoder/encoder_v
weight for backward, 8x larger than the true `(n_head, D, N)` weight.
`bdh_wide_gemm_trainable_forward` (a dense-reshape reformulation of the
same math, already proven bit-exact) sidesteps this entirely: re-
auditing with it dropped peak memory to 13.9GB with ZERO checkpointing.

But a SEPARATE follow-up probe (checkpointing alone, no wide-GEMM) found
`checkpoint_segment_size=1` gets batch=8's backward peak down to 8.0GB
-- LOWER than wide-GEMM alone's 13.9GB -- because checkpointing avoids
retaining ALL `n_iterations` rounds' activations simultaneously (the
reason checkpointing exists at all), while wide-GEMM-without-
checkpointing still does, just at the bug-fixed (not bug-inflated) size
per round. The two fixes attack DIFFERENT problems (one dtype/shape
bug in a single op; one structural "backward needs every round's
activations" cost) and are not mutually exclusive -- this file combines
them: the checkpointed segment body uses the wide-GEMM math instead of
the plain broadcast matmul, so backward never re-derives the broadcast-
expansion bug's cost NOR retains every round simultaneously.

Same real per-layer computation as `bdh_variable_depth_forward` (via
`bdh_wide_gemm_trainable_forward`'s already-proven-bit-exact
reformulation) -- zero change to BDH's math, only WHEN activations are
computed (checkpointed recompute during backward, like every other
checkpointed variant in this project) and HOW the encoder/encoder_v
projections are executed (wide-GEMM/bmm layout, not broadcast). Proven
bit-exact (logits + gradients) against `bdh_variable_depth_forward` by
`tests/reference/test_hz0h_bdh_wide_gemm_checkpointed_torch.py`.

Never modifies `reference/hz0h_bdh_torch.py`,
`reference/hz0h_bdh_variable_depth_torch.py`,
`reference/hz0h_bdh_checkpointed_torch.py`,
`reference/hz0h_bdh_wide_gemm_trainable_torch.py`,
`reference/hz0h_bdh_wide_gemm_encoder_torch.py`, or
`reference/hz0h_bdh_bmm_encoder_v_torch.py`.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from reference.hz0h_bdh_bmm_encoder_v_torch import bmm_encoder_v_step
from reference.hz0h_bdh_torch import BDH
from reference.hz0h_bdh_wide_gemm_encoder_torch import bdh_wide_gemm_encoder_step, wide_encoder_view
from reference.hz0h_bdh_wide_gemm_trainable_torch import bdh_wide_gemm_trainable_forward


def _bdh_wide_gemm_checkpoint_iteration(
    x: torch.Tensor,
    model: BDH,
    B: int,
    T: int,
    D: int,
    nh: int,
    N: int,
) -> torch.Tensor:
    """Single layer iteration, wide-GEMM math, for checkpointing. Passed
    to torch.utils.checkpoint.checkpoint, recomputed during backward
    rather than stored -- see reference/hz0h_bdh_checkpointed_torch.py's
    `_bdh_checkpoint_iteration` for the plain-matmul analog this
    mirrors exactly, math swapped for the wide-GEMM/bmm layout."""
    encoder_wide = wide_encoder_view(model.encoder)
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


def _bdh_wide_gemm_checkpoint_segment(
    x: torch.Tensor,
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
        x = _bdh_wide_gemm_checkpoint_iteration(x, model, B, T, D, nh, N)
    return x


def bdh_wide_gemm_forward_checkpointed(
    model: BDH,
    idx: torch.Tensor,
    n_iterations: int,
    targets: torch.Tensor | None = None,
    checkpoint_segment_size: int = 1,
):
    """Same real per-layer computation as `bdh_wide_gemm_trainable_forward`,
    but groups of iterations are wrapped in
    torch.utils.checkpoint.checkpoint (use_reentrant=False) to avoid
    retaining all `n_iterations` rounds' wide-GEMM activations
    simultaneously. See this module's own docstring for why this
    combination (not either fix alone) is the real target.

    Args:
        model: BDH instance
        idx: input token indices, shape (B, T)
        n_iterations: number of layer-loop iterations (independent of
                      model.config.n_layer)
        targets: optional target tokens for loss computation, shape (B, T)
        checkpoint_segment_size: rounds recomputed as one segment. One
                                 gives minimum memory; larger values
                                 reduce boundary overhead while
                                 retaining more activations.

    Returns:
        (logits, loss): logits shape (B, T, vocab_size), loss scalar or None
    """
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

    iterations_remaining = n_iterations
    while iterations_remaining:
        segment_iterations = min(checkpoint_segment_size, iterations_remaining)
        x = torch.utils.checkpoint.checkpoint(
            _bdh_wide_gemm_checkpoint_segment,
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
