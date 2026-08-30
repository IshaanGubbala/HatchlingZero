"""The crux ablation, 2026-08-29: BDH-Delta bundled TWO changes into one
experiment -- decoupled refresh cadence (speed idea) AND a brand-new
compressed reasoning system (384-d belief, 8x96 workspace, Think Cell,
belief cell, cross-chunk carry, convergence machinery -- all freshly
initialized, ~4.61M new params). The real result (val_loss=1.7862,
+0.35-0.37 over baseline) can't tell us which change broke quality.

This file isolates the FIRST change alone. It is deliberately the most
boring possible decoupled-refresh BDH:

    e_j       = A_exact(h_j, x)              expensive, every K-th iteration
    h_{j,1}   = F_existing(h_j, e_j)          cheap, reuses cached e
    h_{j,2}   = F_existing(h_{j,1}, e_j)      cheap, reuses cached e
    ...
    e_{j+1}   = A_exact(h_{j,M}, x)           next refresh

ZERO new parameters. Full D=2496 state throughout (no belief bottleneck,
no workspace, no bridges). F_existing is EXACTLY the existing compound
model's per-round computation (reference/hz0h_bdh_vb_subspace_decoder_gated_residual_torch.py's
single-stream g1 arm -- the one validated real win this session
produced), just with its own attention output (`yKV`, here renamed `e`
to match the plan's own notation) supplied from a cache instead of
recomputed every iteration. At refresh_every=1 (refresh every
iteration, no caching ever happens) this is mathematically IDENTICAL to
that existing gated-residual single-stream forward -- verified locally,
bit-for-bit, before any GPU spend (see the smoke test this file's
commit includes).

A_exact is the expensive part specifically: the O(T^2) self-attention
call (`model.attn`) plus its O-projection and LN. x_sparse (from
encoder) IS recomputed every iteration, even non-refresh ones -- it's
a cheap dense projection of the CURRENT state, not the expensive part,
and xy_sparse's gating (x_sparse * y_sparse) needs it fresh every step
regardless of whether e is cached.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
import torch.utils.checkpoint

from reference.hz0h_bdh_vb_subspace_decoder_torch import BDHVBSubspaceDecoder


def refresh_schedule(n_iterations: int, n_refresh: int) -> set[int]:
    """n_refresh evenly-spaced iteration indices (0-indexed) within
    range(n_iterations) that trigger a real exact-address refresh; every
    other iteration reuses the most recent cached evidence. Always
    includes iteration 0 (no evidence exists before the first refresh).
    n_refresh=n_iterations -> every iteration refreshes (the known-good
    baseline, mathematically). Doesn't require n_refresh to evenly
    divide n_iterations (real requirement from the plan's own K=6-over-8
    case)."""
    assert 1 <= n_refresh <= n_iterations
    idxs = {round(i * n_iterations / n_refresh) for i in range(n_refresh)}
    idxs.add(0)
    return idxs


def _existing_compute(x: torch.Tensor, x_sparse: torch.Tensor, e: torch.Tensor,
                       model: BDHVBSubspaceDecoder, nh: int, N: int) -> torch.Tensor:
    """The unchanged rest of the compound+g1 round: encoder_v -> relu ->
    gate with x_sparse -> decoder_up compress -> decoder_down expand ->
    g1-gated residual add. Identical to
    hz0h_bdh_vb_subspace_decoder_gated_residual_torch.py's single-stream
    arm, just taking e (the attention output) as an argument instead of
    computing it inline."""
    y_latent = e @ model.encoder_v
    y_sparse = F.relu(y_latent)
    xy_sparse = model.drop(x_sparse * y_sparse)
    alpha = torch.matmul(xy_sparse, model.decoder_up.view(nh, N, -1)).sum(dim=1, keepdim=True)
    y1 = model.ln(alpha @ model.decoder_down)
    y = model.g1 * y1 if hasattr(model, "g1") else y1
    return model.ln(x + y)


def _address(x: torch.Tensor, model: BDHVBSubspaceDecoder, nh: int, N: int) -> torch.Tensor:
    """A_exact(h, x) -- the expensive part: real self-attention, O(T^2)."""
    x_sparse = F.relu(x @ model.encoder)
    v_bottleneck = x @ model.P
    yKV_bottleneck = model.attn(Q=x_sparse, K=x_sparse, V=v_bottleneck)
    return model.ln(yKV_bottleneck @ model.O)


def _refresh_iteration(x: torch.Tensor, model: BDHVBSubspaceDecoder, B: int, T: int, D: int, nh: int, N: int):
    x_sparse = F.relu(x @ model.encoder)
    v_bottleneck = x @ model.P
    yKV_bottleneck = model.attn(Q=x_sparse, K=x_sparse, V=v_bottleneck)
    e = model.ln(yKV_bottleneck @ model.O)
    x_new = _existing_compute(x, x_sparse, e, model, nh, N)
    return x_new, e


def _cached_iteration(x: torch.Tensor, e: torch.Tensor, model: BDHVBSubspaceDecoder, B: int, T: int, D: int, nh: int, N: int):
    x_sparse = F.relu(x @ model.encoder)  # cheap, recomputed fresh from current x -- NOT the expensive part
    x_new = _existing_compute(x, x_sparse, e, model, nh, N)
    return x_new, e  # e passed through unchanged, no re-address


def bdh_cached_evidence_forward_checkpointed(
    model: BDHVBSubspaceDecoder,
    idx: torch.Tensor,
    n_iterations: int,
    n_refresh: int,
    targets: torch.Tensor | None = None,
):
    C = model.config
    B, T = idx.size()
    D = C.n_embd
    nh = C.n_head
    N = D * C.mlp_internal_dim_multiplier // nh

    refresh_at = refresh_schedule(n_iterations, n_refresh)

    x = model.embed(idx).unsqueeze(1)
    x = model.ln(x)
    e = None
    for it in range(n_iterations):
        if it in refresh_at or e is None:
            x, e = torch.utils.checkpoint.checkpoint(_refresh_iteration, x, model, B, T, D, nh, N, use_reentrant=False)
        else:
            x, e = torch.utils.checkpoint.checkpoint(_cached_iteration, x, e, model, B, T, D, nh, N, use_reentrant=False)

    logits = x.view(B, T, D) @ model.lm_head
    loss = None
    if targets is not None:
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
    return logits, loss
