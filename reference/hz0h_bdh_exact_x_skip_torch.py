"""Tier 3 item 16 of plans/HatchlingZero_BDH_Efficiency_Architecture_Plan_2026-08-24.md
(Phase S-B): correctness-only exact x-mask skip for the encoder_v and
decoder projections, gated behind Tier 2's real result (item 11's
oracle-packed benchmark measured 3.33x, decisively clearing the plan's
own gate for pursuing this).

BDH computes, per round (reference/hz0h_bdh_torch.py's BDH.forward):
    x_sparse = ReLU(x @ E)               # (B, nh, T, N)
    yKV = attn(Q=x_sparse, K=x_sparse, V=x)
    y_sparse = ReLU(yKV @ E_v)            # (B, nh, T, N)
    g = x_sparse * y_sparse               # (B, nh, T, N)
    yMLP = g.reshape(B,1,T,nh*N) @ D      # decoder, (nh*N, D_model)

If x_sparse[token, head, n] == 0, then g[token, head, n] == 0 regardless
of y_sparse -- so the corresponding E_v OUTPUT COLUMN and decoder INPUT
ROW never need to be computed for that token at all. This module
implements that skip literally (real gather over the active support, not
just a mask multiply that still does the full dense matmul), verified
bit/tolerance-exact against the real dense computation before any speed
work is attempted -- "correctness first, not speed" per the plan.

Deliberately per-token (Python loop over B*T), not batched/padded or
GPU-fused -- this is the plan's OWN "first implementation goal," item 17
(grouped/fused GPU implementation) is a separate, later step gated on
this one passing its correctness checks first.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def bdh_round_dense(x: torch.Tensor, encoder: torch.Tensor, encoder_v: torch.Tensor,
                     decoder: torch.Tensor, attn, ln, config) -> torch.Tensor:
    """One BDH recurrent round, byte-for-byte matching BDH.forward's inner
    loop body -- the ground truth this module's exact-skip path must
    reproduce."""
    nh = config.n_head
    D = config.n_embd
    N = D * config.mlp_internal_dim_multiplier // nh
    B, _, T, _ = x.shape

    x_latent = x @ encoder
    x_sparse = F.relu(x_latent)
    yKV = attn(Q=x_sparse, K=x_sparse, V=x)
    yKV = ln(yKV)
    y_latent = yKV @ encoder_v
    y_sparse = F.relu(y_latent)
    xy_sparse = x_sparse * y_sparse
    yMLP = xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ decoder
    y = ln(yMLP)
    return ln(x + y)


def bdh_round_exact_x_skip(x: torch.Tensor, encoder: torch.Tensor, encoder_v: torch.Tensor,
                            decoder: torch.Tensor, attn, ln, config) -> torch.Tensor:
    """Same math, same output, but y_sparse and the decoder contraction
    are computed ONLY over each token's real nonzero x_sparse support --
    a genuine gather, not a masked-dense computation. attn's own inputs
    (Q=K=x_sparse) stay dense: the skip opportunity here is specifically
    E_v's output columns and decoder's input rows, per the plan's own
    scoping (attention itself is not what section 7 targets)."""
    nh = config.n_head
    D = config.n_embd
    N = D * config.mlp_internal_dim_multiplier // nh
    B, _, T, _ = x.shape

    x_latent = x @ encoder
    x_sparse = F.relu(x_latent)  # (B, nh, T, N) -- dense, unavoidable (this IS what defines the support)
    yKV = attn(Q=x_sparse, K=x_sparse, V=x)
    yKV = ln(yKV)  # (B, nh, T, D)

    decoder_reshaped = decoder.view(nh, N, D)  # (nh, N, D) -- undo the (nh*N, D) flatten for per-head row selection
    yMLP = torch.zeros(B, 1, T, D, dtype=x.dtype, device=x.device)

    for b in range(B):
        for t in range(T):
            acc = torch.zeros(D, dtype=x.dtype, device=x.device)
            for h in range(nh):
                support = torch.nonzero(x_sparse[b, h, t], as_tuple=True)[0]  # real per-token active indices
                if support.numel() == 0:
                    continue
                ev_cols = encoder_v[h, :, support]  # (D, |support|) -- ONLY the active E_v output columns
                y_latent_active = yKV[b, h, t] @ ev_cols  # (|support|,)
                y_sparse_active = F.relu(y_latent_active)
                g_active = x_sparse[b, h, t, support] * y_sparse_active  # (|support|,)
                d_rows = decoder_reshaped[h, support, :]  # (|support|, D) -- ONLY the active decoder input rows
                acc = acc + g_active @ d_rows
            yMLP[b, 0, t] = acc

    y = ln(yMLP)
    return ln(x + y)
