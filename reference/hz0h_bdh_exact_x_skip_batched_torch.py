"""Tier 3 item 17 of plans/HatchlingZero_BDH_Efficiency_Architecture_Plan_2026-08-24.md
(Phase S-B, second implementation goal): a real, GPU-parallel (no Python
loop over tokens) "generic PyTorch gather/scatter" implementation of the
exact x-mask skip verified correct in item 16
(reference/hz0h_bdh_exact_x_skip_torch.py's per-token loop version).

Padded-batch gather: for a flattened batch of M=B*T tokens and a given
head, each token has a different number of active x_sparse neurons.
`torch.argsort` on the boolean support mask (descending) is a stable
sort that puts every active index first in its original relative order
-- taking the first max_k columns of that per-token sorted index list
gives a real (M, max_k) padded active-index tensor with NO Python loop,
where max_k is the worst-case active count in this chunk. Padding slots
are zeroed via an explicit validity mask before they can contribute to
any sum, so the result is mathematically EXACT regardless of what
(unused) index padding happens to gather.

Deliberately built and benchmarked BEFORE any custom Triton/CUDA kernel
-- this project's own disclosed history (plan section 3, Constraint D)
shows multiple cases where a mathematically-sound sparse approach lost
on real wall-clock time (launch overhead, irregular access) despite a
clean FLOP-count win. This version tests whether plain vectorized
PyTorch gather/scatter alone gets real speedup before further
kernel-engineering investment is justified.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def bdh_round_exact_x_skip_batched(x: torch.Tensor, encoder: torch.Tensor, encoder_v: torch.Tensor,
                                    decoder: torch.Tensor, attn, ln, config) -> torch.Tensor:
    nh = config.n_head
    D = config.n_embd
    N = D * config.mlp_internal_dim_multiplier // nh
    B, _, T, _ = x.shape
    M = B * T

    x_latent = x @ encoder
    x_sparse = F.relu(x_latent)  # (B, nh, T, N)
    yKV = attn(Q=x_sparse, K=x_sparse, V=x)
    yKV = ln(yKV)  # (B, nh, T, D)

    x_sparse_flat = x_sparse.permute(1, 0, 2, 3).reshape(nh, M, N)  # (nh, M, N)
    yKV_flat = yKV.permute(1, 0, 2, 3).reshape(nh, M, D)  # (nh, M, D)
    decoder_reshaped = decoder.view(nh, N, D)  # (nh, N, D)

    yMLP_flat = torch.zeros(M, D, dtype=x.dtype, device=x.device)

    for h in range(nh):  # real, unavoidable loop is only over heads (nh=8 typical), NOT over tokens
        support_mask = x_sparse_flat[h] != 0  # (M, N)
        support_count = support_mask.sum(dim=-1)  # (M,)
        max_k = int(support_count.max().item())
        if max_k == 0:
            continue

        # Stable descending sort of the boolean mask puts every True index first,
        # in original relative order -- a real, vectorized way to build a padded
        # per-token active-index list with no Python loop over tokens.
        sorted_idx = torch.argsort(support_mask.int(), dim=-1, descending=True, stable=True)  # (M, N)
        gathered_idx = sorted_idx[:, :max_k]  # (M, max_k) -- only real active-or-padding indices
        valid = torch.arange(max_k, device=x.device).unsqueeze(0) < support_count.unsqueeze(1)  # (M, max_k)

        encoder_v_T = encoder_v[h].transpose(0, 1)  # (N, D)
        ev_gathered = encoder_v_T[gathered_idx]  # (M, max_k, D) -- ONLY the active E_v output columns, gathered
        y_latent_active = torch.einsum("md,mkd->mk", yKV_flat[h], ev_gathered)  # (M, max_k)
        y_sparse_active = F.relu(y_latent_active) * valid

        x_sparse_active = torch.gather(x_sparse_flat[h], 1, gathered_idx) * valid  # (M, max_k)
        g_active = x_sparse_active * y_sparse_active  # (M, max_k) -- zero at every padding slot, exact

        d_gathered = decoder_reshaped[h][gathered_idx]  # (M, max_k, D) -- ONLY the active decoder input rows
        yMLP_flat = yMLP_flat + torch.einsum("mk,mkd->md", g_active, d_gathered)

    yMLP = yMLP_flat.view(B, T, 1, D).permute(0, 2, 1, 3)  # (B, 1, T, D)
    y = ln(yMLP)
    return ln(x + y)
