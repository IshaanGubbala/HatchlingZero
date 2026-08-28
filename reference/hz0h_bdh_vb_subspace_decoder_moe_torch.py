"""Value/output MoE for BDHVBSubspaceDecoder, Phase 7 of
plans/HatchlingZero_Qwen_Integration_Plan_2026-08-26.md#11.

Explicitly NOT routing the addressing side (encoder/encoder_v/Q/K/
neuron activation blocks stay dense) -- per this project's own
repeatedly-confirmed principle ([[20.1]]): every addressing-side
compression/routing attempt has failed (neuron reordering, CertiGate,
activation templates, K-means, Key-State Subspace), every value/output-
side compression attempt has worked (VB, subspace decoder, compound,
and now the Phase-4 gated residual). This routes ONLY the final
decoder/output step.

Design directly reuses the Phase-4 gated-residual lesson (conservative
init beats perturb-from-step-0 every time this session: Muon/MTP/
n-gram all perturbed and lost, gated residual started near the known-
good solution and won): the SHARED expert IS the existing, warmstart-
compatible decoder_up/decoder_down pair (weight 1.0, unchanged
behavior), and the NEW routed experts are gated by a single scalar
g_moe starting near zero, so at initialization this reproduces the
plain compound model almost exactly.

`decoder_up` (D -> r, shared across the shared expert AND every routed
expert -- reusing the existing "compressed representation" bottleneck
directly matches the plan's own `g_t -> compressed representation ->
top-k output experts` framing) is unchanged; only the r -> D projection
becomes a mixture (shared decoder_down + top-k of `n_experts` new
decoder_down_experts, expert rank = subspace_rank, already in the
plan's stated 32-64 band).

Honest scope note: this first implementation computes the routing
weight combination via a dense einsum over ALL experts (weighted by a
mostly-zero top-k mask) rather than a real sparse gather/dispatch --
correct for a QUALITY test (matches every expert's real mathematical
contribution) but does NOT yet demonstrate the FLOP/wall-clock savings
real MoE promises. That's an explicitly later, systems-only concern
per the plan's own Tier A/B discipline -- only worth building if the
quality signal here is positive first.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from reference.hz0h_bdh_vb_subspace_decoder_torch import BDHVBSubspaceDecoder


def add_moe_decoder(model: BDHVBSubspaceDecoder, n_experts: int = 8, top_k: int = 2, g_moe_init: float = 0.01) -> None:
    C = model.config
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    r = C.subspace_rank
    model.decoder_down_experts = nn.Parameter(torch.zeros((n_experts, r, C.n_embd), device=device, dtype=dtype).normal_(std=0.02))
    model.moe_router = nn.Parameter(torch.zeros((r, n_experts), device=device, dtype=dtype).normal_(std=0.02))
    model.g_moe = nn.Parameter(torch.tensor(g_moe_init, device=device, dtype=torch.float32))
    model._moe_n_experts = n_experts
    model._moe_top_k = top_k
    print(f"[moe] n_experts={n_experts} top_k={top_k} expert_rank={r} g_moe_init={g_moe_init}", flush=True)


def _moe_checkpoint_iteration(x: torch.Tensor, model: BDHVBSubspaceDecoder, B: int, T: int, D: int, nh: int, N: int):
    x_latent = x @ model.encoder
    x_sparse = F.relu(x_latent)

    v_bottleneck = x @ model.P
    yKV_bottleneck = model.attn(Q=x_sparse, K=x_sparse, V=v_bottleneck)
    yKV = yKV_bottleneck @ model.O
    yKV = model.ln(yKV)

    y_latent = yKV @ model.encoder_v
    y_sparse = F.relu(y_latent)
    xy_sparse = x_sparse * y_sparse
    xy_sparse = model.drop(xy_sparse)

    alpha = torch.matmul(xy_sparse, model.decoder_up.view(nh, N, -1)).sum(dim=1, keepdim=True)  # (B,1,T,r)

    y_shared = model.ln(alpha @ model.decoder_down)

    router_logits = alpha @ model.moe_router  # (B,1,T,E)
    E, k = model._moe_n_experts, model._moe_top_k
    topk_vals, topk_idx = router_logits.topk(k, dim=-1)  # (B,1,T,k)
    topk_weights = F.softmax(topk_vals, dim=-1)
    dense_weights = torch.zeros_like(router_logits).scatter(-1, topk_idx, topk_weights)  # (B,1,T,E), mostly zero
    y_routed = torch.einsum("bhte,erd->bhtd", dense_weights, model.decoder_down_experts)
    y_routed = model.ln(y_routed)

    y = y_shared + model.g_moe * y_routed
    x = model.ln(x + y)

    full_probs = F.softmax(router_logits, dim=-1)  # (B,1,T,E), for the load-balancing aux loss
    f_i = (dense_weights.float() > 0).float().mean(dim=(0, 1, 2))
    p_i = full_probs.float().mean(dim=(0, 1, 2))
    return x, f_i, p_i


def bdh_vb_subspace_decoder_forward_moe_checkpointed(
    model: BDHVBSubspaceDecoder,
    idx: torch.Tensor,
    n_iterations: int,
    targets: torch.Tensor | None = None,
    aux_loss_coef: float = 0.01,
):
    C = model.config
    B, T = idx.size()
    D = C.n_embd
    nh = C.n_head
    N = D * C.mlp_internal_dim_multiplier // nh

    x = model.embed(idx).unsqueeze(1)
    x = model.ln(x)

    aux_losses = []
    for _iteration in range(n_iterations):
        x, f_i, p_i = torch.utils.checkpoint.checkpoint(
            _moe_checkpoint_iteration, x, model, B, T, D, nh, N, use_reentrant=False,
        )
        aux_losses.append(model._moe_n_experts * (f_i * p_i).sum())

    logits = x.view(B, T, D) @ model.lm_head
    loss = None
    if targets is not None:
        ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        aux_loss = torch.stack(aux_losses).mean()
        loss = ce_loss + aux_loss_coef * aux_loss
    return logits, loss
