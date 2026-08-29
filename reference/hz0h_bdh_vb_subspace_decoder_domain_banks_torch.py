"""Domain-banked write specialization for BDHVBSubspaceDecoder --
real test of the "Developmental Neuron Specialization" idea discussed
2026-08-28: write-side-only (never touches encoder/encoder_v/Q/K, per
this project's own addressing-resists-compression finding), FIXED
domain-lookup bank selection (not a learned router -- this session's
own MoE result already showed learned routing barely engages at 25M
tokens, so testing the freeze hypothesis through a second untested
learned mechanism would confound the result), hard 0/1 selection so a
non-selected domain's bank receives exactly zero gradient for that
batch (freezing falls out of the forward computation automatically,
no separate requires_grad toggling needed).

Deliberately NOT using a near-zero conservative gate the way Phase 4
(gated residual) and Phase 7 (MoE) did. That convention exists to
protect an already-good solution from a LEARNED mechanism's noisy
early decisions. Here the "routing" is a fixed, correct-from-step-0
domain lookup, not a decision the model has to learn -- the risk a
conservative gate protects against doesn't apply, and gating the bank
near zero would specifically suppress the gradient signal this
experiment needs banks to receive in order to test whether they
specialize at all.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from reference.hz0h_bdh_vb_subspace_decoder_torch import BDHVBSubspaceDecoder


def add_domain_banks(model: BDHVBSubspaceDecoder, n_domains: int) -> None:
    C = model.config
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    r = C.subspace_rank
    model.decoder_down_banks = nn.Parameter(torch.zeros((n_domains, r, C.n_embd), device=device, dtype=dtype).normal_(std=0.02))
    model._n_domains = n_domains
    print(f"[domain_banks] n_domains={n_domains} expert_rank={r} (no gate -- see module docstring)", flush=True)


def _domain_bank_checkpoint_iteration(x: torch.Tensor, model: BDHVBSubspaceDecoder, B: int, T: int, D: int, nh: int, N: int, domain_id: int):
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
    y_bank = model.ln(alpha @ model.decoder_down_banks[domain_id])
    y = y_shared + y_bank
    x = model.ln(x + y)
    return x, xy_sparse


def _dense_checkpoint_iteration_with_support(x: torch.Tensor, model: BDHVBSubspaceDecoder, B: int, T: int, D: int, nh: int, N: int):
    """Same computation as hz0h_bdh_vb_subspace_decoder_checkpointed_torch's
    _vb_subspace_decoder_checkpoint_iteration, duplicated (not imported)
    so it can also return xy_sparse for the dense baseline's Jaccard
    measurement, without changing that shared function's return
    signature for its other callers."""
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

    alpha = torch.matmul(xy_sparse, model.decoder_up.view(nh, N, -1)).sum(dim=1, keepdim=True)
    yMLP = alpha @ model.decoder_down
    y = model.ln(yMLP)
    x = model.ln(x + y)
    return x, xy_sparse


def bdh_vb_subspace_decoder_forward_dense_with_support_checkpointed(
    model: BDHVBSubspaceDecoder,
    idx: torch.Tensor,
    n_iterations: int,
    targets: torch.Tensor | None = None,
    collect_support: bool = False,
):
    C = model.config
    B, T = idx.size()
    D = C.n_embd
    nh = C.n_head
    N = D * C.mlp_internal_dim_multiplier // nh

    x = model.embed(idx).unsqueeze(1)
    x = model.ln(x)

    last_support = None
    for iteration in range(n_iterations):
        x, xy_sparse = torch.utils.checkpoint.checkpoint(
            _dense_checkpoint_iteration_with_support, x, model, B, T, D, nh, N, use_reentrant=False,
        )
        if collect_support and iteration == n_iterations - 1:
            last_support = xy_sparse.detach()

    logits = x.view(B, T, D) @ model.lm_head
    loss = None
    if targets is not None:
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
    if collect_support:
        return logits, loss, last_support
    return logits, loss


def bdh_vb_subspace_decoder_forward_domain_banks_checkpointed(
    model: BDHVBSubspaceDecoder,
    idx: torch.Tensor,
    n_iterations: int,
    domain_id: int,
    targets: torch.Tensor | None = None,
    collect_support: bool = False,
):
    """domain_id: single int, the whole batch belongs to one domain
    phase (matches the "per document/per phase" framing, not per-token
    routing). collect_support=True also returns the last round's
    xy_sparse (for the within/across-domain Jaccard measurement) --
    off by default to avoid retaining it during normal training."""
    C = model.config
    B, T = idx.size()
    D = C.n_embd
    nh = C.n_head
    N = D * C.mlp_internal_dim_multiplier // nh

    x = model.embed(idx).unsqueeze(1)
    x = model.ln(x)

    last_support = None
    for iteration in range(n_iterations):
        x, xy_sparse = torch.utils.checkpoint.checkpoint(
            _domain_bank_checkpoint_iteration, x, model, B, T, D, nh, N, domain_id, use_reentrant=False,
        )
        if collect_support and iteration == n_iterations - 1:
            last_support = xy_sparse.detach()

    logits = x.view(B, T, D) @ model.lm_head
    loss = None
    if targets is not None:
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
    if collect_support:
        return logits, loss, last_support
    return logits, loss
