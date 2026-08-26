"""Activation-checkpointed variable-depth forward for
reference/hz0h_bdh_hierarchical_region_gated_torch.py, mirroring
reference/hz0h_bdh_vb_checkpointed_torch.py's own structure exactly
(same reason: avoid storing intermediate activations across the
recurrent-depth loop, needed at production scale per this project's own
prior 100M-param memory-wall history)."""
from __future__ import annotations

import torch
import torch.nn.functional as F
import torch.utils.checkpoint

from reference.hz0h_bdh_hierarchical_region_gated_torch import BDHHierarchicalRegionGated


def _region_gated_checkpoint_iteration(x: torch.Tensor, model: BDHHierarchicalRegionGated, B: int, T: int, D: int, nh: int, N: int) -> torch.Tensor:
    x_latent = x @ model._w(model.encoder)
    raw_sparse = F.relu(x_latent)
    region_gate = model._region_gate(x)
    x_sparse = raw_sparse * region_gate

    yKV = model.attn(Q=x_sparse, K=x_sparse, V=x)
    yKV = model.ln(yKV)

    y_latent = yKV @ model._w(model.encoder_v)
    y_sparse = F.relu(y_latent)
    xy_sparse = x_sparse * y_sparse
    xy_sparse = model.drop(xy_sparse)

    yMLP = xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ model._w(model.decoder)
    y = model.ln(yMLP)
    x = model.ln(x + y)
    return x


def bdh_hierarchical_region_gated_forward_checkpointed(
    model: BDHHierarchicalRegionGated,
    idx: torch.Tensor,
    n_iterations: int,
    targets: torch.Tensor | None = None,
):
    C = model.config
    B, T = idx.size()
    D = C.n_embd
    nh = C.n_head
    N = D * C.mlp_internal_dim_multiplier // nh

    x = model.embed(idx).unsqueeze(1)
    x = model.ln(x)

    for _iteration in range(n_iterations):
        x = torch.utils.checkpoint.checkpoint(
            _region_gated_checkpoint_iteration, x, model, B, T, D, nh, N, use_reentrant=False,
        )

    logits = x.view(B, T, D) @ model.lm_head
    loss = None
    if targets is not None:
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
    return logits, loss
