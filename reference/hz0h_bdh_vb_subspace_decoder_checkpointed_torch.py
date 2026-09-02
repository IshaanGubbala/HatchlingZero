"""Activation-checkpointed variable-depth forward for
reference/hz0h_bdh_vb_subspace_decoder_torch.py, mirroring
reference/hz0h_bdh_vb_checkpointed_torch.py's structure (same VB
bottleneck) combined with
reference/hz0h_bdh_subspace_decoder_checkpointed_torch.py's structure
(same factored decoder)."""
from __future__ import annotations

import torch
import torch.nn.functional as F
import torch.utils.checkpoint

from reference.hz0h_bdh_vb_subspace_decoder_torch import BDHVBSubspaceDecoder


def _vb_subspace_decoder_checkpoint_iteration(x: torch.Tensor, model: BDHVBSubspaceDecoder, B: int, T: int, D: int, nh: int, N: int) -> torch.Tensor:
    x_latent = x @ model.encoder
    x_sparse = F.relu(x_latent)

    v_bottleneck = x @ model.P
    yKV_bottleneck = model.attn(Q=x_sparse, K=x_sparse, V=v_bottleneck)
    yKV = yKV_bottleneck @ model.O
    yKV = model.ln(yKV)

    y_latent = yKV @ model._w("encoder_v")
    y_sparse = F.relu(y_latent)
    xy_sparse = x_sparse * y_sparse
    xy_sparse = model.drop(xy_sparse)

    # Real Phase B fix, 2026-08-26: see hz0h_bdh_vb_subspace_decoder_stream_torch.py's
    # matching comment -- avoids a non-contiguous-tensor materialization that
    # dominates cost at batch>1 (verified mathematically identical).
    alpha = torch.matmul(xy_sparse, model._w("decoder_up").view(nh, N, -1)).sum(dim=1, keepdim=True)
    yMLP = alpha @ model._w("decoder_down")
    y = model.ln(yMLP)
    x = model.ln(x + y)
    return x


def bdh_vb_subspace_decoder_forward_checkpointed(
    model: BDHVBSubspaceDecoder,
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
            _vb_subspace_decoder_checkpoint_iteration, x, model, B, T, D, nh, N, use_reentrant=False,
        )

    logits = x.view(B, T, D) @ model.lm_head
    loss = None
    if targets is not None:
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
    return logits, loss
