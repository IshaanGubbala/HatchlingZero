"""Activation-checkpointed variable-depth forward for
reference/hz0h_bdh_subspace_decoder_torch.py, mirroring
reference/hz0h_bdh_vb_checkpointed_torch.py's structure."""
from __future__ import annotations

import torch
import torch.nn.functional as F
import torch.utils.checkpoint

from reference.hz0h_bdh_subspace_decoder_torch import BDHSubspaceDecoder


def _subspace_decoder_checkpoint_iteration(x: torch.Tensor, model: BDHSubspaceDecoder, B: int, T: int, D: int, nh: int, N: int) -> torch.Tensor:
    x_latent = x @ model._w(model.encoder)
    x_sparse = F.relu(x_latent)

    yKV = model.attn(Q=x_sparse, K=x_sparse, V=x)
    yKV = model.ln(yKV)

    y_latent = yKV @ model._w(model.encoder_v)
    y_sparse = F.relu(y_latent)
    xy_sparse = x_sparse * y_sparse
    xy_sparse = model.drop(xy_sparse)

    xy_flat = xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh)
    alpha = xy_flat @ model._w(model.decoder_up)
    yMLP = alpha @ model._w(model.decoder_down)
    y = model.ln(yMLP)
    x = model.ln(x + y)
    return x


def bdh_subspace_decoder_forward_checkpointed(
    model: BDHSubspaceDecoder,
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
            _subspace_decoder_checkpoint_iteration, x, model, B, T, D, nh, N, use_reentrant=False,
        )

    logits = x.view(B, T, D) @ model.lm_head
    loss = None
    if targets is not None:
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
    return logits, loss
