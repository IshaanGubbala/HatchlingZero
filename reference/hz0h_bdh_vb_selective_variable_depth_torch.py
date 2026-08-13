"""HZ Phase B2, curriculum-compatible: the value-bottleneck-with-
selective-write-gate analog of `reference/hz0h_bdh_vb_variable_depth_torch.py`,
needed because `BDHVBSelective.forward` hard-codes its layer loop to
`range(C.n_layer)` the same way every other BDH variant did before its
own variable-depth version existed. Required to train Phase B2 with the
now-locked recurrent-depth curriculum
(`docs/restart/hz0h_phase6_depth_curriculum_results.md`).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from reference.hz0h_bdh_vb_selective_torch import BDHVBSelective, compute_write_gate


def bdh_vb_selective_variable_depth_forward(model: BDHVBSelective, idx: torch.Tensor, n_iterations: int, targets: torch.Tensor | None = None):
    C = model.config
    B, T = idx.size()
    D = C.n_embd
    nh = C.n_head
    N = D * C.mlp_internal_dim_multiplier // nh

    x = model.embed(idx).unsqueeze(1)
    x = model.ln(x)

    for _iteration in range(n_iterations):
        x_latent = x @ model.encoder
        x_sparse = F.relu(x_latent)

        v_bottleneck = x @ model.P
        gate = compute_write_gate(model, x)
        v_bottleneck_gated = gate * v_bottleneck

        yKV_bottleneck = model.attn(Q=x_sparse, K=x_sparse, V=v_bottleneck_gated)
        yKV = yKV_bottleneck @ model.O
        yKV = model.ln(yKV)

        y_latent = yKV @ model.encoder_v
        y_sparse = F.relu(y_latent)
        xy_sparse = x_sparse * y_sparse
        xy_sparse = model.drop(xy_sparse)

        yMLP = xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ model.decoder
        y = model.ln(yMLP)
        x = model.ln(x + y)

    logits = x.view(B, T, D) @ model.lm_head
    loss = None
    if targets is not None:
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
    return logits, loss
