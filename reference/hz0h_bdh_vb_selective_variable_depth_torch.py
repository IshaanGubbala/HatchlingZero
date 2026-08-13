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


def bdh_vb_selective_variable_depth_forward(model: BDHVBSelective, idx: torch.Tensor, n_iterations: int, targets: torch.Tensor | None = None, gate_reg_lambda: float = 0.0):
    """gate_reg_lambda > 0 adds a gate-decisiveness regularizer,
    -lambda * mean((gate - 0.5)^2) averaged over iterations, to the LM
    loss. Rewards gate values far from the uninformative 0.5 midpoint
    (opposite sign from a collapse-prevention penalty) -- added per
    docs/restart/hz0h_phase_b2_selective_write_results.md Update 5,
    which found the gate's per-token selectivity measurably erodes
    toward 0.5 as iteration depth increases, plausibly explaining why
    Phase B2's advantage doesn't carry through the curriculum's
    depth-8-only final quarter. Default 0.0 preserves prior behavior
    exactly.
    """
    C = model.config
    B, T = idx.size()
    D = C.n_embd
    nh = C.n_head
    N = D * C.mlp_internal_dim_multiplier // nh

    x = model.embed(idx).unsqueeze(1)
    x = model.ln(x)

    gate_reg_terms = [] if gate_reg_lambda > 0.0 else None

    for _iteration in range(n_iterations):
        x_latent = x @ model.encoder
        x_sparse = F.relu(x_latent)

        v_bottleneck = x @ model.P
        gate = compute_write_gate(model, x)
        v_bottleneck_gated = gate * v_bottleneck

        if gate_reg_terms is not None:
            gate_reg_terms.append(((gate - 0.5) ** 2).mean())

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
        if gate_reg_terms is not None:
            loss = loss - gate_reg_lambda * torch.stack(gate_reg_terms).mean()
    return logits, loss
