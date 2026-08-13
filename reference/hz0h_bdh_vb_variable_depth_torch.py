"""HZ Phase 6 / Training B, applied to the value-bottleneck arm: real
next step after `docs/restart/hz0h_phase6_depth_curriculum_results.md`
confirmed (2 seeds) that a narrow-to-wide recurrent-depth curriculum
beats fixed-depth full BPTT on BOTH quality and wall-clock time for
exact BDH at 25M-param real-text scale. HZ-Core-1's own quality
regression (`docs/restart/hz0h_core1_quality_25m_results.md`, VB
+8.24%/+9.12% relative CE vs exact BDH, confirmed at 2 seeds) is a
SEPARATE, independent axis from depth scheduling -- untested whether
the same curriculum trick that helped exact BDH's own optimization also
helps close VB's gap. This is the value-bottleneck analog of
`reference/hz0h_bdh_variable_depth_torch.py`'s `bdh_variable_depth_forward`,
needed because `BDHVB.forward` (`reference/hz0h_bdh_vb_torch.py`) hard-codes
its layer loop to `range(C.n_layer)`, the same way exact `BDH.forward`
did before Phase 5 built the variable-depth version.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from reference.hz0h_bdh_vb_torch import BDHVB


def bdh_vb_variable_depth_forward(model: BDHVB, idx: torch.Tensor, n_iterations: int, targets: torch.Tensor | None = None):
    """Same real per-layer computation as `BDHVB.forward`, but the
    number of layer-loop iterations is `n_iterations`, independent of
    `model.config.n_layer` -- reuses the SAME shared `encoder`/
    `encoder_v`/`decoder`/`P`/`O`/`attn`/`ln` weights every iteration,
    mirroring `bdh_variable_depth_forward`'s exact decoupling for exact
    BDH."""
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
        yKV_bottleneck = model.attn(Q=x_sparse, K=x_sparse, V=v_bottleneck)
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
