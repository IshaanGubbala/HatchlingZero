"""HZ Phase 6: activation checkpointing for the value-bottleneck (VB)
variable-depth forward -- the VB analog of
`reference/hz0h_bdh_checkpointed_torch.py`'s
`bdh_variable_depth_forward_checkpointed`.

Real motivation: `docs/restart/hz0h_phase_g_100m_scale_gate_pilot_results.md`
documented a hard GPU memory ceiling for BOTH exact BDH and VB D/4 at
~101M params, exactly at the curriculum's depth-2-to-4 transition.
`docs/restart/hz0h_phase_g_checkpointed_retry_results.md` confirmed
activation checkpointing completely clears this wall for exact BDH (real,
trained, full-25M-token-budget run). VB's own version of this wall
remains untested with checkpointing -- this file exists to test it, not
to assume the exact-BDH result transfers automatically (VB's per-layer
computation differs: a value-bottleneck projection `P`/`O` pair sits
inside the attention step that exact BDH doesn't have).

Deliberately a separate, opt-in extension file mirroring
`reference/hz0h_bdh_checkpointed_torch.py`'s own structure exactly, not
a modification of `reference/hz0h_bdh_vb_torch.py` or
`reference/hz0h_bdh_vb_variable_depth_torch.py`.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
import torch.utils.checkpoint

from reference.hz0h_bdh_vb_torch import BDHVB


def _bdh_vb_checkpoint_iteration(
    x: torch.Tensor,
    model: BDHVB,
    B: int,
    T: int,
    D: int,
    nh: int,
    N: int,
) -> torch.Tensor:
    """Single VB layer iteration for checkpointing -- byte-for-byte the
    same computation as one iteration of `bdh_vb_variable_depth_forward`'s
    own loop body. Passed to `torch.utils.checkpoint.checkpoint`, which
    recomputes it during backward instead of storing its intermediate
    activations."""
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

    return x


def bdh_vb_variable_depth_forward_checkpointed(
    model: BDHVB,
    idx: torch.Tensor,
    n_iterations: int,
    targets: torch.Tensor | None = None,
):
    """Same real per-layer computation as `bdh_vb_variable_depth_forward`,
    but each iteration wraps its block computation in
    `torch.utils.checkpoint.checkpoint` (`use_reentrant=False`) to avoid
    storing intermediate activations across the depth loop -- same
    real memory-for-compute trade as the exact-BDH version, unverified
    here whether it transfers with the same magnitude given VB's extra
    P/O projection inside the loop body."""
    C = model.config
    B, T = idx.size()
    D = C.n_embd
    nh = C.n_head
    N = D * C.mlp_internal_dim_multiplier // nh

    x = model.embed(idx).unsqueeze(1)
    x = model.ln(x)

    for _iteration in range(n_iterations):
        x = torch.utils.checkpoint.checkpoint(
            _bdh_vb_checkpoint_iteration,
            x,
            model,
            B,
            T,
            D,
            nh,
            N,
            use_reentrant=False,
        )

    logits = x.view(B, T, D) @ model.lm_head
    loss = None
    if targets is not None:
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
    return logits, loss
