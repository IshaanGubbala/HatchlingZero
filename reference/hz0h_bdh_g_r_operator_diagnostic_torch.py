"""Instruments the realized per-round operator-gate vector

    g_r = x_sparse_r (elementwise*) y_sparse_r

that `reference/hz0h_bdh_torch.py`'s own forward pass already computes
internally (as `xy_sparse`, pre-dropout) but never exposes. `g_r` is the
thing that actually reaches the decoder every recurrent round -- BDH's
per-token update is algebraically `x' ~= x + x @ (E @ diag(g_r) @ D)`,
so `g_r` is the realized coefficient vector selecting which of the
`N`-per-head rank-1 operators `e_n d_n^T` are active THIS token, THIS
round. Every prior optimization attempt in this project (width cuts,
FactorizedBDH, dynamic routing, the jump operator) either preserved this
object (kept quality, kept the compute) or reduced/approximated it
indirectly (bought speed, lost quality) -- this file exists to measure
whether `g_r`'s REALIZED support/rank is much smaller than its nominal
`N`-per-head width, which would be the first real evidence that there is
compute to remove without touching the object that seems to make BDH's
quality-per-parameter high (see Part 9 of
`docs/restart/hz0h_inherited_choices_audit_results.md`).

Same real per-layer computation as `bdh_variable_depth_forward` (which
is itself identical to `BDH.forward`, decoupled from
`model.config.n_layer`) -- zero change to BDH's math, this file only
ADDS a per-iteration capture of `xy_sparse` before dropout. Proven
bit-exact (logits + loss) against `bdh_variable_depth_forward` by
`tests/reference/test_hz0h_bdh_g_r_operator_diagnostic_torch.py`.

Never modifies `reference/hz0h_bdh_torch.py` or
`reference/hz0h_bdh_variable_depth_torch.py`.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from reference.hz0h_bdh_torch import BDH


def bdh_forward_with_g_r(
    model: BDH,
    idx: torch.Tensor,
    n_iterations: int,
    targets: torch.Tensor | None = None,
):
    """Returns `(logits, loss, g_states)` where `g_states` is a list of
    length `n_iterations`, each entry the real `xy_sparse` tensor
    (shape `(B, n_head, T, N)`, `N = D * mult // n_head`) for that
    round, captured BEFORE `model.drop` -- the real deterministic gate
    values, not a dropout-corrupted view of them. Caller is responsible
    for calling `model.eval()` first if dropout-free capture matters
    (matches every other diagnostic/eval forward in this project)."""
    C = model.config
    B, T = idx.size()
    D = C.n_embd
    nh = C.n_head
    N = D * C.mlp_internal_dim_multiplier // nh

    x = model.embed(idx).unsqueeze(1)
    x = model.ln(x)

    g_states = []
    for _iteration in range(n_iterations):
        x_latent = x @ model._w(model.encoder)
        x_sparse = F.relu(x_latent)

        yKV = model.attn(Q=x_sparse, K=x_sparse, V=x)
        yKV = model.ln(yKV)

        y_latent = yKV @ model._w(model.encoder_v)
        y_sparse = F.relu(y_latent)
        xy_sparse = x_sparse * y_sparse
        g_states.append(xy_sparse)
        xy_sparse = model.drop(xy_sparse)

        yMLP = xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ model._w(model.decoder)
        y = model.ln(yMLP)
        x = model.ln(x + y)

    logits = x.view(B, T, D) @ model.lm_head
    loss = None
    if targets is not None:
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
    return logits, loss, g_states
