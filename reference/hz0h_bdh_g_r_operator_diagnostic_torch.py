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
ADDS a per-iteration capture of `x_sparse` (`u`) and `y_sparse` (`v`)
before dropout. Proven bit-exact (logits + loss) against
`bdh_variable_depth_forward` by
`tests/reference/test_hz0h_bdh_g_r_operator_diagnostic_torch.py`.

Captures `u` and `v` SEPARATELY, not just their product `g_r = u*v`,
because `g_r`'s own zero pattern cannot distinguish "u was zero" from
"v was zero" -- and that distinction is exactly what determines the
real, exact (not approximate) compute-skipping opportunity raised
2026-08-20: once `u = ReLU(xE)` is known, any neuron with `u_n = 0`
contributes zero to the decoder regardless of `v_n`, so `E_v` only
needs to be evaluated on `u`'s support (fraction `f_x`), and the
decoder only needs `u`'s support intersected with `v`'s (fraction
`f_xy`) -- both real, exact savings with zero approximation, not a
learned/guessed router.

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
    """Returns `(logits, loss, u_states, v_states)` where each is a list
    of length `n_iterations`, entries shaped `(B, n_head, T, N)`
    (`N = D * mult // n_head`) for that round -- `u_states[r] = x_sparse`
    (the gate known right after `E`, BEFORE attention/`E_v` even run),
    `v_states[r] = y_sparse` (the gate known after `E_v`). `g_r = u*v`
    (BEFORE dropout -- the real deterministic gate, not a
    dropout-corrupted view) can be recovered as `u_states[r] *
    v_states[r]` by the caller when needed. Caller is responsible for
    calling `model.eval()` first if dropout-free capture matters
    (matches every other diagnostic/eval forward in this project)."""
    C = model.config
    B, T = idx.size()
    D = C.n_embd
    nh = C.n_head
    N = D * C.mlp_internal_dim_multiplier // nh

    x = model.embed(idx).unsqueeze(1)
    x = model.ln(x)

    u_states, v_states = [], []
    for _iteration in range(n_iterations):
        x_latent = x @ model._w(model.encoder)
        x_sparse = F.relu(x_latent)
        u_states.append(x_sparse)

        yKV = model.attn(Q=x_sparse, K=x_sparse, V=x)
        yKV = model.ln(yKV)

        y_latent = yKV @ model._w(model.encoder_v)
        y_sparse = F.relu(y_latent)
        v_states.append(y_sparse)
        xy_sparse = x_sparse * y_sparse
        xy_sparse = model.drop(xy_sparse)

        yMLP = xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ model._w(model.decoder)
        y = model.ln(yMLP)
        x = model.ln(x + y)

    logits = x.view(B, T, D) @ model.lm_head
    loss = None
    if targets is not None:
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
    return logits, loss, u_states, v_states
