"""Captures BDH's real recurrent trajectory (x_0, x_1, ..., x_R and the
encoder ReLU activation mask at each step) for the linearizability
diagnostic (`scripts/hz0h_bdh_linearizability_diagnostic.py`).

Real motivation: BDH reuses the SAME weights every recurrent iteration,
but the transform itself is nonlinear and input-dependent -- there is no
fixed matrix A such that x_{r+1} = A x_r, so "just precompute A^n" does
not apply literally. The open, testable question is whether the
recurrence nonetheless lives on a LOW-DIMENSIONAL, approximately LINEAR
manifold -- i.e. whether a small latent operator K (fit once via closed-
form least squares, no gradient training needed) predicts not just one
step (z_{r+1} ~= K z_r) but COMPOSES correctly across multiple steps
(z_{r+2} ~= K^2 z_r, z_{r+4} ~= K^4 z_r). One-step fit alone is a weak
test; multi-step composition is the real bar, and this file exists to
measure exactly that, honestly, before any new architecture is built
around the idea.

Never modifies `reference/hz0h_bdh_torch.py` (read-only upstream
oracle). `bdh_forward_with_trajectory` at `depth=model.config.n_layer`
must reproduce the oracle's own final logits/loss EXACTLY -- proven,
not asserted, by
`tests/reference/test_hz0h_bdh_trajectory_torch.py`.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from reference.hz0h_bdh_torch import BDH


def bdh_forward_with_trajectory(
    model: BDH,
    idx: torch.Tensor,
    depth: int,
    targets: torch.Tensor | None = None,
):
    """Same real per-layer computation as `BDH.forward`
    (`reference/hz0h_bdh_variable_depth_torch.py`'s
    `bdh_variable_depth_forward`, which this mirrors exactly), but ALSO
    returns the full sequence of recurrent states and encoder ReLU
    activation masks.

    Returns `(logits, loss, x_states, relu_masks)`:
    - `x_states`: list of length `depth + 1`, `x_states[0]` is the state
      BEFORE any recurrent iteration (post embed+LN, what the diagnostic
      calls x_0), `x_states[r]` is the state after `r` iterations.
      Each tensor has shape `(B, 1, T, D)`, matching `BDH.forward`'s own
      internal `x`.
    - `relu_masks`: list of length `depth`, `relu_masks[r]` is the
      boolean `(x @ encoder) > 0` mask from iteration `r+1` (the
      encoder-path sparsity pattern for that step), shape
      `(B, nh, T, N)`.
    """
    C = model.config
    B, T = idx.size()
    D = C.n_embd

    x = model.embed(idx).unsqueeze(1)
    x = model.ln(x)

    x_states = [x]
    relu_masks = []

    for _iteration in range(depth):
        x_latent = x @ model._w(model.encoder)
        x_sparse = F.relu(x_latent)
        relu_masks.append(x_sparse > 0)

        yKV = model.attn(Q=x_sparse, K=x_sparse, V=x)
        yKV = model.ln(yKV)

        y_latent = yKV @ model._w(model.encoder_v)
        y_sparse = F.relu(y_latent)
        xy_sparse = x_sparse * y_sparse
        xy_sparse = model.drop(xy_sparse)

        nh = C.n_head
        N = D * C.mlp_internal_dim_multiplier // nh
        yMLP = xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ model._w(model.decoder)
        y = model.ln(yMLP)
        x = model.ln(x + y)
        x_states.append(x)

    logits = x.view(B, T, D) @ model.lm_head
    loss = None
    if targets is not None:
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
    return logits, loss, x_states, relu_masks
