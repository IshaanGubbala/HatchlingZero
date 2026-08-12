"""HZ Phase 5 (`plans/HatchlingZero_Reality_Plan.md`): Shared-Depth
Adaptive Reasoning -- first real test of the core premise before
building any adaptive halting controller.

BDH's `encoder`/`encoder_v`/`decoder` are the literal SAME `nn.Parameter`
objects reused every iteration of `BDH.forward`'s layer loop (`for level
in range(C.n_layer): x_latent = x @ self.encoder; ...` -- `self.encoder`
is not indexed by `level`, confirmed directly in
`reference/hz0h_bdh_torch.py`). This means the loop bound `n_layer` is
purely a COMPUTE choice, not a parameter-count choice -- running the
identical trained weights for MORE iterations costs zero extra
parameters, only extra FLOPs. The plan's own framing: "more internal
computation != more parameters."

This file provides `bdh_variable_depth_forward`, identical to
`BDH.forward` except the iteration count is an explicit argument,
decoupled from `model.config.n_layer` -- so a model trained at one depth
can be evaluated at a DIFFERENT depth using the exact same weights, to
test whether extra (or fewer) iterations actually help before building
any halting/routing machinery on top.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from reference.hz0h_bdh_torch import BDH


def bdh_variable_depth_forward(model: BDH, idx: torch.Tensor, n_iterations: int, targets: torch.Tensor | None = None):
    """Same real per-layer computation as `BDH.forward`, but the number
    of layer-loop iterations is `n_iterations`, independent of
    `model.config.n_layer` -- reuses the SAME shared
    `encoder`/`encoder_v`/`decoder`/`attn`/`ln` weights every iteration,
    exactly as the real trained model already does internally, just for
    a different number of iterations than it may have been trained
    with."""
    C = model.config
    B, T = idx.size()
    D = C.n_embd

    x = model.embed(idx).unsqueeze(1)
    x = model.ln(x)

    for _iteration in range(n_iterations):
        x_latent = x @ model._w(model.encoder)
        x_sparse = F.relu(x_latent)

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

    logits = x.view(B, T, D) @ model.lm_head
    loss = None
    if targets is not None:
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
    return logits, loss
