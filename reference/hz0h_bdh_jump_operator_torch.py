"""A learned jump operator that stands in for TWO real BDH recurrent
iterations, motivated by Part 5's linearizability diagnostic
(`docs/restart/hz0h_inherited_choices_audit_results.md`): a closed-form
linear operator already composed almost exactly at k=2 (gap <1% between
K^2 and a directly-fit A_2). This file asks whether a small TRAINED
(not closed-form) operator does even better, since gradient descent is
not restricted to a single global linear map.

`JumpOperator` is a small residual network: `J(x) = x + mlp(x)`,
operating on BDH's own `(B, 1, T, D)` state -- same shape the real
recurrence produces, so it can be dropped into a forward pass wherever
two real iterations would otherwise run. Residual form is deliberate:
Part 5 found BDH's real updates shrink and settle with depth (relative
delta norm falls from 0.65 to 0.10 across 8 steps), so predicting the
identity-plus-small-correction is the natural parameterization, not
predicting the whole next state from scratch.

Never modifies `reference/hz0h_bdh_torch.py` (read-only upstream
oracle). `jump_bdh_forward` at `real_prefix_iterations=depth,
jumps=0` (i.e. using the jump operator zero times) must reproduce
`bdh_variable_depth_forward`'s own output exactly -- proven by
`tests/reference/test_hz0h_bdh_jump_operator_torch.py`.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from reference.hz0h_bdh_torch import BDH


class JumpOperator(nn.Module):
    """`J(x) = x + mlp(x)`, a small residual correction standing in for
    `jump_size` real BDH recurrent iterations. `hidden_mult` sets the
    MLP's hidden width as a multiple of `D` -- deliberately small (this
    operator's entire point is being far cheaper than the real
    `mlp_internal_dim_multiplier`-wide recurrent transform it replaces).
    """

    def __init__(self, d_model: int, hidden_mult: int = 4, jump_size: int = 2):
        super().__init__()
        self.jump_size = jump_size
        hidden = d_model * hidden_mult
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


def jump_bdh_forward(
    model: BDH,
    jump: JumpOperator | None,
    idx: torch.Tensor,
    real_prefix_iterations: int,
    num_jumps: int,
    targets: torch.Tensor | None = None,
):
    """Runs `real_prefix_iterations` genuine BDH recurrent iterations
    (the model's own shared encoder/encoder_v/decoder/attn, exactly as
    `bdh_variable_depth_forward`), then applies `jump` `num_jumps`
    times. Total depth-equivalent reached is
    `real_prefix_iterations + num_jumps * jump.jump_size`.

    At `num_jumps=0` this is EXACTLY `bdh_variable_depth_forward` at
    `real_prefix_iterations` -- `jump` may be `None` in that case, since
    it is never called.
    """
    C = model.config
    B, T = idx.size()
    D = C.n_embd

    x = model.embed(idx).unsqueeze(1)
    x = model.ln(x)

    for _iteration in range(real_prefix_iterations):
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

    for _ in range(num_jumps):
        x = jump(x)

    logits = x.view(B, T, D) @ model.lm_head
    loss = None
    if targets is not None:
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
    return logits, loss
