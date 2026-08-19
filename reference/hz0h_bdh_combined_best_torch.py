"""Stacks every confirmed positive finding from the inherited-choices
audit (`docs/restart/hz0h_inherited_choices_audit_results.md`) into one
recipe, so it can be benchmarked as a whole against raw BDH and the
matched Transformer -- not just each finding in isolation:

- `mult=16` instead of canonical 32 (Part 2: ~2x FLOP/param cut for a
  small, consistent quality cost).
- `softmax_scaled` attention instead of upstream's raw unnormalized
  scores (Part 3: mean -0.0294 across 3 seeds, a free quality win).
- Weight tying kept EXACTLY as upstream (Part 4: confirmed load-bearing,
  the largest negative effect found in the whole audit if removed --
  this recipe does NOT untie).
- A trained jump operator standing in for some real recurrent
  iterations (Part 6: `real_prefix=4` real iterations + 2 jump calls
  reached 1.9x real wall-clock speedup for +0.029 loss in the
  single-model prototype).

`combined_bdh_forward_with_trajectory` mirrors
`reference/hz0h_bdh_trajectory_torch.py` exactly except the attention
term uses `softmax_scaled` (from
`reference/hz0h_bdh_primitive_ablations_torch.py`'s own tested
formula) instead of upstream's raw scores -- this is what the jump
operator is distilled against for THIS recipe specifically, since its
teacher's trajectories differ from plain BDH's once attention changes.
`combined_bdh_forward` uses that same real-iteration path for a prefix,
then a trained `JumpOperator` for the rest, exactly like
`reference/hz0h_bdh_jump_operator_torch.py`'s `jump_bdh_forward`.

Never modifies `reference/hz0h_bdh_torch.py`. At `num_jumps=0`,
`combined_bdh_forward` must reproduce
`reference/hz0h_bdh_primitive_ablations_torch.py`'s own
`ablated_bdh_forward(..., use_softmax=True, scale_scores=True)`
EXACTLY -- proven, not asserted, by
`tests/reference/test_hz0h_bdh_combined_best_torch.py`.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from reference.hz0h_bdh_jump_operator_torch import JumpOperator
from reference.hz0h_bdh_torch import Attention, BDH


def _softmax_scaled_attention(model: BDH, x: torch.Tensor, x_sparse: torch.Tensor, r_phases: torch.Tensor, T: int) -> torch.Tensor:
    """Exactly Part 3's `softmax_scaled` arm from
    `reference/hz0h_bdh_primitive_ablations_torch.py`: RoPE'd scores,
    scaled by `1/sqrt(latent_width)`, masked to the same STRICTLY
    lower-triangular pattern upstream uses (a token cannot attend to
    itself -- Part 3 also confirmed this masking choice is load-bearing,
    so it is preserved here unchanged), softmaxed, with the fully-masked
    first row's NaN replaced by exact zero to match upstream's own
    unnormalized-path behaviour there."""
    C = model.config
    nh = C.n_head
    D = C.n_embd
    latent_width = D * C.mlp_internal_dim_multiplier // nh
    scale = 1.0 / math.sqrt(latent_width)

    QR = Attention.rope(r_phases, x_sparse)
    scores = (QR @ QR.mT) * scale
    allowed = torch.ones(T, T, dtype=torch.bool, device=scores.device).tril(diagonal=-1)
    scores = scores.masked_fill(~allowed, float("-inf"))
    scores = torch.softmax(scores, dim=-1)
    scores = torch.nan_to_num(scores, nan=0.0)
    return scores @ x


def combined_bdh_forward_with_trajectory(model: BDH, idx: torch.Tensor, depth: int):
    """Same shape of return as `bdh_forward_with_trajectory`
    (`x_states` list of length `depth + 1`), but every recurrent
    iteration uses `softmax_scaled` attention instead of upstream's raw
    scores. Used both to train the combined recipe's own BDH teacher
    and to distill a jump operator against that teacher's real
    trajectories."""
    C = model.config
    B, T = idx.size()
    D = C.n_embd

    freqs = model.attn.freqs
    r_phases = torch.arange(0, T, device=freqs.device, dtype=freqs.dtype).view(1, 1, -1, 1) * freqs

    x = model.ln(model.embed(idx).unsqueeze(1))
    x_states = [x]

    for _iteration in range(depth):
        x_sparse = F.relu(x @ model._w(model.encoder))
        yKV = model.ln(_softmax_scaled_attention(model, x, x_sparse, r_phases, T))

        y_sparse = F.relu(yKV @ model._w(model.encoder_v))
        xy_sparse = model.drop(x_sparse * y_sparse)

        nh = C.n_head
        N = D * C.mlp_internal_dim_multiplier // nh
        yMLP = xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ model._w(model.decoder)
        x = model.ln(x + model.ln(yMLP))
        x_states.append(x)

    return x_states


def combined_bdh_forward(
    model: BDH,
    jump: JumpOperator | None,
    idx: torch.Tensor,
    real_prefix_iterations: int,
    num_jumps: int,
    targets: torch.Tensor | None = None,
):
    """The full combined recipe: `real_prefix_iterations` real
    `softmax_scaled` iterations, then `jump` applied `num_jumps` times.
    At `num_jumps=0` this is exactly
    `combined_bdh_forward_with_trajectory`'s final state (and, in turn,
    exactly `ablated_bdh_forward(..., use_softmax=True,
    scale_scores=True)`)."""
    C = model.config
    B, T = idx.size()
    D = C.n_embd

    x_states = combined_bdh_forward_with_trajectory(model, idx, real_prefix_iterations)
    x = x_states[-1]

    for _ in range(num_jumps):
        x = jump(x)

    logits = x.view(B, T, D) @ model.lm_head
    loss = None
    if targets is not None:
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
    return logits, loss
