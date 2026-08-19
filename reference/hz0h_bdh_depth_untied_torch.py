"""Ablates BDH's single most defining inherited choice, never tested until
now: weight tying across depth. Every prior audit in this project (width,
attention primitives) held tying fixed and questioned everything else;
this file questions tying itself.

Real background: `BDH.forward` (`reference/hz0h_bdh_torch.py`) runs the
SAME `encoder` / `encoder_v` / `decoder` matrices for all `n_layer`
recurrent iterations -- not `n_layer` distinct weight sets like a
Transformer's stacked blocks. This is upstream's own design, inherited
verbatim, and it was never once ablated by this project before now.

`DepthUntiedBDH` gives each recurrent level its OWN encoder/encoder_v/
decoder, while `embed`, `ln`, `attn` (RoPE freqs included), and `lm_head`
stay shared -- exactly the subset upstream itself never re-instantiates
per level either, so this isolates tying specifically, not sharing in
general.

Real, disclosed confound: untying multiplies encoder/encoder_v/decoder
parameter count by `depth` (tying is exactly what keeps those three
matrices at 1x regardless of `n_layer`). Two untied configurations are
provided for this reason:
  - full capacity: each level gets the SAME `mlp_internal_dim_multiplier`
    as the tied baseline -- depth x more params in these three matrices,
    an upper bound on what untying could buy if capacity were free.
  - budget-matched: each level's multiplier is divided by `depth` so the
    untied TOTAL param count across all levels approximately equals the
    tied baseline's -- isolates tying itself from the capacity confound.

`groups` generalizes both ends into one continuous knob (`C ~ (depth/groups) x P`
at fixed total params, per the closed form worked out after the first
tying result): `groups=1` collapses to the tied oracle, `groups=depth`
is fully untied (one distinct set per level). `depth // groups` adjacent
levels share one parameter set -- `W1,W1,W2,W2` for `depth=4,groups=2` --
so the curve between the two previously-tested endpoints can be swept
directly instead of only checking the extremes.

Never modifies `reference/hz0h_bdh_torch.py` (read-only upstream oracle).
`DepthUntiedBDH` with every level's weights forced identical to a shared
copy must reproduce `BDH.forward` EXACTLY -- proven, not asserted, by
`tests/reference/test_hz0h_bdh_depth_untied_torch.py`, which gates every
other claim this file can support.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from reference.hz0h_bdh_torch import Attention, BDH, BDHConfig


class DepthUntiedBDH(nn.Module):
    """Same computation as `BDH.forward`, except `encoder`/`encoder_v`/
    `decoder` are `groups` independent parameter sets instead of one tied
    set reused `depth` times. `embed`, `ln`, `attn`, `lm_head` are shared,
    matching what upstream itself shares.

    `groups` (default `depth`, i.e. fully untied -- one set per level)
    controls how many levels share each parameter set via
    `group_of(level) = level * groups // depth`, an even split that
    assigns adjacent levels to the same group (`W1,W1,W2,W2` for
    `depth=4, groups=2`). `groups=1` makes every level share ONE set,
    the same structure as the tied oracle (see
    `tests/reference/test_hz0h_bdh_depth_untied_torch.py` for the proof
    that this reproduces `BDH.forward` exactly when weights also match).

    Each group's three matrices are initialized via a fresh `BDH(config)`
    instance (borrowing the oracle's own init, `normal_(std=0.02)`
    followed by `self.apply(self._init_weights)`), so untied initialization
    statistics match the tied baseline's exactly -- only independence
    across groups differs, not the init distribution.
    """

    def __init__(self, config: BDHConfig, depth: int, groups: int | None = None):
        super().__init__()
        self.config = config
        self.depth = depth
        self.groups = depth if groups is None else groups
        assert 1 <= self.groups <= depth

        shared = BDH(config)
        self.embed = shared.embed
        self.ln = shared.ln
        self.attn = shared.attn
        self.drop = shared.drop
        self.lm_head = shared.lm_head

        self.level_encoders = nn.ParameterList()
        self.level_encoders_v = nn.ParameterList()
        self.level_decoders = nn.ParameterList()
        for _ in range(self.groups):
            group = BDH(config)
            self.level_encoders.append(group.encoder)
            self.level_encoders_v.append(group.encoder_v)
            self.level_decoders.append(group.decoder)

    def group_of(self, level: int) -> int:
        return level * self.groups // self.depth

    def tie_all_levels_to(self, encoder: torch.Tensor, encoder_v: torch.Tensor, decoder: torch.Tensor) -> None:
        """Test-only helper: force every group's weights to the SAME
        tensor values, collapsing this module back to the fully-tied
        case, so its forward can be checked bit-for-bit against the real
        oracle."""
        with torch.no_grad():
            for group in range(self.groups):
                self.level_encoders[group].copy_(encoder)
                self.level_encoders_v[group].copy_(encoder_v)
                self.level_decoders[group].copy_(decoder)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None, depth: int | None = None):
        """`depth` (default `self.depth`, the max number of levels this
        module was built with) lets a curriculum run only the FIRST
        `depth` levels' own weights -- e.g. depth=2 during an early
        curriculum stage exercises levels 0-1 only, leaving levels 2+
        untouched (no gradient) until the curriculum ramps up to them.
        Mirrors `bdh_variable_depth_forward`'s own depth-truncation
        convention for the tied oracle, applied here per-level instead."""
        C = self.config
        B, T = idx.size()
        D = C.n_embd
        depth = self.depth if depth is None else depth

        x = self.embed(idx).unsqueeze(1)
        x = self.ln(x)

        for level in range(depth):
            group = self.group_of(level)
            encoder = self.level_encoders[group]
            encoder_v = self.level_encoders_v[group]
            decoder = self.level_decoders[group]

            x_latent = x @ encoder
            x_sparse = F.relu(x_latent)

            yKV = self.attn(Q=x_sparse, K=x_sparse, V=x)
            yKV = self.ln(yKV)

            y_latent = yKV @ encoder_v
            y_sparse = F.relu(y_latent)
            xy_sparse = x_sparse * y_sparse
            xy_sparse = self.drop(xy_sparse)

            nh = C.n_head
            N = D * C.mlp_internal_dim_multiplier // nh
            yMLP = xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ decoder
            y = self.ln(yMLP)
            x = self.ln(x + y)

        logits = x.view(B, T, D) @ self.lm_head
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss


def budget_matched_multiplier(tied_multiplier: int, group_count: int) -> int:
    """Real capacity-matching rule used by the sweep: divide the tied
    baseline's multiplier by the number of independent parameter GROUPS
    (not necessarily `depth` -- pass `groups` for a partial-untying arm)
    so untied's TOTAL encoder/encoder_v/decoder param count across all
    groups is approximately equal to the tied baseline's single set.
    Floors at 1 (can't go narrower)."""
    return max(1, tied_multiplier // group_count)
