"""HZ-BDH-GS: Grouped Synaptic State (Phase 2R-C, `plans/HZ Phase 2R
State Redesign Plan.md`).

Real, explicit divergence from upstream BDH's STREAMING behavior only --
NOT a new model architecture and NOT a retrained variant. BDH's exact
streaming form (`reference/hz0h_bdh_torch.py`'s `bdh_stream_chunk`)
keeps one persistent state tensor PER LAYER, even though
`encoder`/`encoder_v`/`decoder` are already shared/tied across every
layer -- the state is the one thing that's still per-layer, which is
exactly why it's `n_layer` times bigger than it needs to be if layers
could share.

Key structural fact this file exploits: a state only matters ACROSS
STREAMING CALLS (it summarizes strictly-earlier time steps). A single
full-sequence forward pass (`BDH.forward`, or `bdh_stream_chunk` called
once with `start_position=0` on a freshly-initialized state) has NO
prior state to share -- the intra-chunk term alone covers the whole
sequence, and the cross-chunk term is exactly zero regardless of how
many state banks exist. That means grouping state banks changes
NOTHING about a single full-sequence forward pass, training included --
an already-trained exact-BDH model can be evaluated under grouped
streaming with ZERO retraining, and the exact-BDH oracle and every
group-count variant use the IDENTICAL weights. Only genuinely
multi-call streaming (chunked/token-by-token decode) is affected.

Design: `n_groups` state banks instead of `n_layer`, contiguous
depth-block assignment (layer `l` -> group `l * n_groups // n_layer`,
matching the user's own "depth 0,1,2 -> state A; depth 3,4,5 -> state B"
example). Every layer in a chunk-processing call reads its group's state
AS OF BEFORE THIS CHUNK (not updated mid-chunk by an earlier layer in
the same group -- a deliberate, disclosed design choice for
predictability); each layer's own chunk contribution (`K_t^T V_t`,
unaffected -- no per-layer read/write projections in this first version,
see the module-level TODO) is SUMMED across every layer in the group and
applied once, after all layers in the chunk have been processed. With
`n_groups == n_layer` (one group per layer, i.e. no sharing), this is
mathematically IDENTICAL to `bdh_stream_chunk` -- verified in
`tests/reference/test_hz0h_bdh_gs_torch.py`.

Real, deliberate scope limit vs. the user's own fuller sketch: no
per-layer read/write projections (`P_l`/`O_l`) yet -- this first version
tests whether PLAIN state-bank sharing (no extra learned disambiguation)
already preserves quality on its own. If it doesn't, that's the direct,
disclosed motivation for adding per-layer projections next (real future
work, not built here).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from reference.hz0h_bdh_torch import BDH


def layer_group_assignment(n_layer: int, n_groups: int) -> list[int]:
    """Contiguous depth blocks: layer l -> group l * n_groups // n_layer.
    E.g. n_layer=6, n_groups=2 -> [0,0,0,1,1,1] (depths 0-2 share bank 0,
    depths 3-5 share bank 1), matching the user's own example diagram."""
    if not (1 <= n_groups <= n_layer):
        raise ValueError(f"n_groups must be in [1, n_layer={n_layer}], got {n_groups}")
    return [layer * n_groups // n_layer for layer in range(n_layer)]


def init_bdh_grouped_states(model: BDH, n_groups: int, batch_size: int, device=None, dtype=None) -> list[torch.Tensor]:
    c = model.config
    nh, D = c.n_head, c.n_embd
    N = D * c.mlp_internal_dim_multiplier // nh
    device = device if device is not None else model.encoder.device
    dtype = dtype if dtype is not None else model.encoder.dtype
    return [torch.zeros(batch_size, nh, N, D, device=device, dtype=dtype) for _ in range(n_groups)]


def bdh_grouped_stream_chunk(
    model: BDH, group_states: list[torch.Tensor], idx_chunk: torch.Tensor, start_position: int, n_groups: int,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Same real per-layer computation as `bdh_stream_chunk` (identical
    intra-chunk term, identical use of shared encoder/encoder_v/decoder),
    except the CROSS-chunk term reads from a state bank shared by every
    layer in the same depth group (see `layer_group_assignment`), and
    each layer's own chunk contribution is summed into that shared bank
    (applied once, after every layer in this chunk has read the
    pre-chunk value -- see the module docstring for why)."""
    c = model.config
    B, L = idx_chunk.shape
    D = c.n_embd
    nh = c.n_head
    N = D * c.mlp_internal_dim_multiplier // nh
    device = idx_chunk.device
    layer_to_group = layer_group_assignment(c.n_layer, n_groups)

    x = model.embed(idx_chunk).unsqueeze(1)
    x = model.ln(x)

    positions = torch.arange(start_position, start_position + L, device=device, dtype=model.attn.freqs.dtype).view(1, 1, L, 1)
    r_phases = positions * model.attn.freqs

    group_contributions: dict[int, torch.Tensor] = {}
    for level in range(c.n_layer):
        group = layer_to_group[level]
        x_latent = x @ model._w(model.encoder)
        x_sparse = F.relu(x_latent)
        QR = model.attn.rope(r_phases, x_sparse)
        KR = QR
        V = x

        intra = (QR @ KR.mT).tril(diagonal=-1) @ V
        prefix_state = group_states[group]
        cross = QR @ prefix_state
        yKV = intra + cross
        yKV = model.ln(yKV)

        y_latent = yKV @ model._w(model.encoder_v)
        y_sparse = F.relu(y_latent)
        xy_sparse = x_sparse * y_sparse
        xy_sparse = model.drop(xy_sparse)
        yMLP = xy_sparse.transpose(1, 2).reshape(B, 1, L, N * nh) @ model._w(model.decoder)
        y = model.ln(yMLP)
        x = model.ln(x + y)

        chunk_contribution = KR.mT @ V
        if group in group_contributions:
            group_contributions[group] = group_contributions[group] + chunk_contribution
        else:
            group_contributions[group] = chunk_contribution

    new_group_states = list(group_states)
    for group, contribution in group_contributions.items():
        new_group_states[group] = group_states[group] + contribution

    logits = x.view(B, L, D) @ model.lm_head
    return new_group_states, logits
