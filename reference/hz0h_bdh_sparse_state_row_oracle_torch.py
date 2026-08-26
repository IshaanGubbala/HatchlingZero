"""Tier 3 item 18 of plans/HatchlingZero_BDH_Efficiency_Architecture_Plan_2026-08-24.md
(Phase SR-A): correctness-only exact sparse-row oracle for the streaming
recurrent state, no speed claim (per the plan's own scoping -- item 17's
real GPU test decisively showed naive PyTorch gather/scatter can lose
badly on wall-clock time even when mathematically exact, so this stays
correctness-only until a real kernel investment is separately justified).

reference/hz0h_bdh_torch.py's bdh_stream_chunk computes, per layer, per
token:
    QR = KR = rope(x_sparse)              # (B, nh, L, N)
    y_read  = QR @ prefix_state            # (B, nh, L, D) -- state READ
    new_state = prefix_state + KR^T @ V    # (B, nh, N, D) -- state WRITE

`rope` operates on adjacent (2i, 2i+1) coordinate PAIRS: `rope((0,0)) ==
(0,0)` exactly (a zero pair rotates to a zero pair), but if EITHER
coordinate in a pair is nonzero pre-rotation, BOTH output coordinates
become generically nonzero post-rotation (a real, disclosed reason the
plan calls out "measure actual active post-RoPE PAIR density" instead of
assuming raw x_sparse coordinate density carries over unchanged). So the
real exact-skip unit for state rows is a PAIR of adjacent N-indices, not
a single coordinate: a pair is skippable (for both the read contraction
and the state write) iff x_sparse is exactly zero at BOTH of its two
coordinates.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def active_pair_mask(x_sparse: torch.Tensor) -> torch.Tensor:
    """x_sparse: (..., N), N even. Returns (..., N//2) bool -- True where
    EITHER coordinate in that adjacent pair is nonzero (i.e. NOT
    skippable post-rotation)."""
    pairs = x_sparse.view(*x_sparse.shape[:-1], -1, 2)
    return pairs.abs().sum(dim=-1) != 0


def bdh_stream_step_sparse_row_oracle(model, states: list, idx_token: torch.Tensor, position: int) -> tuple[list, torch.Tensor]:
    """Single-token (L=1) decode step, mathematically identical to
    bdh_stream_chunk(model, states, idx_token, position) at L=1, but the
    state READ contraction and the state WRITE both use a real gather
    over only the active pair-rows (2 rows per active pair) instead of
    the full N rows -- exact, not approximate: inactive pairs contribute
    exactly zero to both the read and the write by construction, so
    skipping them changes nothing."""
    c = model.config
    B, L = idx_token.shape
    assert L == 1, "this oracle is scoped to single-token decode steps"
    D = c.n_embd
    nh = c.n_head
    N = D * c.mlp_internal_dim_multiplier // nh
    device = idx_token.device

    x = model.ln(model.embed(idx_token).unsqueeze(1))
    r_phases = torch.tensor([[position]], device=device, dtype=model.attn.freqs.dtype).view(1, 1, 1, 1) * model.attn.freqs

    new_states = []
    for level in range(c.n_layer):
        x_latent = x @ model._w(model.encoder)
        x_sparse = F.relu(x_latent)  # (B, nh, 1, N)
        QR = model.attn.rope(r_phases, x_sparse)  # (B, nh, 1, N)
        KR = QR
        V = x  # (B, 1, 1, D)

        prefix_state = states[level]  # (B, nh, N, D)

        pair_active = active_pair_mask(x_sparse)  # (B, nh, 1, N//2) -- per-token, per-head
        # This oracle processes one (B=1-style) config at a time for the gather to stay simple and
        # obviously-correct; real batching is a separate, later concern (this is Tier 3 item 18's
        # own "no speed claim" scope, same as item 16's per-token loop precedent).
        yKV = torch.zeros(B, nh, 1, D, dtype=x.dtype, device=device)
        write_state = prefix_state.clone()
        for b in range(B):
            for h in range(nh):
                pair_idx = torch.nonzero(pair_active[b, h, 0], as_tuple=True)[0]  # active pair indices
                if pair_idx.numel() == 0:
                    continue
                row_idx = torch.stack([pair_idx * 2, pair_idx * 2 + 1], dim=-1).reshape(-1)  # active N-row indices (2 per pair)
                row_idx = row_idx.sort().values

                QR_active = QR[b, h, 0, row_idx]  # (|active_rows|,)
                state_active = prefix_state[b, h, row_idx, :]  # (|active_rows|, D)
                yKV[b, h, 0] = QR_active @ state_active  # exact read, only active rows contribute (rest are truly zero)

                KR_active = KR[b, h, 0, row_idx]  # (|active_rows|,)
                write_state[b, h, row_idx, :] = prefix_state[b, h, row_idx, :] + torch.outer(KR_active, V[b, 0, 0])
                # rows NOT in row_idx are untouched -- exact, since their KR contribution is provably zero

        yKV = model.ln(yKV)
        y_latent = yKV @ model._w(model.encoder_v)
        y_sparse = F.relu(y_latent)
        xy_sparse = x_sparse * y_sparse
        yMLP = xy_sparse.transpose(1, 2).reshape(B, 1, 1, N * nh) @ model._w(model.decoder)
        y = model.ln(yMLP)
        x = model.ln(x + y)

        new_states.append(write_state)

    logits = x.view(B, 1, D) @ model.lm_head
    return new_states, logits
