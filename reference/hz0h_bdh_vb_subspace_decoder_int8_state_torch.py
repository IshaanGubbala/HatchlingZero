"""INT8 base+delta synaptic state (Phase 2R-E / HZ Next-Phase Plan D1,
reference/hz0h_bdh_vb_torch.py's bdh_vb_stream_chunk_int8_base_delta_state)
applied to the compound BDHVBSubspaceDecoder (VB frozen-identity state
compression + SVD-warmstarted subspace decoder). Real, previously-found
win (2c654d0/6589681, this project's own history) that never got
revisited once frozen-identity superseded trainable VB as the
state-compression story, and was never tried stacked with the subspace
decoder at all -- a genuine audit-flagged gap, not new research: same
init_bdh_vb_states_int8_base_delta (generic, only depends on
model.config.d_state) and the same base+delta amortized
quantize/dequantize logic as the existing int8 code, with only the
decoder matmul swapped for the factored decoder_up/decoder_down (same
swap bdh_vb_subspace_decoder_stream_chunk already made to
bdh_vb_stream_chunk).

Stacks three independent compressions on the SAME axis this project has
kept isolated until now: state WIDTH (VB, d_state=624, 4x), state
PRECISION (this file, INT8 base+delta, up to ~4x more on top), and
decoder RANK (subspace, r=64, 37x on the decoder specifically). None of
the three touch the same tensor, so in principle they compose
multiplicatively -- untested until this file existed.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from reference.hz0h_bdh_torch import dequantize_state_int8, quantize_state_int8
from reference.hz0h_bdh_vb_subspace_decoder_torch import BDHVBSubspaceDecoder
from reference.hz0h_bdh_vb_torch import init_bdh_vb_states_int8_base_delta  # noqa: F401 -- re-exported, generic over any BDHVB subclass


def bdh_vb_subspace_decoder_stream_chunk_int8_base_delta_state(
    model: BDHVBSubspaceDecoder, states: list[dict], idx_chunk: torch.Tensor, start_position: int, merge_every_k: int,
) -> tuple[list[dict], torch.Tensor]:
    c = model.config
    B, L = idx_chunk.shape
    D = c.n_embd
    nh = c.n_head
    N = D * c.mlp_internal_dim_multiplier // nh
    device = idx_chunk.device

    x = model.embed(idx_chunk).unsqueeze(1)
    x = model.ln(x)

    positions = torch.arange(start_position, start_position + L, device=device, dtype=model.attn.freqs.dtype).view(1, 1, L, 1)
    r_phases = positions * model.attn.freqs

    new_states = []
    for level in range(c.n_layer):
        x_latent = x @ model.encoder
        x_sparse = F.relu(x_latent)
        QR = model.attn.rope(r_phases, x_sparse)
        KR = QR
        v_bottleneck = x @ model.P

        intra = (QR @ KR.mT).tril(diagonal=-1) @ v_bottleneck
        base = dequantize_state_int8(states[level]["base_q"], states[level]["base_scale"]).to(QR.dtype)
        prefix_state = base + states[level]["delta"]
        cross = QR @ prefix_state
        yKV_bottleneck = intra + cross
        yKV = yKV_bottleneck @ model.O
        yKV = model.ln(yKV)

        y_latent = yKV @ model.encoder_v
        y_sparse = F.relu(y_latent)
        xy_sparse = x_sparse * y_sparse
        xy_sparse = model.drop(xy_sparse)

        # Real Phase B fix, 2026-08-26: see
        # reference/hz0h_bdh_vb_subspace_decoder_stream_torch.py's matching
        # comment -- avoids a non-contiguous-tensor materialization that
        # dominates cost at batch>1 (verified mathematically identical).
        alpha = torch.matmul(xy_sparse, model.decoder_up.view(nh, N, -1)).sum(dim=1, keepdim=True)
        yMLP = alpha @ model.decoder_down
        y = model.ln(yMLP)
        x = model.ln(x + y)

        chunk_contribution = KR.mT @ v_bottleneck
        new_delta = states[level]["delta"] + chunk_contribution
        new_tokens_since_merge = states[level]["tokens_since_merge"] + L

        if new_tokens_since_merge >= merge_every_k:
            merged = base + new_delta
            new_q, new_scale = quantize_state_int8(merged)
            new_states.append({"base_q": new_q, "base_scale": new_scale, "delta": torch.zeros_like(new_delta), "tokens_since_merge": 0})
        else:
            new_states.append({"base_q": states[level]["base_q"], "base_scale": states[level]["base_scale"], "delta": new_delta, "tokens_since_merge": new_tokens_since_merge})

    logits = x.view(B, L, D) @ model.lm_head
    return new_states, logits


@torch.no_grad()
def bdh_vb_subspace_decoder_stream_prefill_chunked_int8_base_delta(
    model: BDHVBSubspaceDecoder,
    idx: torch.Tensor,
    *,
    chunk_length: int,
    merge_every_k: int,
    states: list[dict] | None = None,
    start_position: int = 0,
) -> tuple[list[dict], torch.Tensor]:
    if idx.ndim != 2 or idx.shape[1] == 0:
        raise ValueError("idx must have shape (batch, non-empty sequence)")
    if chunk_length < 1:
        raise ValueError("chunk_length must be positive")
    running_states = states if states is not None else init_bdh_vb_states_int8_base_delta(model, idx.shape[0], device=idx.device)
    logits_parts = []
    for offset in range(0, idx.shape[1], chunk_length):
        chunk = idx[:, offset:offset + chunk_length]
        running_states, logits = bdh_vb_subspace_decoder_stream_chunk_int8_base_delta_state(
            model, running_states, chunk, start_position=start_position + offset, merge_every_k=merge_every_k,
        )
        logits_parts.append(logits)
    return running_states, torch.cat(logits_parts, dim=1)
