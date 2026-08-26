"""Real O(1)-state streaming decode path for BDHSubspaceDecoder
(reference/hz0h_bdh_subspace_decoder_torch.py), mirroring
reference/hz0h_bdh_torch.py's bdh_stream_chunk/bdh_stream_prefill_chunked
exactly -- same recurrent-state math, only the decoder matmul differs
(two small dense GEMMs through decoder_up/decoder_down instead of one
big dense GEMM through decoder). init_bdh_states is reused unchanged
since state shape only depends on (n_head, N, n_embd), not the decoder.

This is what actually lets Tier 4 item 23's warm-started subspace
decoder be judged on the plan's real goal (decode throughput), not just
training wall-clock -- the decoder weight is read from HBM every round
in the streaming decode path (same weight-tied module reused
`n_layer` times per generated token), so shrinking it from
`nh*N*D*2bytes` (~199MB at production shape) to
`(nh*N*r + r*D)*2bytes` (~5.4MB at r=64) is a real per-round HBM
traffic cut, not just a training-compute cut.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from reference.hz0h_bdh_subspace_decoder_torch import BDHSubspaceDecoder


def bdh_subspace_decoder_stream_chunk(
    model: BDHSubspaceDecoder, states: list[torch.Tensor], idx_chunk: torch.Tensor, start_position: int,
) -> tuple[list[torch.Tensor], torch.Tensor]:
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
        x_latent = x @ model._w(model.encoder)
        x_sparse = F.relu(x_latent)
        QR = model.attn.rope(r_phases, x_sparse)
        KR = QR
        V = x

        intra = (QR @ KR.mT).tril(diagonal=-1) @ V
        prefix_state = states[level]
        cross = QR @ prefix_state
        yKV = intra + cross
        yKV = model.ln(yKV)

        y_latent = yKV @ model._w(model.encoder_v)
        y_sparse = F.relu(y_latent)
        xy_sparse = x_sparse * y_sparse
        xy_sparse = model.drop(xy_sparse)

        # Real Phase B fix, 2026-08-26: see
        # reference/hz0h_bdh_vb_subspace_decoder_stream_torch.py's matching
        # comment -- avoids a non-contiguous-tensor materialization that
        # dominates cost at batch>1 (verified mathematically identical).
        alpha = torch.matmul(xy_sparse, model._w(model.decoder_up).view(nh, N, -1)).sum(dim=1, keepdim=True)
        yMLP = alpha @ model._w(model.decoder_down)
        y = model.ln(yMLP)
        x = model.ln(x + y)

        chunk_contribution = KR.mT @ V
        new_states.append(prefix_state + chunk_contribution)

    logits = x.view(B, L, D) @ model.lm_head
    return new_states, logits


@torch.no_grad()
def bdh_subspace_decoder_stream_prefill_chunked(
    model: BDHSubspaceDecoder,
    idx: torch.Tensor,
    *,
    chunk_length: int,
    states: list[torch.Tensor] | None = None,
    start_position: int = 0,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    if idx.ndim != 2 or idx.shape[1] == 0:
        raise ValueError("idx must have shape (batch, non-empty sequence)")
    if chunk_length < 1:
        raise ValueError("chunk_length must be positive")
    from reference.hz0h_bdh_torch import init_bdh_states
    running_states = states if states is not None else init_bdh_states(model, idx.shape[0], idx.device, model.encoder.dtype)
    logits_parts = []
    for offset in range(0, idx.shape[1], chunk_length):
        chunk = idx[:, offset:offset + chunk_length]
        running_states, logits = bdh_subspace_decoder_stream_chunk(
            model, running_states, chunk, start_position=start_position + offset
        )
        logits_parts.append(logits)
    return running_states, torch.cat(logits_parts, dim=1)
