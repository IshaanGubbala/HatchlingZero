"""Real O(1)-state streaming decode path for
reference/hz0h_bdh_vb_subspace_decoder_torch.py's BDHVBSubspaceDecoder
(the compound VB-frozen-identity + subspace-decoder-warmstart
architecture), mirroring reference/hz0h_bdh_vb_torch.py's
bdh_vb_stream_chunk (VB's d_state-wide bottleneck state) combined with
reference/hz0h_bdh_subspace_decoder_stream_torch.py's
bdh_subspace_decoder_stream_chunk (the factored decoder). Both real
savings should show up together here: VB's state shrinks from
`(B,nh,N,D)` to `(B,nh,N,d_state)` (4x smaller at d_state=624/D=2496),
and the decoder weight itself shrinks 36.7x (same as the subspace-alone
streaming path) -- untested until now whether they both actually
materialize as decode speedup in the SAME model, or whether one
dominates/masks the other."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from reference.hz0h_bdh_vb_subspace_decoder_torch import BDHVBSubspaceDecoder
from reference.hz0h_bdh_vb_torch import init_bdh_vb_states


def bdh_vb_subspace_decoder_stream_chunk(
    model: BDHVBSubspaceDecoder, states: list[torch.Tensor], idx_chunk: torch.Tensor, start_position: int,
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
        x_latent = x @ model.encoder
        x_sparse = F.relu(x_latent)
        QR = model.attn.rope(r_phases, x_sparse)
        KR = QR
        v_bottleneck = x @ model.P

        intra = (QR @ KR.mT).tril(diagonal=-1) @ v_bottleneck
        prefix_state = states[level]
        cross = QR @ prefix_state
        yKV_bottleneck = intra + cross
        yKV = yKV_bottleneck @ model.O
        yKV = model.ln(yKV)

        y_latent = yKV @ model.encoder_v
        y_sparse = F.relu(y_latent)
        xy_sparse = x_sparse * y_sparse
        xy_sparse = model.drop(xy_sparse)

        # Real Phase B fix, 2026-08-26: the transpose+reshape below forces a
        # full non-contiguous-tensor materialization (torch.profiler showed
        # aten::clone/copy_ dominating decode time at B>1, absent at B=1 --
        # 7.4x/14.4x slower than linear at B=2/4, not the ~2x/4x a
        # well-behaved batched op should cost). Batched matmul over heads +
        # sum is mathematically identical (verified bit-exact locally) and
        # never touches a non-contiguous layout.
        alpha = torch.matmul(xy_sparse, model.decoder_up.view(nh, N, -1)).sum(dim=1, keepdim=True)
        yMLP = alpha @ model.decoder_down
        y = model.ln(yMLP)
        x = model.ln(x + y)

        chunk_contribution = KR.mT @ v_bottleneck
        new_states.append(prefix_state + chunk_contribution)

    logits = x.view(B, L, D) @ model.lm_head
    return new_states, logits


@torch.no_grad()
def bdh_vb_subspace_decoder_stream_prefill_chunked(
    model: BDHVBSubspaceDecoder,
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
    running_states = states if states is not None else init_bdh_vb_states(model, idx.shape[0], idx.device, model.encoder.dtype)
    logits_parts = []
    for offset in range(0, idx.shape[1], chunk_length):
        chunk = idx[:, offset:offset + chunk_length]
        running_states, logits = bdh_vb_subspace_decoder_stream_chunk(
            model, running_states, chunk, start_position=start_position + offset
        )
        logits_parts.append(logits)
    return running_states, torch.cat(logits_parts, dim=1)
