"""Single-token BDH streaming decode step, using the packed encoder
layout (reference/hz0h_bdh_packed_encoder_torch.py's PackedEncoderBDH,
model.encoder_packed natively (D, nh*N)) instead of the oracle's
broadcast `x @ model._w(model.encoder)`.

Direct test of whether this session's training-side packed-encoder win
(+4.19% full training step) does anything for decode. Real reason to be
skeptical going in, not assumed either way: the manual-CUDA-graph test
(reference/hz0h_bdh_stream_decode_step_graph_safe_torch.py's own results,
scripts/hz0h_inference_bdh_manual_cudagraph_test.py) proved decode's
bottleneck is reading/writing the ~1.6GB per-layer state every token, not
the encoder weight (~200MB, much smaller) -- so this weight-layout change
touches a comparatively small piece of the real cost. Built and measured
anyway rather than assumed, same discipline as every other claim in this
project.

Identical structure to
hz0h_bdh_stream_decode_step_graph_safe_torch.py's in-place variant,
swapping only the encoder projection's execution path -- everything else
(encoder_v, decoder, attention, state update) is unchanged.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from reference.hz0h_bdh_packed_encoder_torch import PackedEncoderBDH
from reference.hz0h_bdh_wide_gemm_encoder_torch import bdh_wide_gemm_encoder_step


def bdh_stream_decode_step_packed_encoder_inplace(
    model: PackedEncoderBDH,
    states: list[torch.Tensor],
    idx_token: torch.Tensor,
    position: torch.Tensor,
) -> torch.Tensor:
    """Single-token (L=1) decode step using model.encoder_packed. Mutates
    `states` in place (same CUDA-graph-safe design as
    bdh_stream_decode_step_graph_safe_inplace) and returns only logits."""
    c = model.config
    B = idx_token.shape[0]
    D = c.n_embd
    nh = c.n_head
    N = D * c.mlp_internal_dim_multiplier // nh

    x = model.embed(idx_token).unsqueeze(1)
    x = model.ln(x)

    r_phases = position.to(model.attn.freqs.dtype).view(1, 1, 1, 1) * model.attn.freqs

    for level in range(c.n_layer):
        x_latent = bdh_wide_gemm_encoder_step(x, model.encoder_packed, nh, N)
        x_sparse = F.relu(x_latent)
        QR = model.attn.rope(r_phases, x_sparse)
        KR = QR
        V = x

        prefix_state = states[level]
        cross = QR @ prefix_state
        yKV = model.ln(cross)

        y_latent = yKV @ model._w(model.encoder_v)
        y_sparse = F.relu(y_latent)
        xy_sparse = x_sparse * y_sparse
        xy_sparse = model.drop(xy_sparse)
        yMLP = xy_sparse.transpose(1, 2).reshape(B, 1, 1, N * nh) @ model._w(model.decoder)
        y = model.ln(yMLP)
        x = model.ln(x + y)

        chunk_contribution = KR.mT @ V
        prefix_state.add_(chunk_contribution)

    logits = x.view(B, 1, D) @ model.lm_head
    return logits
