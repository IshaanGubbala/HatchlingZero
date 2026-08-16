"""Real, trainable, end-to-end integration of the Stage 1 GPU-native
remaps validated in isolation this session
(plans/hatchlingzero_bdh_transformer_planning.md Stage 1): the wide-GEMM
encoder (reference/hz0h_bdh_wide_gemm_encoder_torch.py, real measured
1.705x forward-only), the explicit bmm encoder_v
(reference/hz0h_bdh_bmm_encoder_v_torch.py, real measured 1.509x
forward-only), and the Triton attention kernel
(reference/hz0h_bdh_triton_attention_torch.py, real measured 1.551x,
CUDA-only, falls back to the exact bounded PyTorch attention off-CUDA).

Each remap was validated in isolation: forward-only, frozen weights, one
op timed by itself. This file combines all three into one real
forward+backward pass to check whether those wins compose, survive real
gradient flow, and produce a real measured end-to-end speedup -- not
assumed multiplicative, that is the entire point of this file existing.

Real, disclosed cost this file pays that the isolated benchmarks did not:
`wide_encoder_step_live` rebuilds the wide encoder view from the LIVE
(non-detached) parameter every forward call, since a real trainable model
cannot use a cached/detached view without it going stale after every
optimizer step -- exactly the permute-every-forward cost the encoder
remap's own module docstring warned against. If this file's real
end-to-end number comes in worse than the isolated forward-only numbers
would suggest, that reshape-every-step cost is the first thing to check,
not a silently discarded assumption.

Never touches reference/hz0h_bdh_torch.py -- reads the oracle's own
weights (encoder/encoder_v/decoder/embed/ln/lm_head) directly to stay
parity-comparable; only the recurrent loop's execution layout differs.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from reference.hz0h_bdh_bmm_encoder_v_torch import bmm_encoder_v_step
from reference.hz0h_bdh_torch import BDH
from reference.hz0h_bdh_triton_attention_torch import bdh_triton_attention
from reference.hz0h_bdh_wide_gemm_encoder_torch import bdh_wide_gemm_encoder_step


def wide_encoder_step_live(x: torch.Tensor, encoder: torch.Tensor, nh: int, N: int) -> torch.Tensor:
    """Differentiable version of ``bdh_wide_gemm_encoder_step``: builds
    the wide view from the LIVE (non-detached) encoder parameter every
    call, so gradients flow back into ``encoder`` normally. See this
    module's own docstring for the real cost this pays versus the cached/
    detached ``wide_encoder_view`` used by the isolated forward-only
    benchmark."""
    D = encoder.shape[1]
    encoder_wide = encoder.permute(1, 0, 2).reshape(D, nh * N)
    return bdh_wide_gemm_encoder_step(x, encoder_wide, nh, N)


def bdh_gpu_native_forward(model: BDH, idx: torch.Tensor, targets: torch.Tensor | None = None):
    """Drop-in replacement for ``BDH.forward`` using the three validated
    Stage 1 GPU-native remaps. Same math as the oracle at every step --
    only execution layout differs. ``bdh_triton_attention`` already
    handles the CUDA/Triton-vs-fallback dispatch internally, so it's
    called unconditionally here."""
    C = model.config
    B, T = idx.size()
    D = C.n_embd
    nh = C.n_head
    N = D * C.mlp_internal_dim_multiplier // nh

    x = model.embed(idx).unsqueeze(1)
    x = model.ln(x)

    for _level in range(C.n_layer):
        x_latent = wide_encoder_step_live(x, model._w(model.encoder), nh, N)
        x_sparse = F.relu(x_latent)

        yKV = bdh_triton_attention(x_sparse, x, model.attn.freqs)
        yKV = model.ln(yKV)

        y_latent = bmm_encoder_v_step(yKV, model._w(model.encoder_v))
        y_sparse = F.relu(y_latent)
        xy_sparse = x_sparse * y_sparse
        xy_sparse = model.drop(xy_sparse)

        yMLP = xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ model._w(model.decoder)
        y = model.ln(yMLP)
        x = model.ln(x + y)

    logits = x.view(B, T, D) @ model.lm_head
    loss = None
    if targets is not None:
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
    return logits, loss
