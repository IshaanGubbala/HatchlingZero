"""Wires both real GPU-layout remaps
(`reference/hz0h_bdh_wide_gemm_encoder_torch.py`'s wide-GEMM encoder
projection, `reference/hz0h_bdh_bmm_encoder_v_torch.py`'s batched-GEMM
encoder_v projection) into a full trainable recurrent forward pass --
the follow-up both of those files' own docstrings flagged as
"a separate, disclosed follow-up, left for after the layout claim
itself is confirmed."

Both remaps were originally forward-only. `wide_encoder_view` no longer
detaches (see that file's own updated docstring) -- `permute`/
`reshape`/`contiguous` are natively differentiable, so removing the
`.detach()` call was the entire fix needed for the encoder projection.
`bmm_encoder_v_step` never detached anything and was already
differentiable as written; it just had never been plugged into an
actual training loop and checked end-to-end.

Zero change to BDH's math -- same real per-layer computation as
`bdh_variable_depth_forward`, just using the wide-GEMM/batched-GEMM
layouts instead of the oracle's broadcasted per-head matmuls for the
encoder and encoder_v projections specifically (attention, decoder, and
everything else stays byte-for-byte the oracle's own expression).
Proven bit-exact on BOTH logits AND gradients against
`bdh_variable_depth_forward` -- not asserted, checked by
`tests/reference/test_hz0h_bdh_wide_gemm_trainable_torch.py`.

Never modifies `reference/hz0h_bdh_torch.py`,
`reference/hz0h_bdh_wide_gemm_encoder_torch.py`, or
`reference/hz0h_bdh_bmm_encoder_v_torch.py`.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from reference.hz0h_bdh_bmm_encoder_v_torch import bmm_encoder_v_step
from reference.hz0h_bdh_torch import BDH
from reference.hz0h_bdh_wide_gemm_encoder_torch import bdh_wide_gemm_encoder_step, wide_encoder_view


def bdh_wide_gemm_trainable_forward(
    model: BDH,
    idx: torch.Tensor,
    n_iterations: int,
    targets: torch.Tensor | None = None,
):
    """Same real per-layer computation as `bdh_variable_depth_forward`,
    with the encoder projection computed via one big wide-GEMM
    (`bdh_wide_gemm_encoder_step`) instead of a broadcasted per-head
    matmul, and the encoder_v projection computed via an explicit
    per-head batched GEMM (`bmm_encoder_v_step`) instead of relying on
    implicit broadcast layout. Ternary quantization (`model._w`) is
    intentionally NOT applied here -- both remaps assume a plain
    `nn.Parameter`, matching this project's own T0 ternary contract
    scope (encoder/encoder_v/decoder only quantized via the oracle's
    own `_w` hook, not through these execution-layout-only remaps)."""
    C = model.config
    B, T = idx.size()
    D = C.n_embd
    nh = C.n_head
    N = D * C.mlp_internal_dim_multiplier // nh

    x = model.embed(idx).unsqueeze(1)
    x = model.ln(x)

    for _iteration in range(n_iterations):
        encoder_wide = wide_encoder_view(model.encoder)
        x_latent = bdh_wide_gemm_encoder_step(x, encoder_wide, nh, N)
        x_sparse = F.relu(x_latent)

        yKV = model.attn(Q=x_sparse, K=x_sparse, V=x)
        yKV = model.ln(yKV)

        y_latent = bmm_encoder_v_step(yKV, model.encoder_v)
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
