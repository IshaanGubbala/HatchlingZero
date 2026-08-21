"""Exact activation-sparse decoder path for the faithful BDH model.

BDH's decoder input is ``relu(xE) * relu(yE_v)``.  Every zero in that
product is exact, not a pruning threshold or approximation.  Converting the
2-D decoder input to a sparse tensor therefore permits a sparse-dense matrix
multiplication to skip zero products without changing the model function.

This module intentionally leaves the encoder, attention, and encoder_v paths
unchanged from the already-verified wide-GEMM implementation.  It is the
smallest useful experiment for determining whether the vendor sparse backend
turns BDH's realized activation sparsity into real training savings.
"""
from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F

from reference.hz0h_bdh_bmm_encoder_v_torch import bmm_encoder_v_step
from reference.hz0h_bdh_torch import BDH
from reference.hz0h_bdh_wide_gemm_encoder_torch import (
    bdh_wide_gemm_encoder_step,
    wide_encoder_view,
)

SparseLayout = Literal["auto", "coo", "csr"]


def exact_sparse_decoder_mm(
    decoder_input: torch.Tensor,
    decoder: torch.Tensor,
    *,
    layout: SparseLayout = "auto",
) -> torch.Tensor:
    """Compute ``decoder_input @ decoder`` while skipping exact zeros.

    ``decoder_input`` must be two-dimensional and must come from BDH's
    ``relu(a) * relu(b)`` gate. Sparse autograd returns zero input-gradient
    at omitted entries rather than materializing the dense matmul gradient;
    this is exact for BDH because the gate Jacobian is also zero at every
    omitted entry. It is not a drop-in gradient-equivalent operator for an
    arbitrary leaf tensor whose zeros could later become trainable.

    COO is the compatibility
    fallback because it supports BF16 autograd on CPU in current PyTorch.
    CUDA defaults to CSR, whose single column-index array is materially less
    expensive than COO's row-and-column arrays at production scale.
    """
    if decoder_input.ndim != 2 or decoder.ndim != 2:
        raise ValueError("decoder_input and decoder must both be 2-D")
    if decoder_input.shape[1] != decoder.shape[0]:
        raise ValueError(
            f"incompatible decoder shapes {tuple(decoder_input.shape)} and {tuple(decoder.shape)}"
        )
    if layout not in {"auto", "coo", "csr"}:
        raise ValueError(f"unsupported sparse layout: {layout}")

    selected_layout = "csr" if layout == "auto" and decoder_input.is_cuda else layout
    if selected_layout == "auto":
        selected_layout = "coo"
    sparse_input = (
        decoder_input.to_sparse_csr()
        if selected_layout == "csr"
        else decoder_input.to_sparse()
    )
    return torch.sparse.mm(sparse_input, decoder)


def bdh_exact_sparse_decoder_forward(
    model: BDH,
    idx: torch.Tensor,
    n_iterations: int,
    targets: torch.Tensor | None = None,
    *,
    sparse_layout: SparseLayout = "auto",
):
    """Run exact BDH math with only the final per-round decoder sparse."""
    if n_iterations < 0:
        raise ValueError("n_iterations must be non-negative")

    config = model.config
    batch, sequence = idx.shape
    dim = config.n_embd
    heads = config.n_head
    latent = dim * config.mlp_internal_dim_multiplier // heads

    x = model.ln(model.embed(idx).unsqueeze(1))
    encoder_wide = wide_encoder_view(model.encoder)

    for _ in range(n_iterations):
        x_latent = bdh_wide_gemm_encoder_step(x, encoder_wide, heads, latent)
        x_sparse = F.relu(x_latent)

        y_kv = model.attn(Q=x_sparse, K=x_sparse, V=x)
        y_kv = model.ln(y_kv)
        y_sparse = F.relu(bmm_encoder_v_step(y_kv, model.encoder_v))

        decoder_input = (x_sparse * y_sparse).transpose(1, 2).reshape(
            batch * sequence, heads * latent
        )
        decoder_output = exact_sparse_decoder_mm(
            decoder_input,
            model.decoder,
            layout=sparse_layout,
        ).view(batch, 1, sequence, dim)
        x = model.ln(x + model.ln(decoder_output))

    logits = x.view(batch, sequence, dim) @ model.lm_head
    loss = None
    if targets is not None:
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
    return logits, loss
