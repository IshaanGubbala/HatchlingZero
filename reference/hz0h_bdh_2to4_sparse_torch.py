"""Real 2:4 structured-sparsity pruning for BDH's three dense projection
matrices (encoder, encoder_v, decoder), per
plans/deep-research-report(6).md's structured-sparsity recommendation.

Real, disclosed hardware constraint: NVIDIA Ampere's actual 2:4
sparse-Tensor-Core acceleration (`torch.sparse.to_sparse_semi_structured`,
backed by cuSPARSELt) is CUDA-only and only accepts 2D tensors -- it
cannot even be constructed on this Mac (no CUDA), let alone benchmarked
for real speedup here. This file splits the work honestly along that
line:

- `prune_to_2_of_4` and `apply_2to4_pruning_to_bdh`: the actual pruning
  MATH (which of every 4 consecutive weights along the contraction
  dimension survive) -- pure PyTorch, dimension-agnostic, runs
  identically on CPU/MPS/CUDA, fully testable here.
- `bdh_2to4_semi_structured_forward`: the CUDA-only execution path that
  wraps the pruned weights in the real hardware sparse format for actual
  Tensor Core acceleration. Raises clearly if CUDA is unavailable rather
  than silently falling back -- there is no honest speed claim to make
  on a machine that can't run this path at all.

The pruning itself changes BDH's math (this is a real architecture
modification, not a layout remap) -- real quality impact must be
measured, never assumed zero.
"""
from __future__ import annotations

import torch

from reference.hz0h_bdh_torch import BDH, BDHConfig


def prune_to_2_of_4(weight: torch.Tensor, dim: int) -> torch.Tensor:
    """Zero all but the 2 largest-magnitude entries in every consecutive
    group of 4 elements along `dim`. Matches the real hardware 2:4
    semi-structured sparsity pattern (NVIDIA Ampere+ sparse Tensor
    Cores): the size along `dim` must be a positive multiple of 4.

    Ties are broken by `torch.topk`'s own stable ordering (lower index
    wins) -- deterministic, not randomized, so the same input always
    prunes to the same mask.
    """
    size_along_dim = weight.shape[dim]
    if size_along_dim % 4 != 0:
        raise ValueError(
            f"2:4 pruning requires size along dim {dim} to be a multiple of 4, got {size_along_dim}"
        )
    moved = weight.movedim(dim, -1)
    grouped_shape = moved.shape[:-1] + (size_along_dim // 4, 4)
    grouped = moved.reshape(grouped_shape)
    magnitudes = grouped.abs()
    _, keep_indices = torch.topk(magnitudes, k=2, dim=-1)
    mask = torch.zeros_like(grouped, dtype=torch.bool)
    mask.scatter_(-1, keep_indices, True)
    pruned_grouped = torch.where(mask, grouped, torch.zeros_like(grouped))
    pruned_moved = pruned_grouped.reshape(moved.shape)
    return pruned_moved.movedim(-1, dim)


def apply_2to4_pruning_to_bdh(model: BDH) -> BDH:
    """Returns a NEW BDH model (does not mutate `model`) with encoder,
    encoder_v, and decoder pruned to the real 2:4 pattern along each
    matrix's own contraction dimension: `encoder`/`encoder_v` are
    `(nh, D, N)`, contracted over `D` (dim=1, `x @ encoder`); `decoder`
    is `(nh*N, D)`, contracted over `nh*N` (dim=0,
    `xy_sparse_flat @ decoder`).
    """
    import copy

    pruned = copy.deepcopy(model)
    with torch.no_grad():
        pruned.encoder.copy_(prune_to_2_of_4(pruned.encoder, dim=1))
        pruned.encoder_v.copy_(prune_to_2_of_4(pruned.encoder_v, dim=1))
        pruned.decoder.copy_(prune_to_2_of_4(pruned.decoder, dim=0))
    return pruned


def bdh_2to4_semi_structured_forward(model: BDH, idx: torch.Tensor, targets: torch.Tensor | None = None):
    """Real, PARTIAL hardware-accelerated forward using CUDA's
    semi-structured sparse Tensor Core path. Requires the model's
    encoder/encoder_v/decoder to already be 2:4-pruned (e.g. via
    `apply_2to4_pruning_to_bdh`) -- this function does not prune, it
    only executes the already-pruned weights.

    Real, disclosed scope limit: only `encoder` and `decoder` go through
    the actual sparse hardware path here. Both reduce to a single 2D
    GEMM over a SHARED input (`encoder`: all heads read the same `x`,
    exactly like `reference/hz0h_bdh_wide_gemm_encoder_torch.py`;
    `decoder`: reads the already-flattened, head-merged `xy_sparse`).
    `encoder_v` does NOT reduce this way -- each head's `yKV` input is
    genuinely different post-attention data (the same real limitation
    already documented in `reference/hz0h_bdh_bmm_encoder_v_torch.py`),
    and `to_sparse_semi_structured` only accepts a single 2D tensor, no
    per-head batched variant exists in this torch version. `encoder_v`
    runs as a plain dense per-head matmul here (still 2:4-pruned in
    VALUE, just not through the accelerated sparse GEMM path) -- so this
    function measures a real but partial hardware win, 2 of 3 matrices,
    not all 3.

    CUDA-only by hard library constraint (`to_sparse_semi_structured`
    only supports 2D CUDA tensors) -- raises clearly rather than
    silently falling back to a dense/CPU path, since a fallback here
    would make no real speed claim meaningful.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "bdh_2to4_semi_structured_forward requires CUDA -- "
            "torch.sparse.to_sparse_semi_structured only supports 2D CUDA tensors, "
            "there is no CPU/MPS execution path for the real hardware-accelerated GEMM."
        )
    import torch.nn.functional as F
    from torch.sparse import to_sparse_semi_structured

    C = model.config
    B, T = idx.size()
    D = C.n_embd
    nh = C.n_head
    N = D * C.mlp_internal_dim_multiplier // nh

    encoder_wide = model.encoder.detach().permute(1, 0, 2).reshape(D, nh * N).contiguous()
    encoder_sparse = to_sparse_semi_structured(encoder_wide)
    decoder_sparse = to_sparse_semi_structured(model.decoder.detach().contiguous())

    x = model.ln(model.embed(idx).unsqueeze(1))
    for _ in range(C.n_layer):
        x2 = x.reshape(B * T, D)
        x_latent = (x2 @ encoder_sparse).reshape(B, T, nh, N).permute(0, 2, 1, 3)
        x_sparse = F.relu(x_latent)

        yKV = model.ln(model.attn(Q=x_sparse, K=x_sparse, V=x))

        y_latent = yKV @ model.encoder_v  # dense per-head fallback, see docstring
        y_sparse = F.relu(y_latent)
        xy_sparse = model.drop(x_sparse * y_sparse)

        xy_flat = xy_sparse.transpose(1, 2).reshape(B * T, N * nh)
        yMLP = (xy_flat @ decoder_sparse).reshape(B, 1, T, D)
        y = model.ln(yMLP)
        x = model.ln(x + y)

    logits = x.view(B, T, D) @ model.lm_head
    loss = None
    if targets is not None:
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
    return logits, loss
