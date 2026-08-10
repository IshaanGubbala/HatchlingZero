"""HZ-0I: block-routed sparse encode/decode via MLX's native `mx.gather_mm`,
NOT a hand-written Metal kernel.

`reference/hz0i_block_sparse_bdh_mlx.py`'s hand-rolled `mx.fast.metal_kernel`
version proved correct but was 12x SLOWER than dense (see
docs/restart/hz0i_block_sparse_kernel_results.md) -- a naive per-element
scalar loop cannot compete with Apple's tuned GEMM (simdgroup matrix ops,
tiling, vectorization) even doing 8x fewer real FLOPs.

This follows the SAME precedent `reference/hz0e_e9_gather_mm_kernel.py`
already established for MoE: use `mx.gather_mm` (MLX's native grouped/
gathered matmul, public since MLX 0.31.2) so the actual multiply-accumulate
work runs through Apple's own GEMM, and only the WEIGHT SELECTION (which
block's slice of the projection matrix) is gathered per row -- getting both
the real FLOP reduction (only Nb of N columns computed) and vendor-GEMM
per-FLOP efficiency, instead of neither.

Design: enc_r (H,rank,N) is reshaped into a bank of H*G separate (rank,Nb)
matrices, one per (head, block) pair. Each token's combined index
`head*G + routed_block` selects its matrix via `rhs_indices`; `gather_mm`
does the actual per-row batched matmul. Symmetric for dec_l ((H,N,rank) ->
bank of H*G (Nb,rank) matrices). The output stays in COMPACT (tokens, Nb)
form between encode and decode -- no full N-width materialization needed
for this pipeline (unlike the Metal-kernel version, which had to write into
a full N-width buffer since a Metal kernel's output shape is fixed at
dispatch time).
"""
from __future__ import annotations

import mlx.core as mx


def pack_encode_bank(enc_r: mx.array, num_blocks: int) -> mx.array:
    """enc_r:[H,rank,N] -> bank:[H*num_blocks, rank, Nb], one (rank,Nb)
    matrix per (head, block) pair, indexed as head*num_blocks + block."""
    H, rank, N = enc_r.shape
    Nb = N // num_blocks
    return enc_r.reshape(H, rank, num_blocks, Nb).transpose(0, 2, 1, 3).reshape(H * num_blocks, rank, Nb)


def pack_decode_bank(dec_l: mx.array, num_blocks: int) -> mx.array:
    """dec_l:[H,N,rank] -> bank:[H*num_blocks, Nb, rank]."""
    H, N, rank = dec_l.shape
    Nb = N // num_blocks
    return dec_l.reshape(H, num_blocks, Nb, rank).reshape(H * num_blocks, Nb, rank)


def combined_index(head_of_token: mx.array, block_idx: mx.array, num_blocks: int) -> mx.array:
    """head_of_token, block_idx: [tokens] int -> [tokens] uint32 combined
    (head, block) index into the H*num_blocks bank."""
    return (head_of_token.astype(mx.int32) * num_blocks + block_idx.astype(mx.int32)).astype(mx.uint32)


def gather_mm_block_encode(z_flat: mx.array, enc_bank: mx.array, idx: mx.array) -> mx.array:
    """z_flat:[tokens,rank], enc_bank:[H*G,rank,Nb], idx:[tokens] uint32 ->
    hidden:[tokens,Nb] (ReLU'd), COMPACT -- not padded to full N."""
    a = z_flat[:, None, :]  # [tokens,1,rank]
    out = mx.gather_mm(a, enc_bank, rhs_indices=idx)[:, 0, :]  # [tokens,Nb]
    return mx.maximum(out, 0.0)


def gather_mm_block_decode(xs_compact: mx.array, dec_bank: mx.array, idx: mx.array) -> mx.array:
    """xs_compact:[tokens,Nb], dec_bank:[H*G,Nb,rank], idx:[tokens] uint32
    -> z2:[tokens,rank]."""
    a = xs_compact[:, None, :]  # [tokens,1,Nb]
    return mx.gather_mm(a, dec_bank, rhs_indices=idx)[:, 0, :]  # [tokens,rank]


def block_encode_decode_sorted(
    z_flat: mx.array, enc_bank: mx.array, dec_bank: mx.array, idx: mx.array, num_bins: int | None = None,
) -> mx.array:
    """RETRACTED FAST PATH -- kept as a thin, always-correct alias. Do not
    remove the docstring below; it records a real, load-bearing negative
    finding about `mx.gather_mm(..., sorted_indices=True)` in this MLX
    build (0.29.3), discovered and then DISPROVEN in the same
    investigation, so nobody re-attempts it without re-reading this.

    First measurement (real, but incomplete): sorting tokens by combined
    (head,block) index and calling `gather_mm(..., sorted_indices=True)`
    measured ~1.65ms for encode alone vs dense's ~12ms (7.4x faster) and a
    full encode+decode+router+sort+unsort pipeline at ~3.67ms vs dense's
    22.4ms (6.1x faster) at the real 0.3B/B=16/T=128 shape (96 populated
    bins, min count 207). This was WRONG to trust without a correctness
    check against it -- and a same-session correctness test caught it:

    REAL BUG: `sorted_indices=True` gives INCORRECT output for this call
    pattern, and NOT only in the empty-bin case originally suspected.
    Controlled test: perfectly balanced bins (12 bins x exactly 100 tokens
    each), idx ALREADY in sorted order (no permutation applied at all) --
    still diverges from the verified-correct unsorted `gather_mm` path by
    0.156 max abs diff. This rules out "empty bins" or "sort/unsort
    permutation bug" as the cause; something about `sorted_indices=True`
    itself is unsound for this shape/usage in this MLX version. Not root-
    caused further (would need MLX source-level debugging, out of scope
    here) -- disclosed as a real, reproducible negative finding rather
    than either hidden or asserted-safe.

    CONSEQUENCE: this function does NOT use `sorted_indices=True` at all.
    It is a plain alias for the verified-correct (but not faster than
    dense) unsorted path, kept under this name only so any earlier caller
    expecting "the fast path" gets a correct answer, not a fast wrong one.
    See docs/restart/hz0i_block_sparse_kernel_results.md for the full,
    corrected measurement history: none of the three approaches tried
    (naive Metal kernel, unsorted gather_mm, sorted gather_mm) beat dense
    while also being correct.
    """
    del num_bins  # unused now that this never takes the sorted_indices path
    return gather_mm_block_decode(gather_mm_block_encode(z_flat, enc_bank, idx), dec_bank, idx)


def scatter_to_full_n(compact: mx.array, block_idx: mx.array, num_blocks: int, N: int) -> mx.array:
    """compact:[tokens,Nb], block_idx:[tokens] -> [tokens,N] padded with
    zeros outside each token's selected block. Only needed if downstream
    code (e.g. attention, which needs a shared N-basis across tokens with
    different active blocks) requires the full width -- the encode->decode
    pipeline itself never needs this."""
    tokens, Nb = compact.shape
    col = mx.arange(N).reshape(1, N) // Nb  # [1,N] block-of-column
    full = mx.zeros((tokens, N), dtype=compact.dtype)
    # place compact into its block's column range per-row via a one-hot-style
    # scatter: broadcast-compare block index, then use take-along-style write.
    starts = (block_idx * Nb).astype(mx.int32)  # [tokens]
    idx_in_row = mx.arange(N).reshape(1, N) - starts[:, None]  # [tokens,N], 0..Nb-1 within block
    in_block = (col == block_idx[:, None]).astype(compact.dtype)
    gathered = mx.take_along_axis(
        mx.concatenate([compact, mx.zeros((tokens, 1), dtype=compact.dtype)], axis=1),
        mx.clip(idx_in_row, 0, Nb).astype(mx.int32),
        axis=1,
    )
    return gathered * in_block
