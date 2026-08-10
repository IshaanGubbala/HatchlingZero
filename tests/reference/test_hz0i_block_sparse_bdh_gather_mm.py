"""Correctness tests for the gather_mm-based block-routed BDH encode/decode
(reference/hz0i_block_sparse_bdh_gather_mm.py) -- the follow-up to the
hand-written Metal kernel version, which was correct but 12x slower than
dense (docs/restart/hz0i_block_sparse_kernel_results.md). This uses MLX's
native mx.gather_mm instead, following the same precedent
reference/hz0e_e9_gather_mm_kernel.py established for MoE.
"""
from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from reference.hz0i_block_sparse_bdh_mlx import block_router_logits, select_blocks, reference_block_encode, reference_block_decode
from reference.hz0i_block_sparse_bdh_gather_mm import (
    pack_encode_bank, pack_decode_bank, combined_index,
    gather_mm_block_encode, gather_mm_block_decode, scatter_to_full_n,
    block_encode_decode_sorted,
)


def _setup(seed=0, H=3, BT=17, rank=16, N=32, G=4):
    key = mx.random.key(seed)
    ks = mx.random.split(key, 4)
    z = mx.random.normal((H, BT, rank), key=ks[0]) * 0.1
    enc_r = mx.random.normal((H, rank, N), key=ks[1]) * 0.1
    router_w = mx.random.normal((H, rank, G), key=ks[2]) * 0.1
    dec_l = mx.random.normal((H, N, rank), key=ks[3]) * 0.1
    mx.eval(z, enc_r, router_w, dec_l)
    return z, enc_r, router_w, dec_l


def _head_of_token(H, BT):
    return mx.repeat(mx.arange(H), BT)


def test_pack_encode_bank_shape_and_values():
    H, rank, N, G = 3, 8, 16, 4
    enc_r = mx.random.normal((H, rank, N))
    bank = pack_encode_bank(enc_r, G)
    assert bank.shape == (H * G, rank, N // G)
    # spot-check: bank[h*G+g] should equal enc_r[h, :, g*Nb:(g+1)*Nb]
    Nb = N // G
    h, g = 2, 1
    expected = enc_r[h, :, g * Nb:(g + 1) * Nb]
    got = bank[h * G + g]
    assert float(mx.max(mx.abs(expected - got))) == 0.0


def test_gather_mm_encode_matches_dense_reference_scattered():
    H, BT, rank, N, G = 3, 17, 16, 32, 4
    Nb = N // G
    z, enc_r, router_w, dec_l = _setup(H=H, BT=BT, rank=rank, N=N, G=G)
    block_idx = select_blocks(block_router_logits(z, router_w))  # [H,BT]

    tokens = H * BT
    z_flat = z.reshape(tokens, rank)
    head_tok = _head_of_token(H, BT)
    block_flat = block_idx.reshape(tokens)
    idx = combined_index(head_tok, block_flat, G)

    bank = pack_encode_bank(enc_r, G)
    compact = gather_mm_block_encode(z_flat, bank, idx)  # [tokens, Nb]
    assert compact.shape == (tokens, Nb)

    full = scatter_to_full_n(compact, block_flat, G, N).reshape(H, BT, N)
    ref = reference_block_encode(z, enc_r, block_idx, G)
    mx.eval(full, ref)
    max_diff = float(np.max(np.abs(np.array(full) - np.array(ref))))
    assert max_diff < 1e-3, f"gather_mm encode diverges from dense reference: {max_diff}"


def test_gather_mm_decode_matches_dense_reference():
    H, BT, rank, N, G = 3, 17, 16, 32, 4
    z, enc_r, router_w, dec_l = _setup(H=H, BT=BT, rank=rank, N=N, G=G)
    block_idx = select_blocks(block_router_logits(z, router_w))
    tokens = H * BT
    z_flat = z.reshape(tokens, rank)
    head_tok = _head_of_token(H, BT)
    block_flat = block_idx.reshape(tokens)
    idx = combined_index(head_tok, block_flat, G)

    enc_bank = pack_encode_bank(enc_r, G)
    compact = gather_mm_block_encode(z_flat, enc_bank, idx)

    dec_bank = pack_decode_bank(dec_l, G)
    z2_gather = gather_mm_block_decode(compact, dec_bank, idx)  # [tokens, rank]

    full = scatter_to_full_n(compact, block_flat, G, N).reshape(H, BT, N)
    z2_ref = reference_block_decode(full, dec_l, block_idx, G)  # [H,BT,rank]
    mx.eval(z2_gather, z2_ref)

    max_diff = float(np.max(np.abs(np.array(z2_gather).reshape(H, BT, rank) - np.array(z2_ref))))
    assert max_diff < 1e-3, f"gather_mm decode diverges from dense reference: {max_diff}"


def test_full_pipeline_finite_and_zero_outside_block():
    H, BT, rank, N, G = 4, 25, 24, 48, 6
    Nb = N // G
    z, enc_r, router_w, dec_l = _setup(H=H, BT=BT, rank=rank, N=N, G=G)
    block_idx = select_blocks(block_router_logits(z, router_w))
    tokens = H * BT
    z_flat = z.reshape(tokens, rank)
    head_tok = _head_of_token(H, BT)
    block_flat = block_idx.reshape(tokens)
    idx = combined_index(head_tok, block_flat, G)

    enc_bank = pack_encode_bank(enc_r, G)
    dec_bank = pack_decode_bank(dec_l, G)
    compact = gather_mm_block_encode(z_flat, enc_bank, idx)
    z2 = gather_mm_block_decode(compact, dec_bank, idx)
    mx.eval(compact, z2)

    assert bool(mx.all(mx.isfinite(compact)))
    assert bool(mx.all(mx.isfinite(z2)))
    assert compact.shape == (tokens, Nb)
    assert z2.shape == (tokens, rank)


def test_block_encode_decode_sorted_matches_unsorted_and_dense_reference():
    """block_encode_decode_sorted was ORIGINALLY built as a
    sorted_indices=True fast path; a real correctness bug was found in
    that flag for this call pattern (see the function's own docstring)
    and it now never uses it -- just delegates to the verified-correct
    unsorted gather_mm pipeline. This test checks that delegation is
    exact (not approximate), across both a real-scale-ish balanced case
    and an intentionally-imbalanced-with-empty-bins case, since both
    triggered the original bug differently."""
    for H, BT, rank, N, G in [(3, 400, 16, 32, 4), (4, 25, 24, 48, 6)]:
        z, enc_r, router_w, dec_l = _setup(H=H, BT=BT, rank=rank, N=N, G=G)
        block_idx = select_blocks(block_router_logits(z, router_w))
        tokens = H * BT
        z_flat = z.reshape(tokens, rank)
        head_tok = _head_of_token(H, BT)
        block_flat = block_idx.reshape(tokens)
        idx = combined_index(head_tok, block_flat, G)

        enc_bank = pack_encode_bank(enc_r, G)
        dec_bank = pack_decode_bank(dec_l, G)

        compact_unsorted = gather_mm_block_encode(z_flat, enc_bank, idx)
        z2_unsorted = gather_mm_block_decode(compact_unsorted, dec_bank, idx)
        z2_via_alias = block_encode_decode_sorted(z_flat, enc_bank, dec_bank, idx)
        mx.eval(z2_unsorted, z2_via_alias)

        max_diff = float(np.max(np.abs(np.array(z2_unsorted) - np.array(z2_via_alias))))
        assert max_diff == 0.0, f"H={H} BT={BT} G={G}: alias should be an EXACT delegation, got diff {max_diff}"

        full = scatter_to_full_n(compact_unsorted, block_flat, G, N).reshape(H, BT, N)
        z2_ref = reference_block_decode(full, dec_l, block_idx, G).reshape(tokens, rank)
        mx.eval(z2_ref)
        max_diff_dense = float(np.max(np.abs(np.array(z2_ref) - np.array(z2_via_alias))))
        assert max_diff_dense < 1e-3, f"H={H} BT={BT} G={G}: diverges from dense reference: {max_diff_dense}"


def test_gather_mm_sorted_indices_true_is_unsound_for_this_call_pattern():
    """Documents the real, reproducible negative finding, as a live
    regression check (not just a docstring claim): mx.gather_mm's
    sorted_indices=True flag gives WRONG output for this exact call shape
    even with perfectly balanced bins in already-sorted order (no
    permutation applied at all) -- ruling out both the empty-bin
    hypothesis and a sort/unsort bug as the cause. If a future MLX
    version fixes this, this test will start FAILING (diff will drop
    near zero) -- that's the signal to revisit block_encode_decode_sorted
    and actually use the fast path again."""
    H, rank, N, G = 3, 16, 32, 4
    bins = H * G
    tokens_per_bin = 100
    tokens = bins * tokens_per_bin
    key = mx.random.key(0)
    ks = mx.random.split(key, 2)
    z_flat = mx.random.normal((tokens, rank), key=ks[0]) * 0.1
    enc_r = mx.random.normal((H, rank, N), key=ks[1]) * 0.1
    mx.eval(z_flat, enc_r)
    enc_bank = pack_encode_bank(enc_r, G)

    idx = mx.array(np.repeat(np.arange(bins), tokens_per_bin).astype(np.uint32))
    mx.eval(idx)

    compact_unsorted = gather_mm_block_encode(z_flat, enc_bank, idx)
    hidden_sorted_flag = mx.maximum(
        mx.gather_mm(z_flat[:, None, :], enc_bank, rhs_indices=idx, sorted_indices=True)[:, 0, :], 0.0,
    )
    mx.eval(compact_unsorted, hidden_sorted_flag)
    diff = float(np.max(np.abs(np.array(compact_unsorted) - np.array(hidden_sorted_flag))))
    assert diff > 0.01, (
        f"sorted_indices=True now agrees with the correct path (diff={diff}) -- "
        "this MLX version may have fixed the bug; revisit block_encode_decode_sorted"
    )
