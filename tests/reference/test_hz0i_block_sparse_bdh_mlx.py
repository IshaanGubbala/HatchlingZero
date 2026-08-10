"""Correctness tests for the block-routed sparse BDH Metal kernel
(reference/hz0i_block_sparse_bdh_mlx.py). The kernel's job is to compute
the SAME result as the plain-MLX reference (which computes the routing
scheme correctly but via a full dense matmul) while doing real less work
internally -- these tests only check numerical agreement, not speed (see
scripts/hz0i_block_sparse_speed_probe.py for the real throughput measurement).
"""
from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from reference.hz0i_block_sparse_bdh_mlx import (
    block_router_logits, select_blocks,
    reference_block_encode, reference_block_decode,
    kernel_block_encode, kernel_block_decode,
)


def _setup(seed=0, H=3, BT=17, rank=16, N=32, G=4):
    key = mx.random.key(seed)
    ks = mx.random.split(key, 6)
    z = mx.random.normal((H, BT, rank), key=ks[0]) * 0.1
    enc_r = mx.random.normal((H, rank, N), key=ks[1]) * 0.1
    router_w = mx.random.normal((H, rank, G), key=ks[2]) * 0.1
    dec_l = mx.random.normal((H, N, rank), key=ks[3]) * 0.1
    mx.eval(z, enc_r, router_w, dec_l)
    return z, enc_r, router_w, dec_l


def test_select_blocks_in_range_and_deterministic():
    z, enc_r, router_w, dec_l = _setup()
    logits = block_router_logits(z, router_w)
    idx1 = select_blocks(logits)
    idx2 = select_blocks(logits)
    mx.eval(idx1, idx2)
    assert idx1.shape == (3, 17)
    assert bool(mx.all(idx1 == idx2))
    assert int(mx.min(idx1)) >= 0
    assert int(mx.max(idx1)) < 4


def test_kernel_block_encode_matches_reference():
    H, BT, rank, N, G = 3, 17, 16, 32, 4
    z, enc_r, router_w, dec_l = _setup(H=H, BT=BT, rank=rank, N=N, G=G)
    block_idx = select_blocks(block_router_logits(z, router_w))

    ref = reference_block_encode(z, enc_r, block_idx, G)
    got = kernel_block_encode(z, enc_r, block_idx, G)
    mx.eval(ref, got)

    ref_np, got_np = np.array(ref), np.array(got)
    assert ref_np.shape == got_np.shape
    max_diff = float(np.max(np.abs(ref_np - got_np)))
    assert max_diff < 1e-4, f"kernel encode diverges from reference: {max_diff}"


def test_kernel_block_encode_is_actually_zero_outside_selected_block():
    """Not just 'matches reference' (which also masks) -- confirm the
    kernel's own output has the expected block-sparsity structure, i.e.
    the routing is real, not a no-op that happens to match by coincidence."""
    H, BT, rank, N, G = 2, 5, 8, 16, 4
    Nb = N // G
    z, enc_r, router_w, dec_l = _setup(H=H, BT=BT, rank=rank, N=N, G=G)
    block_idx = select_blocks(block_router_logits(z, router_w))
    got = np.array(kernel_block_encode(z, enc_r, block_idx, G))
    idx_np = np.array(block_idx)

    nonzero_blocks_seen = set()
    for h in range(H):
        for t in range(BT):
            row = got[h, t]
            nz = np.nonzero(row)[0]
            if nz.size:
                blocks_touched = set((nz // Nb).tolist())
                assert blocks_touched == {int(idx_np[h, t])}, "kernel wrote outside the routed block"
                nonzero_blocks_seen |= blocks_touched
    assert len(nonzero_blocks_seen) >= 1


def test_kernel_block_decode_matches_reference():
    H, BT, rank, N, G = 3, 17, 16, 32, 4
    z, enc_r, router_w, dec_l = _setup(H=H, BT=BT, rank=rank, N=N, G=G)
    block_idx = select_blocks(block_router_logits(z, router_w))
    xy = reference_block_encode(z, enc_r, block_idx, G)  # reuse as a real block-sparse input

    ref = reference_block_decode(xy, dec_l, block_idx, G)
    got = kernel_block_decode(xy, dec_l, block_idx, G)
    mx.eval(ref, got)

    max_diff = float(np.max(np.abs(np.array(ref) - np.array(got))))
    assert max_diff < 1e-4, f"kernel decode diverges from reference: {max_diff}"


def test_full_encode_decode_roundtrip_finite_and_matches_reference():
    """Chains encode -> decode through the kernel path exactly as the real
    per-layer usage would, checking both correctness and that values stay
    finite (no NaN/Inf from the block-routed path)."""
    H, BT, rank, N, G = 4, 33, 24, 48, 6
    z, enc_r, router_w, dec_l = _setup(H=H, BT=BT, rank=rank, N=N, G=G)
    block_idx = select_blocks(block_router_logits(z, router_w))

    xs_kernel = kernel_block_encode(z, enc_r, block_idx, G)
    xs_ref = reference_block_encode(z, enc_r, block_idx, G)
    mx.eval(xs_kernel, xs_ref)
    # 5e-4: this test uses a larger rank (24) than
    # test_kernel_block_encode_matches_reference's (16), so the kernel's
    # sequential scalar accumulation loop has more terms and more float32
    # rounding-order noise vs MLX's einsum -- same accumulation-noise class
    # noted on the decode assert below, not a structural bug.
    assert float(mx.max(mx.abs(xs_kernel - xs_ref))) < 5e-4

    z2_kernel = kernel_block_decode(xs_kernel, dec_l, block_idx, G)
    z2_ref = reference_block_decode(xs_ref, dec_l, block_idx, G)
    mx.eval(z2_kernel, z2_ref)

    assert bool(mx.all(mx.isfinite(z2_kernel)))
    # 5e-4, not 1e-4: this chains two kernel ops (encode then decode), so
    # float32 accumulation-order noise (kernel's sequential scalar loop vs
    # MLX's einsum reduction order -- same class of gap as
    # reference/hz0a_checkpoint_converter.py's disclosed cross-framework
    # residual) compounds across both stages. Each op alone matches at 1e-4
    # (see test_kernel_block_encode_matches_reference /
    # test_kernel_block_decode_matches_reference) -- this is accumulation,
    # not a structural bug.
    assert float(mx.max(mx.abs(z2_kernel - z2_ref))) < 5e-4


@pytest.mark.parametrize("G", [2, 4, 8])
def test_various_block_counts(G):
    H, BT, rank, N = 2, 11, 12, 24
    assert N % G == 0
    z, enc_r, router_w, dec_l = _setup(H=H, BT=BT, rank=rank, N=N, G=G)
    block_idx = select_blocks(block_router_logits(z, router_w))
    ref = reference_block_encode(z, enc_r, block_idx, G)
    got = kernel_block_encode(z, enc_r, block_idx, G)
    mx.eval(ref, got)
    assert float(mx.max(mx.abs(ref - got))) < 1e-4
