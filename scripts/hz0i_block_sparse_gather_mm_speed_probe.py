"""Real, isolated speed probe: dense factorized encode/decode vs the
gather_mm-based block-routed path (reference/hz0i_block_sparse_bdh_gather_mm.py),
at the real 0.3B shape and the live run's actual batch/seq. Follow-up to
scripts/hz0i_block_sparse_speed_probe.py (the hand-written Metal kernel
version, which was 12x SLOWER than dense -- docs/restart/hz0i_block_sparse_kernel_results.md).
"""
from __future__ import annotations

import time

import mlx.core as mx

from reference.hz0i_bdh_mlx import BDHConfig, _enc, _dec
from reference.hz0i_block_sparse_bdh_mlx import block_router_logits, select_blocks
from reference.hz0i_block_sparse_bdh_gather_mm import (
    pack_encode_bank, pack_decode_bank, combined_index, block_encode_decode_sorted,
)


def _timed(fn, repeats=20, warmup=5):
    for _ in range(warmup):
        mx.eval(fn())
    start = time.perf_counter()
    for _ in range(repeats):
        out = fn()
        mx.eval(out)
    return (time.perf_counter() - start) / repeats * 1000.0


def main():
    cfg = BDHConfig()
    H, D, rank = cfg.n_head, cfg.n_embd, cfg.rank
    N = cfg.mlp_internal_dim_multiplier * D // H
    B, T = 16, 128
    BT = B * T
    G = 8
    Nb = N // G
    tokens = H * BT
    assert N % G == 0

    print(f"config: H={H} D={D} rank={rank} N={N} B={B} T={T} G={G} Nb={Nb}")

    key = mx.random.key(0)
    ks = mx.random.split(key, 6)
    x = mx.random.normal((B, 1, T, D), key=ks[0]).astype(mx.bfloat16)
    enc_l = mx.random.normal((H, D, rank), key=ks[1]) * 0.02
    enc_r = mx.random.normal((H, rank, N), key=ks[2]) * (0.02 / (rank ** 0.5))
    dec_l = mx.random.normal((H, N, rank), key=ks[3]) * (0.02 / (N ** 0.5))
    dec_r = mx.random.normal((H, rank, D), key=ks[4]) * 0.02
    router_w = mx.random.normal((H, rank, G), key=ks[5]) * 0.02
    mx.eval(x, enc_l, enc_r, dec_l, dec_r, router_w)

    # ---- dense baseline ----
    def dense_pass():
        xs = mx.maximum(_enc(x, enc_l, enc_r, cfg), 0.0)
        return _dec(xs, dec_l, dec_r, cfg)

    dense_ms = _timed(dense_pass)

    # ---- gather_mm block-routed path ----
    x_bh = mx.broadcast_to(x, (B, H, T, D)).astype(mx.float32)
    z = mx.einsum("bhtd,hdr->bhtr", x_bh, enc_l)  # [B,H,T,rank]
    z_hbt = mx.transpose(z, (1, 0, 2, 3)).reshape(H, BT, rank)
    z_flat = z_hbt.reshape(tokens, rank)
    head_tok = mx.repeat(mx.arange(H), BT)
    mx.eval(z_flat, head_tok)

    def gather_pass():
        logits = block_router_logits(z_hbt, router_w)
        block_idx = select_blocks(logits).reshape(tokens)
        idx = combined_index(head_tok, block_idx, G)
        enc_bank = pack_encode_bank(enc_r, G)
        dec_bank = pack_decode_bank(dec_l, G)
        # block_encode_decode_sorted no longer uses sorted_indices=True --
        # a real correctness bug was found in that flag for this call
        # pattern (see the function's own docstring and
        # docs/restart/hz0i_block_sparse_kernel_results.md). This is now
        # just the verified-correct unsorted gather_mm path.
        return block_encode_decode_sorted(z_flat, enc_bank, dec_bank, idx)

    gather_ms = _timed(gather_pass)

    print(f"\ndense  _enc+_dec:            {dense_ms:.3f} ms/call")
    print(f"gather_mm block-routed:      {gather_ms:.3f} ms/call  ({100 / G:.1f}% of N)")
    print(f"speedup:                     {dense_ms / gather_ms:.2f}x")
    print("\nHONEST CAVEATS: isolated encode+decode only, not full 8-layer forward.")
    print("Real FLOP reduction via block routing (top-1 of G) is a genuine")
    print("architecture change -- quality/loss impact NOT measured here.")
    print("A sorted_indices=True fast path was tried and measured ~6x faster,")
    print("but was found to give WRONG output for this call pattern and was")
    print("retracted -- see docs/restart/hz0i_block_sparse_kernel_results.md.")


if __name__ == "__main__":
    main()
