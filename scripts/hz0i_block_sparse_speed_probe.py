"""Real, isolated speed probe: dense factorized encode/decode
(reference/hz0i_bdh_mlx.py's `_enc`/`_dec`) vs the block-routed sparse
Metal kernel (reference/hz0i_block_sparse_bdh_mlx.py), at the REAL 0.3B
training shape (H=12, D=768, rank=704, N=9216) and the live run's actual
batch/seq (B=16, T=128 -- docs/restart/hz0i_master_work_log.md section 2's
best config). Isolated probe: measures the encode+decode pair only (the
documented dominant cost), not a full model forward -- report this
honestly, do not extrapolate to full-model tok/s without checking the
project's own precedent for isolated-vs-live gaps (e.g. E9's ctypes
boundary cost, the checkpoint converter's toy-vs-real-scale gap).
"""
from __future__ import annotations

import time

import mlx.core as mx

from reference.hz0i_bdh_mlx import BDHConfig, _enc, _dec
from reference.hz0i_block_sparse_bdh_mlx import (
    block_router_logits, select_blocks, kernel_block_encode, kernel_block_decode,
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
    cfg = BDHConfig()  # n_layer=8, n_embd=768, n_head=12, mlp_internal_dim_multiplier=144, rank=704
    H, D, rank = cfg.n_head, cfg.n_embd, cfg.rank
    N = cfg.mlp_internal_dim_multiplier * D // H
    B, T = 16, 128
    BT = B * T
    G = 8
    assert N % G == 0, (N, G)

    print(f"config: H={H} D={D} rank={rank} N={N} B={B} T={T} G={G} Nb={N // G}")

    key = mx.random.key(0)
    ks = mx.random.split(key, 6)
    x = mx.random.normal((B, 1, T, D), key=ks[0]).astype(mx.bfloat16)  # [B,1,T,D] pre-broadcast
    enc_l = mx.random.normal((H, D, rank), key=ks[1]) * 0.02
    enc_r = mx.random.normal((H, rank, N), key=ks[2]) * (0.02 / (rank ** 0.5))
    dec_l = mx.random.normal((H, N, rank), key=ks[3]) * (0.02 / (N ** 0.5))
    dec_r = mx.random.normal((H, rank, D), key=ks[4]) * 0.02
    router_w = mx.random.normal((H, rank, G), key=ks[5]) * 0.02
    mx.eval(x, enc_l, enc_r, dec_l, dec_r, router_w)

    # ---- dense baseline (existing _enc/_dec, exact match to the live model) ----
    def dense_pass():
        xs = mx.maximum(_enc(x, enc_l, enc_r, cfg), 0.0)  # [B,H,T,N]
        return _dec(xs, dec_l, dec_r, cfg)

    dense_ms = _timed(dense_pass)

    # ---- block-routed kernel path ----
    # z = x @ enc_l (cheap, same as dense's first step; shared by both encode kernel and router)
    x_bh = mx.broadcast_to(x, (B, H, T, D)).astype(mx.float32)
    z = mx.einsum("bhtd,hdr->bhtr", x_bh, enc_l)  # [B,H,T,rank]
    # reshape [B,H,T,rank] -> [H,BT,rank] for the kernel's (H,BT,*) convention
    z_hbt = mx.transpose(z, (1, 0, 2, 3)).reshape(H, BT, rank)
    mx.eval(z_hbt)

    def block_pass():
        logits = block_router_logits(z_hbt, router_w)
        block_idx = select_blocks(logits)
        xs = kernel_block_encode(z_hbt, enc_r, block_idx, G)  # [H,BT,N]
        return kernel_block_decode(xs, dec_l, block_idx, G)  # [H,BT,rank]

    block_ms = _timed(block_pass)

    print(f"\ndense  _enc+_dec:        {dense_ms:.3f} ms/call")
    print(f"block-sparse kernel:     {block_ms:.3f} ms/call  (G={G}, {N // G}/{N} = {100/G:.1f}% of N)")
    print(f"speedup:                 {dense_ms / block_ms:.2f}x")
    print("\nHONEST CAVEATS:")
    print("- isolated probe: encode+decode only, not a full 8-layer model forward")
    print("- block routing changes what the model computes (top-1 of G blocks) --")
    print("  this is NOT numerically equivalent to the dense model; quality impact")
    print("  (loss trajectory on real data) is NOT measured here, only kernel")
    print("  correctness (see tests/reference/test_hz0i_block_sparse_bdh_mlx.py)")
    print("  and this isolated timing.")


if __name__ == "__main__":
    main()
