"""HZ-0I: block-routed sparse encode/decode for the factorized BDH core,
as a real MLX custom Metal kernel (mx.fast.metal_kernel), following the
exact precedent already established and validated in this repo by
reference/hz0e_e9_mlx_native_kernel.py for MoE.

REAL, MEASURED PROBLEM (docs/restart/hz0i_master_work_log.md section 2):
the factorized BDH's dominant cost is materializing a [B,H,T,N] tensor
(N=9216 at the 0.3B profile) in `_enc`/`_dec` via a T*rank*N matmul, 8x
per step (once per layer). Two prior attempts at exploiting BDH's own
sparse-latent structure for real speed BOTH failed to move throughput:
- top-k ReLU sparsity (reference/hz0i_sparse_bdh.py): zeros most of the
  dense [B,H,T,N] tensor via `scatter` AFTER computing it in full --
  measured 26,483 vs 26,368 tok/s, i.e. no win, because the expensive
  matmul still runs at full N width regardless of the later masking.
- grouped factorization (docs/restart/hz0i_grouped_bdh_results.md):
  shares low-rank factors across head-groups -- real but modest ~9%
  (176.7ms -> 161.0ms), small quality cost, never merged into the live
  untied model.

THIS MODULE'S APPROACH: route BEFORE computing, not mask AFTER. Partition
N into G blocks of width Nb=N/G. A cheap router (rank->G logits, using
the ALREADY-COMPUTED cheap `z` intermediate, O(T*rank*G) cost) picks one
block per (batch, head, token). The Metal kernel then does the expensive
O(T*rank*N) matmul as O(T*rank*Nb) instead -- each thread checks whether
its output column falls in the token's selected block, and only runs the
O(rank) inner-product loop if so (an early return, same pattern E9 uses
for its `dff`-conditioned per-expert loop). This is a REAL FLOP reduction
in the matmul itself, not a post-hoc mask.

HONEST SCOPE: this is a genuine architecture change (top-1 block routing
over N), not a drop-in speedup for the exact existing dense model --
quality impact is NOT yet measured with a real training run (see the
results doc's disclosed gaps). What IS established here: the kernel's
arithmetic is correct against a plain-MLX reference of the identical
routing scheme, and its real measured throughput vs the dense baseline
in `reference/hz0i_bdh_mlx.py`.
"""
from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

from reference.hz0i_bdh_mlx import BDHConfig


@dataclass(frozen=True)
class BlockSparseConfig:
    num_blocks: int = 8  # G; N must be divisible by this


def block_router_logits(z: mx.array, router_w: mx.array) -> mx.array:
    """z: [H, BT, rank], router_w: [H, rank, G] -> logits [H, BT, G]."""
    return mx.einsum("htr,hrg->htg", z, router_w)


def select_blocks(logits: mx.array) -> mx.array:
    """Top-1 block index per (head, token). [H, BT, G] -> [H, BT] int32."""
    return mx.argmax(logits, axis=-1).astype(mx.int32)


# ---------------------------------------------------------------------------
# Plain-MLX reference (correctness oracle for the kernel -- same routing
# scheme, computed with ordinary ops, not a speed path).
# ---------------------------------------------------------------------------

def reference_block_encode(z: mx.array, enc_r: mx.array, block_idx: mx.array, num_blocks: int) -> mx.array:
    """z:[H,BT,rank], enc_r:[H,rank,N], block_idx:[H,BT] -> hidden:[H,BT,N],
    dense but nonzero only within each token's selected block. Computes the
    FULL dense matmul then masks -- this is the reference's job (prove
    correctness), not the kernel's (the kernel must avoid the full matmul)."""
    H, BT, rank = z.shape
    N = enc_r.shape[-1]
    Nb = N // num_blocks
    full = mx.maximum(mx.einsum("htr,hrn->htn", z, enc_r), 0.0)  # [H,BT,N]
    col = mx.arange(N).reshape(1, 1, N) // Nb  # [1,1,N] block-of-column
    mask = (col == block_idx[:, :, None]).astype(full.dtype)
    return full * mask


def reference_block_decode(xy: mx.array, dec_l: mx.array, block_idx: mx.array, num_blocks: int) -> mx.array:
    """xy:[H,BT,N] (block-sparse), dec_l:[H,N,rank], block_idx:[H,BT] ->
    z2:[H,BT,rank]. Full dense contraction over N -- correctness oracle."""
    return mx.einsum("htn,hnr->htr", xy, dec_l)


# ---------------------------------------------------------------------------
# Real Metal kernel: only does the O(rank) inner loop for in-block output
# columns / N-rows -- the FLOP reduction actually happens here.
# ---------------------------------------------------------------------------

_ENCODE_SOURCE = """
    uint token = thread_position_in_grid.x;
    uint n = thread_position_in_grid.y;
    uint bt_v = uint(bt[0]);
    uint rank_v = uint(rank[0]);
    uint nblk_v = uint(n_total[0]);
    uint nb_v = uint(nb[0]);
    if (token >= bt_v * uint(heads[0]) || n >= nblk_v) return;
    uint h = token / bt_v;
    int g = block_idx[token];
    if ((int)(n / nb_v) != g) { hidden[token * nblk_v + n] = 0.0f; return; }
    float acc = 0.0f;
    uint z_base = token * rank_v;
    // enc_r_t is pre-transposed to (H, N, rank): for fixed (h, n), the
    // rank dimension is contiguous, so this loop is a coalesced read
    // instead of a stride-N (36KB) jump per step.
    uint w_base = (h * nblk_v + n) * rank_v;
    for (uint r = 0; r < rank_v; ++r) {
        acc += z[z_base + r] * enc_r_t[w_base + r];
    }
    hidden[token * nblk_v + n] = acc > 0.0f ? acc : 0.0f;
"""

_DECODE_SOURCE = """
    uint token = thread_position_in_grid.x;
    uint r_out = thread_position_in_grid.y;
    uint bt_v = uint(bt[0]);
    uint rank_v = uint(rank[0]);
    uint nblk_v = uint(n_total[0]);
    uint nb_v = uint(nb[0]);
    if (token >= bt_v * uint(heads[0]) || r_out >= rank_v) return;
    uint h = token / bt_v;
    int g = block_idx[token];
    uint n0 = uint(g) * nb_v;
    float acc = 0.0f;
    uint xy_base = token * nblk_v + n0;
    // dec_l_t is pre-transposed to (H, rank, N): for fixed (h, r_out), the
    // N dimension is contiguous, so this loop is a coalesced read instead
    // of a stride-rank jump per step.
    uint w_base = (h * rank_v + r_out) * nblk_v + n0;
    for (uint j = 0; j < nb_v; ++j) {
        acc += xy[xy_base + j] * dec_l_t[w_base + j];
    }
    z2[token * rank_v + r_out] = acc;
"""

_encode_kernel = mx.fast.metal_kernel(
    name="hz0i_block_sparse_encode",
    input_names=["z", "enc_r_t", "block_idx", "bt", "rank", "n_total", "nb", "heads"],
    output_names=["hidden"],
    source=_ENCODE_SOURCE,
)

_decode_kernel = mx.fast.metal_kernel(
    name="hz0i_block_sparse_decode",
    input_names=["xy", "dec_l_t", "block_idx", "bt", "rank", "n_total", "nb", "heads"],
    output_names=["z2"],
    source=_DECODE_SOURCE,
)


def _scalar(v, dtype=mx.int32) -> mx.array:
    return mx.array([v], dtype=dtype)


def kernel_block_encode(z: mx.array, enc_r: mx.array, block_idx: mx.array, num_blocks: int) -> mx.array:
    """z:[H,BT,rank], enc_r:[H,rank,N], block_idx:[H,BT] int32 -> [H,BT,N].
    Real kernel path: the inner O(rank) loop only runs for in-block (token,
    n) pairs -- out-of-block threads return immediately without touching
    enc_r at all. Transposes enc_r to (H,N,rank) first so the kernel's
    inner loop reads contiguous memory (see _ENCODE_SOURCE's comment) --
    this transpose cost is included here, not hidden, since a real caller
    pays it once per layer per step."""
    H, BT, rank = z.shape
    N = enc_r.shape[-1]
    Nb = N // num_blocks
    tokens = H * BT
    z_flat = z.reshape(tokens, rank).astype(mx.float32)
    enc_r_t = mx.transpose(enc_r, (0, 2, 1)).astype(mx.float32)  # (H,rank,N) -> (H,N,rank)
    block_idx_flat = block_idx.reshape(tokens).astype(mx.int32)

    out = _encode_kernel(
        inputs=[z_flat, enc_r_t, block_idx_flat,
                _scalar(BT), _scalar(rank), _scalar(N), _scalar(Nb), _scalar(H)],
        grid=(tokens, N, 1),
        threadgroup=(min(tokens, 32), min(N, 32), 1),
        output_shapes=[(tokens, N)],
        output_dtypes=[mx.float32],
    )[0]
    return out.reshape(H, BT, N)


def kernel_block_decode(xy: mx.array, dec_l: mx.array, block_idx: mx.array, num_blocks: int) -> mx.array:
    """xy:[H,BT,N] (nonzero only in-block), dec_l:[H,N,rank], block_idx:
    [H,BT] -> [H,BT,rank]. Real kernel path: the inner loop only runs over
    the Nb in-block rows of dec_l, not the full N. Transposes dec_l to
    (H,rank,N) first for the same contiguous-read reason as encode."""
    H, BT, N = xy.shape
    rank = dec_l.shape[-1]
    Nb = N // num_blocks
    tokens = H * BT
    xy_flat = xy.reshape(tokens, N).astype(mx.float32)
    dec_l_t = mx.transpose(dec_l, (0, 2, 1)).astype(mx.float32)  # (H,N,rank) -> (H,rank,N)
    block_idx_flat = block_idx.reshape(tokens).astype(mx.int32)

    out = _decode_kernel(
        inputs=[xy_flat, dec_l_t, block_idx_flat,
                _scalar(BT), _scalar(rank), _scalar(N), _scalar(Nb), _scalar(H)],
        grid=(tokens, rank, 1),
        threadgroup=(min(tokens, 32), min(rank, 32), 1),
        output_shapes=[(tokens, rank)],
        output_dtypes=[mx.float32],
    )[0]
    return out.reshape(H, BT, rank)
