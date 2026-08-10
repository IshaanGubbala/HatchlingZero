"""HZ-0I: a real tiled Metal matmul kernel using `simdgroup_matrix`
intrinsics, following up on the failed naive per-element kernel
(docs/restart/hz0i_block_sparse_kernel_results.md's attempt 1, which was
12x SLOWER than dense because it did one scalar multiply-accumulate per
thread with no use of the GPU's matrix-multiply hardware).

This uses `simdgroup_float8x8` + `simdgroup_multiply_accumulate` (the same
class of hardware instruction Apple's own tuned GEMM uses) so each thread
GROUP (32 threads, one simdgroup) computes an 8x8 output tile per K-chunk,
instead of one scalar per thread. Batched over a leading `H` (head)
dimension via the grid's z-coordinate.

HONEST SCOPE: no threadgroup-memory staging/double-buffering (each
simdgroup reads A/B tiles directly from device memory every K-chunk,
rather than staging a larger block in shared memory for reuse across
multiple simdgroups) -- a real, disclosed simplification. Whether this
alone gets close enough to vendor GEMM to be worth adding block-routing
on top is the real question this module's benchmark answers.
"""
from __future__ import annotations

import mlx.core as mx

_MATMUL_SOURCE = """
    uint tile_m = thread_position_in_grid.x / 32;
    uint tile_n = thread_position_in_grid.y;
    uint head = thread_position_in_grid.z;
    uint M_v = uint(dims[0]);
    uint K_v = uint(dims[1]);
    uint N_v = uint(dims[2]);
    uint num_tiles_m = M_v / 8;
    uint num_tiles_n = N_v / 8;
    uint num_k_tiles = K_v / 8;
    if (tile_m >= num_tiles_m || tile_n >= num_tiles_n) return;

    device const float* A_head = A + head * M_v * K_v;
    device const float* B_head = B + head * K_v * N_v;
    device float* C_head = C + head * M_v * N_v;

    metal::simdgroup_float8x8 c_tile = metal::simdgroup_float8x8(0.0);
    for (uint kt = 0; kt < num_k_tiles; ++kt) {
        metal::simdgroup_float8x8 a_tile, b_tile;
        metal::simdgroup_load(a_tile, A_head + tile_m * 8 * K_v + kt * 8, K_v);
        metal::simdgroup_load(b_tile, B_head + kt * 8 * N_v + tile_n * 8, N_v);
        metal::simdgroup_multiply_accumulate(c_tile, a_tile, b_tile, c_tile);
    }
    metal::simdgroup_store(c_tile, C_head + tile_m * 8 * N_v + tile_n * 8, N_v);
"""

_matmul_kernel = mx.fast.metal_kernel(
    name="hz0i_tiled_simdgroup_matmul",
    input_names=["A", "B", "dims"],
    output_names=["C"],
    source=_MATMUL_SOURCE,
)


def simdgroup_tiled_matmul(A: mx.array, B: mx.array) -> mx.array:
    """A:[H,M,K], B:[H,K,N] -> C:[H,M,N] = A @ B (per head), via a real
    simdgroup_matrix tiled Metal kernel. M, K, N must all be multiples of 8
    (required by the 8x8 simdgroup tile size; not padded/handled here)."""
    H, M, K = A.shape
    _, K2, N = B.shape
    assert K == K2, (K, K2)
    assert M % 8 == 0 and K % 8 == 0 and N % 8 == 0, (M, K, N)

    A_f = A.astype(mx.float32)
    B_f = B.astype(mx.float32)
    dims = mx.array([M, K, N], dtype=mx.int32)

    num_tiles_m = M // 8
    num_tiles_n = N // 8

    out = _matmul_kernel(
        inputs=[A_f, B_f, dims],
        grid=(num_tiles_m * 32, num_tiles_n, H),
        threadgroup=(32, 1, 1),
        output_shapes=[(H, M, N)],
        output_dtypes=[mx.float32],
    )[0]
    return out
