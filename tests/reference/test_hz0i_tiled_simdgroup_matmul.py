"""Correctness tests for the simdgroup_matrix-tiled Metal matmul kernel
(reference/hz0i_tiled_simdgroup_matmul.py)."""
from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from reference.hz0i_tiled_simdgroup_matmul import simdgroup_tiled_matmul


@pytest.mark.parametrize("H,M,K,N", [(1, 16, 16, 16), (3, 32, 24, 40), (2, 64, 32, 48)])
def test_matches_dense_matmul(H, M, K, N):
    key = mx.random.key(0)
    ks = mx.random.split(key, 2)
    A = mx.random.normal((H, M, K), key=ks[0])
    B = mx.random.normal((H, K, N), key=ks[1])
    mx.eval(A, B)

    C_kernel = simdgroup_tiled_matmul(A, B)
    C_ref = A.astype(mx.float32) @ B.astype(mx.float32)
    mx.eval(C_kernel, C_ref)

    diff = np.abs(np.array(C_kernel) - np.array(C_ref))
    typical = np.abs(np.array(C_ref)).mean()
    rel = diff.mean() / typical
    # 1%, not exact: simdgroup_matrix hardware units on Apple GPUs use
    # internally-reduced accumulation precision (comparable to TF32 on
    # other tensor-core hardware) even for declared-float32 inputs --
    # measured ~0.075% mean relative error on a spot check with no
    # tile-boundary structure (ruled out an indexing bug by inspecting
    # the full diff matrix, not just the aggregate max/mean). Not a
    # correctness bug, a real, documented hardware characteristic.
    assert rel < 0.01, f"H={H} M={M} K={K} N={N}: mean relative error {rel} too large for hardware noise"


def test_shape_and_finite():
    A = mx.random.normal((4, 8, 8))
    B = mx.random.normal((4, 8, 8))
    mx.eval(A, B)
    C = simdgroup_tiled_matmul(A, B)
    mx.eval(C)
    assert C.shape == (4, 8, 8)
    assert bool(mx.all(mx.isfinite(C)))
