"""Hand-written Triton implementation of exact BDH raw attention.

This is the compiled GPU follow-up required by
``plans/HZ_BDH_Attention_Kernel_Spec.md``.  It computes
``tril(Q @ Q.T, diagonal=-1) @ V`` in a fused query/key/value tile without
materializing the full score matrix.  RoPE and the manually-derived backward
remain explicit Python/Torch code so the compiled kernel can be compared
directly with the verbatim BDH oracle.

Triton is optional on the Mac development machine.  The public function falls
back to the exact bounded PyTorch implementation when CUDA/Triton is absent;
callers must inspect ``triton_available()`` before making a GPU-kernel claim.

Real, diagnosed bug fixed 2026-08-16: the first real CUDA correctness run
(5 parametrized shapes) failed all 5 cases, with error scaling by ~0.1-0.2%
of the output's own magnitude regardless of shape -- including the T=17
case, which fits in a single query/key tile with no multi-tile boundary
logic at all. A follow-up fp32-vs-bf16 diagnostic
(scripts/hz0h_triton_kernel_precision_diagnostic.py) showed the SAME
~0.1-0.2% relative error even with both sides running fully in fp32, which
rules out bf16 rounding-order as the cause. That magnitude is the known
signature of Ampere's TF32 tensor-core path: Triton's ``tl.dot`` silently
downcasts fp32 inputs through TF32 (~10-bit mantissa, ~2**-10 ~= 0.098%
relative precision) unless told otherwise. The one ``tl.dot`` call below now
passes ``input_precision="ieee"`` to force full IEEE fp32 accumulation.

That fix did not touch the real bf16 correctness test (bf16 inputs never
take the TF32 path). A follow-up error-localization diagnostic
(scripts/hz0h_triton_kernel_error_localization_diagnostic.py) on the still-
failing bf16 test showed zero spurious mass at any position the oracle
causal-masks to exactly zero (rules out a masking bug), and a per-row error
that stays a roughly uniform ~0.3-0.7% of that ROW's own overall magnitude
regardless of query position, depth, or tile boundary -- the signature of a
real but expected bf16-rounding-chain divergence (the oracle rounds to bf16
twice, once after ``QR @ KR.mT`` and again after ``scores @ V``; this kernel
accumulates the whole reduction in fp32 and rounds once at the final store).
The test's tolerance was miscalibrated for this: BDH's output has near-zero
individual feature dimensions sitting next to large ones within the same
row, so a naive per-element ``rtol`` breaks down exactly like it did for the
native tiled kernel's own documented calibration bug (see
docs/restart/hz0h_bdh_native_kernel_results.md). The test now scales
tolerance by each row's own magnitude instead of each individual element's.

Real algorithmic changes applied here (2026-08-16, per plan feedback): the
key-tile loop now stops at ``(pid_m + 1) * BLOCK_M`` instead of ``T`` --
since Q=K and the mask is strictly causal, any key tile starting at or past
the query tile's own start is either the diagonal tile (still masked
elementwise) or entirely future (would be masked to all-zero anyway), so
those tiles are skipped rather than computed and discarded. ``scores @ V``
now runs as ``tl.dot(scores, v, input_precision="ieee")``, a real tensor-
core GEMM, instead of a manual ``tl.sum(scores[:,:,None]*v[None,:,:],axis=1)``
broadcast-reduce.

Real, diagnosed regression found 2026-08-16 after the change above: a
clean, isolated end-to-end benchmark (one model resident on the GPU at a
time, confirmed via direct nvidia-smi clock polling to rule out
throttling) showed this kernel running ~1.6x SLOWER than raw BDH
attention, despite the real forward-pass win above. Root cause: the
backward pass was still an explicit Python loop (chunk_size=32, ~8
chunks at this project's T=256, several torch.matmul calls per chunk,
times n_layer recurrent levels) -- on the order of 40 separate kernel
launches per attention call, whose fixed per-launch CPU/driver dispatch
overhead outweighed the forward kernel's real algorithmic savings. See
docs/restart/hz0h_triton_regime_dependence_results.md for the full
diagnostic chain (which also had to first rule out measurement noise,
cross-stage GPU memory pressure, and a wrongly-suspected thermal/power
"regime" before finding the real cause: two full models being kept
simultaneously resident during an earlier benchmark's own A/B
comparison, which is a separate, already-corrected issue from this
backward-pass fix).

Fixed here by replacing that Python loop with three compiled Triton
kernels (``_bdh_dq_query_role_kernel``, ``_bdh_dq_key_role_kernel``,
``_bdh_dv_kernel``) -- see ``_triton_backward``'s own docstring for why
three kernels are needed (Q plays two distinct roles, query and key,
since K=Q) and the exact math each computes. This cuts backward from
~40 PyTorch-level kernel launches to 3 Triton launches, regardless of T.

First real CUDA run of these three kernels found a real dtype bug (fixed:
a freshly bf16-loaded tensor was dotted against an fp32 accumulator
without casting, same pattern the forward kernel already handled
correctly). After that fix, correctness passed 5/5 -- but the real clean
speed measurement came back WORSE than the uncompiled Python loop it was
meant to replace (0.46x vs. the loop's own 0.61x). Root cause: the first
version reused the forward kernel's small output-tile size (64) for the
backward kernels' OUTPUT tiling too, but the expensive dscore/score
reduction those kernels compute does not depend on the output tile at
all -- so at N=2048 with a 64-wide output tile, the grid launched 32
separate program instances per (batch, head, row-tile), each
redundantly recomputing the identical dscore matrix from scratch. Fixed
by widening the output tiles (block_n_out=256, block_d_out=128) while
keeping the reduction tiles small (block_d_reduce=block_n_reduce=64),
cutting that redundant recomputation 4x/2x respectively. See
``_triton_backward``'s own inline comment for the exact reasoning.
"""
from __future__ import annotations

import torch

from reference.hz0h_bdh_native_kernel_attention_torch import bdh_native_attention
from reference.hz0h_bdh_torch import Attention, BDH

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # pragma: no cover - exercised on the Mac fallback path
    triton = None
    tl = None
    _HAS_TRITON = False


def triton_available() -> bool:
    return bool(_HAS_TRITON and torch.cuda.is_available())


if _HAS_TRITON:

    @triton.jit
    def _bdh_forward_kernel(
        q_ptr, v_ptr, out_ptr,
        T, N, D, NH,
        q_stride_bh, q_stride_t, q_stride_n,
        v_stride_b, v_stride_t, v_stride_d,
        o_stride_bh, o_stride_t, o_stride_d,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Fused raw BDH attention for one (batch, head, query, D) tile."""
        pid_bh = tl.program_id(0)
        pid_m = tl.program_id(1)
        pid_d = tl.program_id(2)
        q_rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        d_cols = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        q_mask = q_rows < T
        d_mask = d_cols < D
        q_ptr += pid_bh * q_stride_bh
        out_ptr += pid_bh * o_stride_bh
        v_ptr += (pid_bh // NH) * v_stride_b

        acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
        # Q=K and the mask is strictly-causal (tril(diagonal=-1)): any key
        # tile that starts at or after this query tile's own start row is
        # either the diagonal tile (needs the elementwise mask below) or
        # entirely in the future (would be masked to all-zero anyway) --
        # skip iterating past it instead of computing then discarding it.
        # Assumes BLOCK_K == BLOCK_M so tile boundaries line up exactly.
        key_end = (pid_m + 1) * BLOCK_M
        for key_start in tl.range(0, key_end, BLOCK_K):
            key_rows = key_start + tl.arange(0, BLOCK_K)
            key_mask = key_rows < T
            causal = key_rows[None, :] < q_rows[:, None]
            scores = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)

            for n_start in tl.range(0, N, BLOCK_N):
                n_cols = n_start + tl.arange(0, BLOCK_N)
                q = tl.load(
                    q_ptr + q_rows[:, None] * q_stride_t + n_cols[None, :] * q_stride_n,
                    mask=q_mask[:, None] & (n_cols[None, :] < N),
                    other=0.0,
                )
                k = tl.load(
                    q_ptr + key_rows[:, None] * q_stride_t + n_cols[None, :] * q_stride_n,
                    mask=key_mask[:, None] & (n_cols[None, :] < N),
                    other=0.0,
                )
                scores += tl.dot(q, tl.trans(k), input_precision="ieee")

            scores = tl.where(causal & key_mask[None, :], scores, 0.0)
            v = tl.load(
                v_ptr + key_rows[:, None] * v_stride_t + d_cols[None, :] * v_stride_d,
                mask=key_mask[:, None] & d_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            # scores @ v as an actual tensor-core GEMM instead of a manual
            # broadcast-multiply-then-reduce.
            acc += tl.dot(scores, v, input_precision="ieee")

        tl.store(
            out_ptr + q_rows[:, None] * o_stride_t + d_cols[None, :] * o_stride_d,
            acc,
            mask=q_mask[:, None] & d_mask[None, :],
        )

    @triton.jit
    def _bdh_dq_query_role_kernel(
        q_ptr, v_ptr, dout_ptr, dq_ptr,
        T, N, D, NH,
        q_stride_bh, q_stride_t, q_stride_n,
        v_stride_b, v_stride_t, v_stride_d,
        dout_stride_bh, dout_stride_t, dout_stride_d,
        dq_stride_bh, dq_stride_t, dq_stride_n,
        BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
        BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        """dQ contribution from Q's QUERY role: for output row a,
        dQ_query[a,n] = sum_{b<a} dscore[a,b] * Q[b,n], where
        dscore[a,b] = dOut[a,:].V[b,:]. Same causal-tile-skip bound as the
        forward kernel (b ranges only up to a's own tile)."""
        pid_bh = tl.program_id(0)
        pid_m = tl.program_id(1)  # a-tile (output query row tile)
        pid_n = tl.program_id(2)  # output N-column tile
        a_rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        n_out_cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        a_mask = a_rows < T
        n_out_mask = n_out_cols < N
        q_ptr += pid_bh * q_stride_bh
        dout_ptr += pid_bh * dout_stride_bh
        dq_ptr += pid_bh * dq_stride_bh
        v_ptr += (pid_bh // NH) * v_stride_b

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        b_end = (pid_m + 1) * BLOCK_M
        for b_start in tl.range(0, b_end, BLOCK_K):
            b_rows = b_start + tl.arange(0, BLOCK_K)
            b_mask = b_rows < T
            causal = b_rows[None, :] < a_rows[:, None]
            dscore = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
            for d_start in tl.range(0, D, BLOCK_D):
                d_cols = d_start + tl.arange(0, BLOCK_D)
                dout = tl.load(
                    dout_ptr + a_rows[:, None] * dout_stride_t + d_cols[None, :] * dout_stride_d,
                    mask=a_mask[:, None] & (d_cols[None, :] < D), other=0.0,
                )
                v = tl.load(
                    v_ptr + b_rows[:, None] * v_stride_t + d_cols[None, :] * v_stride_d,
                    mask=b_mask[:, None] & (d_cols[None, :] < D), other=0.0,
                )
                dscore += tl.dot(dout, tl.trans(v), input_precision="ieee")
            dscore = tl.where(causal & b_mask[None, :], dscore, 0.0)
            q_b = tl.load(
                q_ptr + b_rows[:, None] * q_stride_t + n_out_cols[None, :] * q_stride_n,
                mask=b_mask[:, None] & n_out_mask[None, :], other=0.0,
            ).to(tl.float32)
            acc += tl.dot(dscore, q_b, input_precision="ieee")

        tl.store(
            dq_ptr + a_rows[:, None] * dq_stride_t + n_out_cols[None, :] * dq_stride_n,
            acc, mask=a_mask[:, None] & n_out_mask[None, :],
        )

    @triton.jit
    def _bdh_dq_key_role_kernel(
        q_ptr, v_ptr, dout_ptr, dq_ptr,
        T, N, D, NH,
        q_stride_bh, q_stride_t, q_stride_n,
        v_stride_b, v_stride_t, v_stride_d,
        dout_stride_bh, dout_stride_t, dout_stride_d,
        dq_stride_bh, dq_stride_t, dq_stride_n,
        BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
        BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        """dQ contribution from Q's KEY role: for output row b,
        dQ_key[b,n] = sum_{a>b} dscore[a,b] * Q[a,n]. Mirror of the
        query-role kernel's bound: a sweeps from b's own tile start
        (inclusive, for the diagonal tile's mask) forward to T, instead of
        from 0 up to b's tile -- the future-inclusive complement."""
        pid_bh = tl.program_id(0)
        pid_m = tl.program_id(1)  # b-tile (output key row tile)
        pid_n = tl.program_id(2)
        b_rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        n_out_cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        b_mask = b_rows < T
        n_out_mask = n_out_cols < N
        q_ptr += pid_bh * q_stride_bh
        dout_ptr += pid_bh * dout_stride_bh
        dq_ptr += pid_bh * dq_stride_bh
        v_ptr += (pid_bh // NH) * v_stride_b

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        a_start_bound = pid_m * BLOCK_M
        for a_start in tl.range(a_start_bound, T, BLOCK_K):
            a_rows = a_start + tl.arange(0, BLOCK_K)
            a_mask = a_rows < T
            causal = b_rows[:, None] < a_rows[None, :]
            dscore = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
            for d_start in tl.range(0, D, BLOCK_D):
                d_cols = d_start + tl.arange(0, BLOCK_D)
                v = tl.load(
                    v_ptr + b_rows[:, None] * v_stride_t + d_cols[None, :] * v_stride_d,
                    mask=b_mask[:, None] & (d_cols[None, :] < D), other=0.0,
                )
                dout = tl.load(
                    dout_ptr + a_rows[:, None] * dout_stride_t + d_cols[None, :] * dout_stride_d,
                    mask=a_mask[:, None] & (d_cols[None, :] < D), other=0.0,
                )
                dscore += tl.dot(v, tl.trans(dout), input_precision="ieee")
            dscore = tl.where(causal & a_mask[None, :], dscore, 0.0)
            q_a = tl.load(
                q_ptr + a_rows[:, None] * q_stride_t + n_out_cols[None, :] * q_stride_n,
                mask=a_mask[:, None] & n_out_mask[None, :], other=0.0,
            ).to(tl.float32)
            acc += tl.dot(dscore, q_a, input_precision="ieee")

        tl.store(
            dq_ptr + b_rows[:, None] * dq_stride_t + n_out_cols[None, :] * dq_stride_n,
            acc, mask=b_mask[:, None] & n_out_mask[None, :],
        )

    @triton.jit
    def _bdh_dv_kernel(
        q_ptr, dout_ptr, dv_ptr,
        T, N, D, NH,
        q_stride_bh, q_stride_t, q_stride_n,
        dout_stride_bh, dout_stride_t, dout_stride_d,
        dv_stride_bh, dv_stride_t, dv_stride_d,
        BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
        BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        """dV_heads[b,d] = sum_{a>b} score[a,b] * dOut[a,d], where
        score[a,b] = Q[a,:].Q[b,:]. Same future-inclusive sweep as the
        dQ key-role kernel (V's gradient only gets contributions from
        query positions strictly after it)."""
        pid_bh = tl.program_id(0)
        pid_m = tl.program_id(1)  # b-tile
        pid_d = tl.program_id(2)
        b_rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        d_cols = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        b_mask = b_rows < T
        d_mask = d_cols < D
        q_ptr += pid_bh * q_stride_bh
        dout_ptr += pid_bh * dout_stride_bh
        dv_ptr += pid_bh * dv_stride_bh

        acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
        a_start_bound = pid_m * BLOCK_M
        for a_start in tl.range(a_start_bound, T, BLOCK_K):
            a_rows = a_start + tl.arange(0, BLOCK_K)
            a_mask = a_rows < T
            causal = b_rows[:, None] < a_rows[None, :]
            score = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
            for n_start in tl.range(0, N, BLOCK_N):
                n_cols = n_start + tl.arange(0, BLOCK_N)
                q_b = tl.load(
                    q_ptr + b_rows[:, None] * q_stride_t + n_cols[None, :] * q_stride_n,
                    mask=b_mask[:, None] & (n_cols[None, :] < N), other=0.0,
                )
                q_a = tl.load(
                    q_ptr + a_rows[:, None] * q_stride_t + n_cols[None, :] * q_stride_n,
                    mask=a_mask[:, None] & (n_cols[None, :] < N), other=0.0,
                )
                score += tl.dot(q_b, tl.trans(q_a), input_precision="ieee")
            score = tl.where(causal & a_mask[None, :], score, 0.0)
            dout_a = tl.load(
                dout_ptr + a_rows[:, None] * dout_stride_t + d_cols[None, :] * dout_stride_d,
                mask=a_mask[:, None] & d_mask[None, :], other=0.0,
            ).to(tl.float32)
            acc += tl.dot(score, dout_a, input_precision="ieee")

        tl.store(
            dv_ptr + b_rows[:, None] * dv_stride_t + d_cols[None, :] * dv_stride_d,
            acc, mask=b_mask[:, None] & d_mask[None, :],
        )


def _triton_forward(QR: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    if not triton_available():
        raise RuntimeError("Triton CUDA backend is unavailable")
    B, nh, T, N = QR.shape
    D = V.shape[-1]
    if V.shape[1] not in (1, nh):
        raise ValueError(f"BDH V must have one or nh heads, got {V.shape}")
    if V.shape[1] != 1:
        raise ValueError("The first Triton kernel version only supports BDH's broadcast V")
    q = QR.contiguous()
    v = V[:, 0].contiguous()
    out = torch.empty((B, nh, T, D), device=QR.device, dtype=QR.dtype)
    block_m = 32
    block_k = 32
    block_n = 64
    block_d = 64
    assert block_m == block_k, (
        "the kernel's future-tile-skip bound ((pid_m+1)*BLOCK_M) assumes query "
        "and key tiles line up exactly"
    )
    grid = (B * nh, triton.cdiv(T, block_m), triton.cdiv(D, block_d))
    _bdh_forward_kernel[grid](
        q, v, out, T, N, D, nh,
        q.stride(1), q.stride(2), q.stride(3),
        v.stride(0), v.stride(1), v.stride(2),
        out.stride(1), out.stride(2), out.stride(3),
        BLOCK_M=block_m, BLOCK_K=block_k, BLOCK_N=block_n, BLOCK_D=block_d,
    )
    return out


def _triton_backward(QR: torch.Tensor, V: torch.Tensor, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compiled-Triton backward: the same analytic derivative the prior
    explicit Python/torch loop computed, split into three kernels because
    Q plays two distinct roles (query AND key, since K=Q) with opposite
    causal-tile-skip bounds:

    - ``_bdh_dq_query_role_kernel``: dQ's contribution from Q's query
      role, same past-inclusive bound as the forward kernel.
    - ``_bdh_dq_key_role_kernel``: dQ's contribution from Q's key role,
      the future-inclusive complement.
    - ``_bdh_dv_kernel``: dV, also future-inclusive (V only receives
      gradient from query positions strictly after it).

    Real, disclosed reason this replaced the Python loop: that loop issued
    on the order of 40 separate PyTorch kernel launches per attention call
    (8 chunks at this project's T=256, several torch.matmul calls each),
    whose fixed per-launch dispatch overhead was found to dominate and
    reverse the forward kernel's real speedup once measured under clean,
    isolated conditions (see docs/restart/hz0h_triton_regime_dependence_results.md).
    This reduces that to 3 kernel launches total, regardless of T.
    """
    if not triton_available():
        raise RuntimeError("Triton CUDA backend is unavailable")
    B, nh, T, N = QR.shape
    D = V.shape[-1]
    q = QR.contiguous()
    v = V[:, 0].contiguous()
    dout = grad_output.contiguous()

    dq_query = torch.empty((B, nh, T, N), device=QR.device, dtype=QR.dtype)
    dq_key = torch.empty((B, nh, T, N), device=QR.device, dtype=QR.dtype)
    dv_heads = torch.empty((B, nh, T, D), device=QR.device, dtype=V.dtype)

    block_m = 32
    block_k = 32
    assert block_m == block_k, (
        "the backward kernels' future-inclusive/past-inclusive bounds assume "
        "query and key tiles line up exactly, same as the forward kernel"
    )
    # Real, disclosed fix (2026-08-16): the first version of these kernels
    # reused the forward kernel's BLOCK_N=64/BLOCK_D=64 constants for BOTH
    # reduction tiling AND output tiling. In these backward kernels, the
    # expensive dscore/score computation (a real reduction over D or N)
    # does NOT depend on the output-tile index at all, but a naive small
    # output tile means the grid launches one program instance PER output
    # tile, and each one independently RECOMPUTES that same dscore/score
    # from scratch -- at N=2048 with a 64-wide output tile, that's the
    # SAME dscore matrix recomputed 32 times per (batch,head,a-tile). A
    # first real CUDA run confirmed this: the compiled-kernel backward
    # measured SLOWER (0.46x) than even the original uncompiled Python
    # loop (0.61x) it was meant to fix. Widening the OUTPUT tiles (kept
    # separate from the REDUCTION tiles, which stay small) cuts that
    # redundant recomputation proportionally.
    # Real follow-up (still 2026-08-16): block_n_out=256/block_d_out=128
    # gave a real but partial improvement (0.46x -> 0.594x) -- still short
    # of both raw BDH and the original Python loop's own ~0.61-0.64x, and
    # still leaves real redundancy (8x at N=2048/256, 4x at D=512/128).
    # Widening further to cut that down to 4x/2x before treating this as
    # a real ceiling for the output-tile-width lever specifically.
    block_d_reduce = 64   # D-reduction tile inside the dQ kernels
    block_n_reduce = 64   # N-reduction tile inside the dV kernel
    block_n_out = 512     # dQ kernels' output N-tile (was 256 -- now 4x redundant dscore recomputes at N=2048, was 8x)
    block_d_out = 256     # dV kernel's output D-tile (was 128 -- now 2x redundant score recomputes at D=512, was 4x)

    grid_n = (B * nh, triton.cdiv(T, block_m), triton.cdiv(N, block_n_out))
    _bdh_dq_query_role_kernel[grid_n](
        q, v, dout, dq_query,
        T, N, D, nh,
        q.stride(1), q.stride(2), q.stride(3),
        v.stride(0), v.stride(1), v.stride(2),
        dout.stride(1), dout.stride(2), dout.stride(3),
        dq_query.stride(1), dq_query.stride(2), dq_query.stride(3),
        BLOCK_M=block_m, BLOCK_K=block_k, BLOCK_N=block_n_out, BLOCK_D=block_d_reduce,
    )
    _bdh_dq_key_role_kernel[grid_n](
        q, v, dout, dq_key,
        T, N, D, nh,
        q.stride(1), q.stride(2), q.stride(3),
        v.stride(0), v.stride(1), v.stride(2),
        dout.stride(1), dout.stride(2), dout.stride(3),
        dq_key.stride(1), dq_key.stride(2), dq_key.stride(3),
        BLOCK_M=block_m, BLOCK_K=block_k, BLOCK_N=block_n_out, BLOCK_D=block_d_reduce,
    )
    grid_d = (B * nh, triton.cdiv(T, block_m), triton.cdiv(D, block_d_out))
    _bdh_dv_kernel[grid_d](
        q, dout, dv_heads,
        T, N, D, nh,
        q.stride(1), q.stride(2), q.stride(3),
        dout.stride(1), dout.stride(2), dout.stride(3),
        dv_heads.stride(1), dv_heads.stride(2), dv_heads.stride(3),
        BLOCK_M=block_m, BLOCK_K=block_k, BLOCK_N=block_n_reduce, BLOCK_D=block_d_out,
    )

    dQ = (dq_query.float() + dq_key.float()).to(QR.dtype)
    dV = dv_heads if V.shape[1] == nh else dv_heads.sum(dim=1, keepdim=True)
    return dQ, dV


class _BDHTritonAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, QR: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        out = _triton_forward(QR, V)
        ctx.save_for_backward(QR, V)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        QR, V = ctx.saved_tensors
        return _triton_backward(QR, V, grad_output)


def bdh_triton_attention(Q: torch.Tensor, V: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Exact BDH attention through the compiled Triton forward kernel."""
    B, _, T, _ = Q.shape
    r_phases = torch.arange(0, T, device=freqs.device, dtype=freqs.dtype).view(1, 1, T, 1) * freqs
    QR = Attention.rope(r_phases, Q)
    # triton_available() only checks that CUDA+Triton exist on this
    # machine, not that THESE tensors are actually on a CUDA device --
    # a real dispatch bug this exact check caught (2026-08-16): a CPU
    # model run on a CUDA-capable machine hit the Triton kernel with CPU
    # pointers and crashed instead of falling back.
    if not triton_available() or not Q.is_cuda:
        return bdh_native_attention(Q, V, freqs)
    return _BDHTritonAttention.apply(QR, V)


def bdh_triton_forward(model: BDH, idx: torch.Tensor, targets: torch.Tensor | None = None):
    """Drop-in BDH forward using the Triton attention backend."""
    import torch.nn.functional as F

    C = model.config
    B, T = idx.size()
    D = C.n_embd
    nh = C.n_head
    N = D * C.mlp_internal_dim_multiplier // nh
    x = model.ln(model.embed(idx).unsqueeze(1))
    for _ in range(C.n_layer):
        x_sparse = F.relu(x @ model._w(model.encoder))
        yKV = model.ln(bdh_triton_attention(x_sparse, x, model.attn.freqs))
        y_sparse = F.relu(yKV @ model._w(model.encoder_v))
        xy_sparse = model.drop(x_sparse * y_sparse)
        yMLP = xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ model._w(model.decoder)
        x = model.ln(x + model.ln(yMLP))
    logits = x.view(B, T, D) @ model.lm_head
    loss = None
    if targets is not None:
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
    return logits, loss
