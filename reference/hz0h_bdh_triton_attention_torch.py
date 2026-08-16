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
        for key_start in tl.range(0, T, BLOCK_K):
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
                scores += tl.dot(q, tl.trans(k))

            scores = tl.where(causal & key_mask[None, :], scores, 0.0)
            v = tl.load(
                v_ptr + key_rows[:, None] * v_stride_t + d_cols[None, :] * v_stride_d,
                mask=key_mask[:, None] & d_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            acc += tl.sum(scores[:, :, None] * v[None, :, :], axis=1)

        tl.store(
            out_ptr + q_rows[:, None] * o_stride_t + d_cols[None, :] * o_stride_d,
            acc,
            mask=q_mask[:, None] & d_mask[None, :],
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
    grid = (B * nh, triton.cdiv(T, block_m), triton.cdiv(D, block_d))
    _bdh_forward_kernel[grid](
        q, v, out, T, N, D, nh,
        q.stride(1), q.stride(2), q.stride(3),
        v.stride(0), v.stride(1), v.stride(2),
        out.stride(1), out.stride(2), out.stride(3),
        BLOCK_M=block_m, BLOCK_K=block_k, BLOCK_N=block_n, BLOCK_D=block_d,
    )
    return out


class _BDHTritonAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, QR: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        out = _triton_forward(QR, V)
        ctx.save_for_backward(QR, V)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # Keep the exact analytic derivative shared with the reference path.
        # This is intentionally explicit until a second Triton backward kernel
        # is parity-tested; it never delegates gradient computation to the
        # oracle's autograd graph.
        QR, V = ctx.saved_tensors
        B, nh, T, N = QR.shape
        D = V.shape[-1]
        q = QR.reshape(B * nh, T, N)
        v = V.expand(B, nh, T, D).reshape(B * nh, T, D)
        dy = grad_output.reshape(B * nh, T, D)
        dQ = torch.zeros_like(q)
        dV_heads = torch.zeros_like(v)
        chunk_size = 32
        for start in range(0, T, chunk_size):
            stop = min(T, start + chunk_size)
            q_chunk = q[:, start:stop, :]
            score = torch.matmul(q_chunk, q.transpose(1, 2)).tril(diagonal=start - 1)
            dscore = torch.matmul(dy[:, start:stop, :], v.transpose(1, 2)).tril(diagonal=start - 1)
            dQ[:, start:stop, :] += torch.matmul(dscore, q)
            dQ += torch.matmul(dscore.transpose(1, 2), q_chunk)
            dV_heads += torch.matmul(score.transpose(1, 2), dy[:, start:stop, :])
        dV = dV_heads.reshape(B, nh, T, D)
        dV = dV if V.shape[1] == nh else dV.sum(dim=1, keepdim=True)
        return dQ.reshape(B, nh, T, N), dV


def bdh_triton_attention(Q: torch.Tensor, V: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Exact BDH attention through the compiled Triton forward kernel."""
    B, _, T, _ = Q.shape
    r_phases = torch.arange(0, T, device=freqs.device, dtype=freqs.dtype).view(1, 1, T, 1) * freqs
    QR = Attention.rope(r_phases, Q)
    if not triton_available():
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
