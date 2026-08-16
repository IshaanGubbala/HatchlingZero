"""Real GPU-layout remap for BDH's encoder_v projection -- Stage 1A/1B of
plans/hatchlingzero_bdh_transformer_planning.md ("do something similar to
encoder_v").

The oracle (reference/hz0h_bdh_torch.py, never modified) computes each
recurrent level's second projection as a broadcasted per-head matmul:

    y_latent = yKV @ encoder_v   # yKV:(B,nh,T,D), encoder_v:(nh,D,N) -> (B,nh,T,N)

Unlike the encoder projection (reference/hz0h_bdh_wide_gemm_encoder_torch.py),
this one canNOT become a single GEMM: every head's input `yKV[:, h]` is
genuinely different (it is the head's own attention output), not the same
tokens broadcast across heads. But it can still become one explicit batched
GEMM with head as the batch dimension -- `torch.bmm` over `(nh, B*T, D) x
(nh, D, N)` -- instead of leaving the batching decision to an implicit
PyTorch broadcast. Each GEMM in the batch then has `M = B*T` rather than
being executed as a smaller `T x D` operation, per head, under a broadcast.
Zero change to BDH's math -- purely an execution-layout claim, meant to be
falsified or confirmed by a real parity test and a real GPU timing
comparison, not assumed correct or assumed faster (cuBLAS may already
partially optimize the existing broadcast expression -- that is exactly
the open question this experiment answers).
"""
from __future__ import annotations

import torch


def bmm_encoder_v_step(yKV: torch.Tensor, encoder_v: torch.Tensor) -> torch.Tensor:
    """Compute one recurrent level's ``y_latent = yKV @ encoder_v`` as an
    explicit per-head batched GEMM (``torch.bmm``, head as the batch dim)
    instead of relying on implicit broadcast layout.

    yKV: ``(B, nh, T, D)``. encoder_v: ``(nh, D, N)``. Returns
    ``(B, nh, T, N)``, matching the oracle's own broadcasted matmul output
    exactly.
    """
    B, nh, T, D = yKV.shape
    nh_w, D_w, N = encoder_v.shape
    assert nh == nh_w and D == D_w, (
        f"yKV heads/dim {(nh, D)} must match encoder_v's {(nh_w, D_w)}"
    )
    y = yKV.permute(1, 0, 2, 3).reshape(nh, B * T, D)  # (nh, B*T, D)
    out = torch.bmm(y, encoder_v)  # (nh, B*T, N), one batched GEMM
    return out.reshape(nh, B, T, N).permute(1, 0, 2, 3)  # (B, nh, T, N)
