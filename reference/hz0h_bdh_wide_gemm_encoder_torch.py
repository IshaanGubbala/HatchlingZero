"""Real GPU-layout remap for BDH's encoder projection -- Stage 1A of
plans/hatchlingzero_bdh_transformer_planning.md ("Re-express head-wise
projections as fewer larger batched matrix multiplies").

The oracle (reference/hz0h_bdh_torch.py, never modified) computes each
recurrent level's first projection as a broadcasted per-head matmul:

    x_latent = x @ encoder        # x:(B,1,T,D), encoder:(nh,D,N) -> (B,nh,T,N)

PyTorch/cuBLAS execute this as effectively ``nh`` separate ``(T,D)x(D,N)``
GEMMs under a broadcast, not one big regular GEMM. This module provides a
mathematically-identical remap: reshape ``encoder`` into a GPU-native
``D x (H*N)`` matrix and the tokens into ``(B*T, D)``, so the whole
per-head projection becomes one big ``(B*T, D) x (D, H*N)`` GEMM, then
reshape the result back to BDH's own ``(B, nh, T, N)`` layout. Zero change
to BDH's math -- this is purely an execution-layout claim, meant to be
falsified or confirmed by a real parity test and a real GPU timing
comparison, not assumed correct or assumed faster.

Real, disclosed scope limit: this is forward-only, not yet training-
integrated. ``wide_encoder_view`` returns a detached tensor, so gradients
do not flow back into a wide-native parameter through it. Wiring this into
an actual trainable forward pass -- so the wide cache is rebuilt once per
optimizer step, not once per forward call, avoiding the exact
permute-every-forward anti-pattern this remap exists to eliminate -- is a
separate, disclosed follow-up, left for after the layout claim itself is
confirmed (on real GPU timing) to be worth that added complexity.
"""
from __future__ import annotations

import torch


def wide_encoder_view(encoder: torch.Tensor) -> torch.Tensor:
    """Reshape BDH's per-head encoder ``(nh, D, N)`` into the GPU-native
    wide layout ``D x (nh*N)``. Real data movement (permute + contiguous)
    -- meant to be called once per set of weights (e.g. once per optimizer
    step in a training loop), not once per forward call."""
    nh, D, N = encoder.shape
    return encoder.detach().permute(1, 0, 2).reshape(D, nh * N).contiguous()


def bdh_wide_gemm_encoder_step(x: torch.Tensor, encoder_wide: torch.Tensor, nh: int, N: int) -> torch.Tensor:
    """Compute one recurrent level's ``x_latent = x @ encoder`` using the
    wide-GEMM layout instead of the oracle's broadcasted per-head matmul.

    x: ``(B, 1, T, D)``. Returns ``(B, nh, T, N)``, matching what
    ``Attention``/the rest of BDH's recurrent body expects exactly.
    """
    B, one, T, D = x.shape
    assert one == 1, f"BDH's own convention: x's head dim must be 1 before the per-head split, got {x.shape}"
    x2 = x.reshape(B * T, D)
    x_latent_wide = x2 @ encoder_wide  # (B*T, nh*N) -- one big GEMM
    return x_latent_wide.reshape(B, T, nh, N).permute(0, 2, 1, 3)
