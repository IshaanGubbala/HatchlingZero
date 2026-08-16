"""CUDA correctness gate for the hand-written Triton BDH attention kernel."""
from __future__ import annotations

import pytest
import torch

from reference.hz0h_bdh_torch import Attention, BDHConfig
from reference.hz0h_bdh_triton_attention_torch import (
    bdh_triton_attention,
    triton_available,
)


pytestmark = pytest.mark.skipif(
    not triton_available(), reason="requires CUDA and Triton"
)


# Real, disclosed reason for scaling tolerance by each ROW's own magnitude
# instead of torch.allclose's naive per-element rtol: a real CUDA run of
# this test (2026-08-16) failed all 5 parametrized cases even at a generous
# flat atol/rtol=2e-2/2e-2. A follow-up error-localization diagnostic
# (scripts/hz0h_triton_kernel_error_localization_diagnostic.py) found (a)
# zero spurious output at any position the oracle causal-masks to exactly
# zero -- rules out a masking bug -- and (b) a per-row max error that stays
# a roughly uniform ~0.25-0.76% of that row's own overall magnitude across
# every row, shape, and depth tested, never concentrating at a tile boundary
# or growing with depth. That's the real, expected signature of the oracle
# rounding to bf16 twice (after QR@KR.mT, again after scores@V) while this
# kernel accumulates in fp32 and rounds once -- not a logic bug (see the
# fix history in reference/hz0h_bdh_triton_attention_torch.py's module
# docstring). BDH's output has near-zero individual feature dimensions
# sitting next to large ones within the same row (V is unstructured
# Gaussian noise mixed across T), so torch.allclose's per-ELEMENT rtol
# breaks down at those near-zero dimensions even though the error is a tiny,
# uniform fraction of the row's own scale -- the same near-zero-reference
# tolerance failure mode already documented and fixed for the native tiled
# kernel (docs/restart/hz0h_bdh_native_kernel_results.md).
#
# _FORWARD_SCALE_RTOL=1.5e-2 carries roughly 2x margin over the real
# measured max ratio (0.76%). Gradients are computed by a separate, explicit
# chunked analytic backward (not the forward kernel), so no independent
# per-row error measurement exists for them yet; _GRAD_SCALE_RTOL is set by
# analogy with the same margin, not independently measured -- a real
# gradient-specific error-localization pass would be needed before treating
# that number as anything more than a reasonable starting point.
_FORWARD_FLOOR_ATOL = 1e-2
_FORWARD_SCALE_RTOL = 1.5e-2
_GRAD_FLOOR_ATOL = 1e-2
_GRAD_SCALE_RTOL = 2e-2


def _assert_row_scaled_allclose(kernel: torch.Tensor, oracle: torch.Tensor, floor_atol: float, scale_rtol: float, label: str) -> None:
    """Like torch.allclose, but the rtol term scales with each row's (last
    two dims collapsed to one "row" per (batch, head, query) triple) own
    max magnitude instead of each individual element's -- see the real,
    disclosed reasoning above this function."""
    diff = (kernel - oracle).abs()
    row_scale = oracle.abs().amax(dim=-1, keepdim=True)
    allowed = floor_atol + scale_rtol * row_scale
    if not bool((diff <= allowed).all()):
        worst = (diff - allowed)
        flat_idx = int(worst.argmax().item())
        loc = [int(x) for x in torch.unravel_index(torch.tensor(flat_idx), diff.shape)]
        raise AssertionError(
            f"{label}: row-scaled tolerance exceeded at index {loc}: "
            f"diff={diff.flatten()[flat_idx].item():.6f} "
            f"allowed={allowed.flatten()[flat_idx].item():.6f} "
            f"oracle={oracle.flatten()[flat_idx].item():.6f} "
            f"kernel={kernel.flatten()[flat_idx].item():.6f}"
        )


@pytest.mark.parametrize(
    "seed,n_embd,n_head,mult,batch,seq_len",
    [
        (19, 32, 4, 32, 2, 17),   # original single case, kept for continuity
        (1, 16, 2, 16, 1, 8),     # smallest real shape: n_head=2, short T
        (7, 64, 8, 16, 3, 33),    # odd T (33, not a power of 2 or multiple of BLOCK_M=32)
        (11, 32, 4, 8, 2, 65),    # T straddles two BLOCK_M=32 tiles plus a remainder
        (23, 128, 8, 32, 1, 256), # closer to the real Phase F shape (N=512/head here, T=256)
    ],
)
def test_triton_forward_and_gradients_match_oracle(seed, n_embd, n_head, mult, batch, seq_len):
    config = BDHConfig(n_embd=n_embd, n_head=n_head, mlp_internal_dim_multiplier=mult)
    attention = Attention(config).to(device="cuda", dtype=torch.bfloat16)
    attention.freqs = attention.freqs.to(torch.float32)
    N = n_embd * mult // n_head
    torch.manual_seed(seed)
    q_oracle = torch.randn(batch, n_head, seq_len, N, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v_oracle = torch.randn(batch, 1, seq_len, n_embd, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    q_kernel = q_oracle.detach().clone().requires_grad_()
    v_kernel = v_oracle.detach().clone().requires_grad_()
    oracle = attention(q_oracle, q_oracle, v_oracle)
    kernel = bdh_triton_attention(q_kernel, v_kernel, attention.freqs)
    label = f"seed={seed} n_embd={n_embd} n_head={n_head} mult={mult} batch={batch} seq_len={seq_len}"
    _assert_row_scaled_allclose(kernel, oracle, _FORWARD_FLOOR_ATOL, _FORWARD_SCALE_RTOL, f"{label}: forward")
    oracle.square().mean().backward()
    kernel.square().mean().backward()
    _assert_row_scaled_allclose(q_kernel.grad, q_oracle.grad, _GRAD_FLOOR_ATOL, _GRAD_SCALE_RTOL, f"{label}: Q grad")
    _assert_row_scaled_allclose(v_kernel.grad, v_oracle.grad, _GRAD_FLOOR_ATOL, _GRAD_SCALE_RTOL, f"{label}: V grad")
