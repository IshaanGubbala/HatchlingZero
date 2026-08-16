"""Real correctness tests for reference/hz0h_bdh_wide_gemm_encoder_torch.py:
does the wide-GEMM encoder layout produce numerically-identical output to
BDH's own broadcasted per-head matmul, across real shapes including the
project's actual Phase F configuration."""
from __future__ import annotations

import pytest
import torch

from reference.hz0h_bdh_torch import BDH, BDHConfig
from reference.hz0h_bdh_wide_gemm_encoder_torch import (
    bdh_wide_gemm_encoder_step,
    wide_encoder_view,
)


@pytest.mark.parametrize(
    "seed,n_embd,n_head,mult,batch,seq_len",
    [
        (0, 32, 4, 8, 2, 5),      # small, exact
        (1, 64, 8, 4, 3, 11),     # different head count/multiplier
        (2, 16, 2, 16, 1, 1),     # single token
        (3, 128, 8, 4, 4, 17),    # odd batch/seq combo
        (4, 512, 8, 32, 2, 256),  # this project's real Phase F shape (N=2048/head, T=256)
    ],
)
def test_wide_gemm_matches_oracle_broadcast_matmul(seed, n_embd, n_head, mult, batch, seq_len):
    torch.manual_seed(seed)
    config = BDHConfig(n_embd=n_embd, n_head=n_head, mlp_internal_dim_multiplier=mult)
    model = BDH(config)
    N = n_embd * mult // n_head

    x = torch.randn(batch, 1, seq_len, n_embd)

    oracle_latent = x @ model._w(model.encoder)  # (B, nh, T, N), the oracle's own exact expression

    encoder_wide = wide_encoder_view(model.encoder)
    assert encoder_wide.shape == (n_embd, n_head * N)
    wide_latent = bdh_wide_gemm_encoder_step(x, encoder_wide, n_head, N)

    assert wide_latent.shape == oracle_latent.shape
    max_diff = (wide_latent - oracle_latent).abs().max().item()
    # Same math, same fp32 dtype on both sides -- only floating-point
    # summation-order differences between cuBLAS/backend GEMM shapes are
    # expected, not a logic difference. Tight, fp32-appropriate tolerance.
    assert torch.allclose(wide_latent, oracle_latent, atol=1e-5, rtol=1e-4), (
        f"seed={seed} n_embd={n_embd} n_head={n_head} mult={mult} batch={batch} "
        f"seq_len={seq_len}: max diff {max_diff}"
    )


def test_wide_encoder_view_is_detached_not_the_live_parameter():
    """Real, disclosed scope limit: wide_encoder_view returns a detached
    snapshot, so it does NOT track gradients back into model.encoder. This
    test exists so that scope limit stays enforced/documented, not silently
    dropped in a later edit."""
    config = BDHConfig(n_embd=32, n_head=4, mlp_internal_dim_multiplier=8)
    model = BDH(config)
    encoder_wide = wide_encoder_view(model.encoder)
    assert not encoder_wide.requires_grad


def test_wide_gemm_rejects_wrong_head_dim():
    config = BDHConfig(n_embd=32, n_head=4, mlp_internal_dim_multiplier=8)
    model = BDH(config)
    N = 32 * 8 // 4
    encoder_wide = wide_encoder_view(model.encoder)
    bad_x = torch.randn(2, 2, 5, 32)  # head dim must be 1, not 2
    with pytest.raises(AssertionError):
        bdh_wide_gemm_encoder_step(bad_x, encoder_wide, 4, N)
