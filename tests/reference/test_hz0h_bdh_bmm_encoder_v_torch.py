"""Real correctness tests for reference/hz0h_bdh_bmm_encoder_v_torch.py:
does the explicit per-head batched GEMM produce numerically-identical
output to BDH's own broadcasted per-head matmul, across real shapes
including the project's actual Phase F configuration."""
from __future__ import annotations

import pytest
import torch

from reference.hz0h_bdh_bmm_encoder_v_torch import bmm_encoder_v_step
from reference.hz0h_bdh_torch import BDH, BDHConfig


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
def test_bmm_matches_oracle_broadcast_matmul(seed, n_embd, n_head, mult, batch, seq_len):
    torch.manual_seed(seed)
    config = BDHConfig(n_embd=n_embd, n_head=n_head, mlp_internal_dim_multiplier=mult)
    model = BDH(config)
    N = n_embd * mult // n_head

    # yKV is post-attention, post-LayerNorm in the real forward loop -- use
    # plain randn here since bmm_encoder_v_step only needs to match the
    # oracle's own yKV @ encoder_v expression for ANY yKV, not reproduce
    # attention itself.
    yKV = torch.randn(batch, n_head, seq_len, n_embd)

    oracle_latent = yKV @ model._w(model.encoder_v)  # (B, nh, T, N), the oracle's own exact expression
    bmm_latent = bmm_encoder_v_step(yKV, model.encoder_v)

    assert bmm_latent.shape == oracle_latent.shape
    max_diff = (bmm_latent - oracle_latent).abs().max().item()
    assert torch.allclose(bmm_latent, oracle_latent, atol=1e-5, rtol=1e-4), (
        f"seed={seed} n_embd={n_embd} n_head={n_head} mult={mult} batch={batch} "
        f"seq_len={seq_len}: max diff {max_diff}"
    )


def test_bmm_rejects_head_or_dim_mismatch():
    config = BDHConfig(n_embd=32, n_head=4, mlp_internal_dim_multiplier=8)
    model = BDH(config)
    bad_ykv = torch.randn(2, 3, 5, 32)  # nh=3, doesn't match encoder_v's nh=4
    with pytest.raises(AssertionError):
        bmm_encoder_v_step(bad_ykv, model.encoder_v)
