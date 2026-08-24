"""Real correctness tests for
reference/hz0h_bdh_context_refresh_ablation_torch.py."""
from __future__ import annotations

import torch

from reference.hz0h_bdh_context_refresh_ablation_torch import bdh_context_refresh_forward
from reference.hz0h_bdh_torch import BDH, BDHConfig
from reference.hz0h_bdh_variable_depth_torch import bdh_variable_depth_forward


def _model(seed: int = 17) -> BDH:
    torch.manual_seed(seed)
    return BDH(BDHConfig(n_layer=8, n_embd=32, n_head=4, mlp_internal_dim_multiplier=8, vocab_size=256, dropout=0.0))


def test_refresh_every_1_matches_vanilla_bdh_exactly():
    """refresh_every=1 must be bit-identical to vanilla BDH -- a fresh
    context read every round is exactly what vanilla BDH already does."""
    model = _model()
    idx = torch.randint(256, (2, 10))
    targets = torch.randint(256, (2, 10))
    with torch.no_grad():
        ref_logits, ref_loss = bdh_variable_depth_forward(model, idx, 8, targets)
        abl_logits, abl_loss, real_reads = bdh_context_refresh_forward(model, idx, 8, refresh_every=1, targets=targets)
    assert torch.equal(ref_logits, abl_logits)
    assert torch.equal(ref_loss, abl_loss)
    assert real_reads == 8


def test_real_read_count_matches_refresh_interval():
    model = _model()
    idx = torch.randint(256, (2, 10))
    with torch.no_grad():
        for refresh_every, expected_reads in [(1, 8), (2, 4), (4, 2), (8, 1), (100, 1)]:
            _, _, real_reads = bdh_context_refresh_forward(model, idx, 8, refresh_every=refresh_every)
            assert real_reads == expected_reads, f"refresh_every={refresh_every}"


def test_higher_refresh_every_diverges_from_vanilla():
    """Sanity: reusing stale context must actually change the output for
    refresh_every > 1 (on random, untrained weights) -- confirms the
    freeze logic is actually doing something, not silently a no-op."""
    model = _model(seed=29)
    idx = torch.randint(256, (2, 10))
    with torch.no_grad():
        ref_logits, _ = bdh_variable_depth_forward(model, idx, 8)
        frozen_logits, _, real_reads = bdh_context_refresh_forward(model, idx, 8, refresh_every=8)
    assert real_reads == 1
    assert not torch.allclose(ref_logits, frozen_logits, atol=1e-4)
