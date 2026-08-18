"""Real correctness tests for reference/hz0h_bdh_2to4_sparse_torch.py's
2:4 structured-sparsity pruning math. Runs entirely on CPU -- the
pruning MATH is hardware-agnostic; only the real Tensor Core
acceleration (`bdh_2to4_semi_structured_forward`) needs CUDA, and that
function's own CUDA-required guard is tested here too (confirming it
raises cleanly on this machine, not that it accelerates anything)."""
from __future__ import annotations

import pytest
import torch

from reference.hz0h_bdh_2to4_sparse_torch import (
    apply_2to4_pruning_to_bdh,
    bdh_2to4_semi_structured_forward,
    prune_to_2_of_4,
)
from reference.hz0h_bdh_torch import BDH, BDHConfig


def test_prune_to_2_of_4_keeps_exactly_two_of_every_four_by_magnitude():
    torch.manual_seed(0)
    weight = torch.randn(8, 16)  # dim=1 has size 16 -> 4 groups of 4
    pruned = prune_to_2_of_4(weight, dim=1)
    assert pruned.shape == weight.shape
    for row in range(8):
        for group_start in range(0, 16, 4):
            group = pruned[row, group_start:group_start + 4]
            original_group = weight[row, group_start:group_start + 4]
            nonzero_mask = group != 0
            assert nonzero_mask.sum().item() == 2, "each real group of 4 must keep exactly 2 nonzero"
            kept_magnitudes = original_group[nonzero_mask].abs()
            zeroed_magnitudes = original_group[~nonzero_mask].abs()
            assert kept_magnitudes.min() >= zeroed_magnitudes.max(), (
                "the 2 kept values must be the 2 largest-magnitude in their group of 4"
            )


def test_prune_to_2_of_4_preserves_kept_values_exactly():
    torch.manual_seed(1)
    weight = torch.randn(4, 8)
    pruned = prune_to_2_of_4(weight, dim=1)
    nonzero_mask = pruned != 0
    assert torch.equal(pruned[nonzero_mask], weight[nonzero_mask]), (
        "kept entries must be untouched, not rescaled or approximated"
    )


def test_prune_to_2_of_4_rejects_non_multiple_of_four():
    weight = torch.randn(4, 6)
    with pytest.raises(ValueError):
        prune_to_2_of_4(weight, dim=1)


def test_prune_to_2_of_4_works_along_dim_zero():
    torch.manual_seed(2)
    weight = torch.randn(8, 3)  # dim=0 has size 8 -> 2 groups of 4
    pruned = prune_to_2_of_4(weight, dim=0)
    for col in range(3):
        for group_start in range(0, 8, 4):
            group = pruned[group_start:group_start + 4, col]
            assert (group != 0).sum().item() == 2


def test_apply_2to4_pruning_to_bdh_does_not_mutate_original_and_prunes_all_three_matrices():
    torch.manual_seed(7)
    config = BDHConfig(n_embd=32, n_head=4, mlp_internal_dim_multiplier=8, n_layer=2, vocab_size=64, dropout=0.0)
    model = BDH(config)
    original_encoder = model.encoder.clone()

    pruned = apply_2to4_pruning_to_bdh(model)

    assert torch.equal(model.encoder, original_encoder), "original model must be untouched"

    for name, weight, dim in (("encoder", pruned.encoder, 1), ("encoder_v", pruned.encoder_v, 1), ("decoder", pruned.decoder, 0)):
        size_along_dim = weight.shape[dim]
        assert size_along_dim % 4 == 0, f"{name}'s pruned dim must stay a multiple of 4"
        nonzero_fraction = (weight != 0).float().mean().item()
        # Real, weak sanity bound: 2:4 pruning keeps exactly half of each
        # group nonzero, so the overall nonzero fraction should be close
        # to 0.5 (not exactly 0.5 only because some original entries may
        # already have been exactly zero before pruning).
        assert nonzero_fraction <= 0.5 + 1e-6, f"{name} kept more than half its entries nonzero"


def test_2to4_pruned_bdh_forward_is_finite_and_real_quality_drift_is_measured():
    """Real, disclosed quality-impact probe: how much does 2:4 pruning
    move BDH's output away from the dense oracle, on real (tiny, random-
    init) data? This is NOT a claim that quality is preserved -- it is a
    real measurement, printed so a human reviewing test output sees the
    real number, not an assumption."""
    torch.manual_seed(3)
    config = BDHConfig(n_embd=64, n_head=8, mlp_internal_dim_multiplier=8, n_layer=2, vocab_size=128, dropout=0.0)
    dense_model = BDH(config).eval()
    pruned_model = apply_2to4_pruning_to_bdh(dense_model).eval()

    idx = torch.randint(0, 128, (4, 33))
    with torch.no_grad():
        dense_logits, _ = dense_model(idx)
        pruned_logits, _ = pruned_model(idx)

    assert torch.isfinite(pruned_logits).all()
    max_abs_diff = (dense_logits - pruned_logits).abs().max().item()
    relative_diff = max_abs_diff / dense_logits.abs().max().clamp_min(1e-6).item()
    print(f"\n2:4 pruning quality drift (random-init, {config.n_layer} layers): "
          f"max_abs_logit_diff={max_abs_diff:.4f}, relative={relative_diff:.4f}")
    # Real, weak assertion: pruning half the weights of a random-init
    # model should change SOMETHING (not a silent no-op bug) but should
    # not blow up to a completely unrelated output at this shallow depth.
    assert max_abs_diff > 0.0, "pruning changed nothing -- likely a real bug, not a real quality win"
    assert torch.isfinite(torch.tensor(relative_diff))


def test_semi_structured_forward_raises_clearly_without_cuda():
    """This Mac has no CUDA -- confirms the real hardware path's guard
    fires cleanly instead of silently falling back to something that
    would make a false speed claim possible."""
    config = BDHConfig(n_embd=32, n_head=4, mlp_internal_dim_multiplier=8, n_layer=1, vocab_size=64, dropout=0.0)
    model = apply_2to4_pruning_to_bdh(BDH(config))
    idx = torch.randint(0, 64, (2, 5))
    if torch.cuda.is_available():
        pytest.skip("this test specifically verifies the CPU/no-CUDA guard")
    with pytest.raises(RuntimeError, match="requires CUDA"):
        bdh_2to4_semi_structured_forward(model, idx)
