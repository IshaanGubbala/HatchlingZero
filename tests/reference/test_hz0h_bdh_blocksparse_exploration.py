"""Regression tests for reference/hz0h_bdh_blocksparse_torch.py's
compute_active_blocks exploration_noise parameter (added while
diagnosing the training-instability finding in
docs/restart/hz0h_phase4_blocksparse_results.md's Update 5/6). Pins
down: exploration_noise=0 (the default) behaves identically to before
this parameter existed, and nonzero noise still returns valid,
correctly-shaped output (even though this session's own experiment
found it makes quality WORSE, not better -- that's a real result, not a
reason this code path shouldn't be correctly tested).
"""
from __future__ import annotations

import torch

from reference.hz0h_bdh_blocksparse_torch import compute_active_blocks
from reference.hz0h_bdh_torch import BDH, BDHConfig


def _tiny_config() -> BDHConfig:
    return BDHConfig(n_layer=2, n_embd=32, n_head=4, mlp_internal_dim_multiplier=8, vocab_size=32, dropout=0.0)


def test_zero_noise_is_deterministic_and_matches_no_noise_argument():
    config = _tiny_config()
    torch.manual_seed(0)
    model = BDH(config)
    idx = torch.randint(0, config.vocab_size, (4, 10))

    blocks_default = compute_active_blocks(model, idx, block_size=4, active_fraction=0.5)
    blocks_explicit_zero = compute_active_blocks(model, idx, block_size=4, active_fraction=0.5, exploration_noise=0.0)
    assert torch.equal(blocks_default, blocks_explicit_zero)

    # deterministic across repeated calls with noise=0
    blocks_again = compute_active_blocks(model, idx, block_size=4, active_fraction=0.5, exploration_noise=0.0)
    assert torch.equal(blocks_default, blocks_again)


def test_nonzero_noise_returns_valid_shaped_indices():
    config = _tiny_config()
    torch.manual_seed(1)
    model = BDH(config)
    idx = torch.randint(0, config.vocab_size, (4, 10))
    N = config.mlp_internal_dim_multiplier * config.n_embd // config.n_head
    n_blocks = N // 4

    blocks = compute_active_blocks(model, idx, block_size=4, active_fraction=0.5, exploration_noise=0.5)
    assert blocks.numel() == round(n_blocks * 0.5)
    assert (blocks >= 0).all() and (blocks < n_blocks).all()
    assert len(set(blocks.tolist())) == blocks.numel(), "no duplicate block indices"


def test_nonzero_noise_can_change_the_selection():
    """Real sanity check: noise should be CAPABLE of changing which
    blocks get selected (otherwise the parameter would be a no-op) --
    checked across several random seeds since any single draw could
    coincidentally match the noise-free selection."""
    config = _tiny_config()
    torch.manual_seed(2)
    model = BDH(config)
    idx = torch.randint(0, config.vocab_size, (4, 10))

    blocks_clean = compute_active_blocks(model, idx, block_size=4, active_fraction=0.5, exploration_noise=0.0)
    any_different = False
    for seed in range(10):
        torch.manual_seed(100 + seed)
        blocks_noisy = compute_active_blocks(model, idx, block_size=4, active_fraction=0.5, exploration_noise=2.0)
        if not torch.equal(blocks_clean, blocks_noisy):
            any_different = True
            break
    assert any_different, "exploration_noise should be capable of changing block selection"
