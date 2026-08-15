"""Regression tests for reference/hz0h_bdh_blocksparse_torch.py (Phase 4
BlockBDH, plans/HatchlingZero_Reality_Plan.md). Pins down: 100% active
blocks is byte-identical to dense BDH.forward (the real validation this
implementation depends on -- includes the RoPE-frequency-gathering fix,
without which sub-100% fractions crash or silently rotate columns by
the wrong phase), and every active fraction produces finite output.
"""
from __future__ import annotations

import torch

from reference.hz0h_bdh_blocksparse_torch import bdh_blocksparse_forward, compute_active_blocks
from reference.hz0h_bdh_torch import BDH, BDHConfig


def _tiny_config() -> BDHConfig:
    return BDHConfig(n_layer=3, n_embd=64, n_head=4, mlp_internal_dim_multiplier=16, vocab_size=32, dropout=0.0)


def test_full_density_matches_dense_bdh_exactly():
    config = _tiny_config()
    torch.manual_seed(0)
    model = BDH(config)
    model.eval()
    idx = torch.randint(0, config.vocab_size, (2, 12))

    with torch.no_grad():
        logits_dense, _ = model(idx)
        active_blocks = compute_active_blocks(model, idx, block_size=32, active_fraction=1.0)
        logits_sparse, _ = bdh_blocksparse_forward(model, idx, active_blocks, block_size=32)

    assert torch.equal(logits_dense, logits_sparse)


def test_active_block_count_matches_requested_fraction():
    config = _tiny_config()
    torch.manual_seed(1)
    model = BDH(config)
    idx = torch.randint(0, config.vocab_size, (2, 10))
    n_blocks = config.mlp_internal_dim_multiplier * config.n_embd // config.n_head // 32  # N // block_size

    for fraction, expected_count in ((1.0, n_blocks), (0.5, n_blocks // 2), (0.25, n_blocks // 4)):
        active_blocks = compute_active_blocks(model, idx, block_size=32, active_fraction=fraction)
        assert len(active_blocks) == expected_count, f"fraction={fraction}"


def test_reduced_fractions_produce_finite_output_and_differ_from_dense():
    config = _tiny_config()
    torch.manual_seed(2)
    model = BDH(config)
    model.eval()
    idx = torch.randint(0, config.vocab_size, (2, 12))

    with torch.no_grad():
        logits_dense, _ = model(idx)
        for fraction in (0.5, 0.25, 0.125):
            active_blocks = compute_active_blocks(model, idx, block_size=32, active_fraction=fraction)
            logits_sparse, _ = bdh_blocksparse_forward(model, idx, active_blocks, block_size=32)
            assert torch.isfinite(logits_sparse).all(), f"fraction={fraction}"
            assert not torch.allclose(logits_dense, logits_sparse, atol=1e-3), f"fraction={fraction} should differ from dense (real information is dropped)"


def test_gradients_flow_through_selected_columns_only():
    """Real property this session's positive training result depends on:
    backprop through bdh_blocksparse_forward must reach encoder/
    encoder_v/decoder's SELECTED columns (so training can actually adapt
    them), while unselected columns correctly get zero gradient this
    step (they weren't used in the forward pass at all)."""
    config = _tiny_config()
    torch.manual_seed(5)
    model = BDH(config)
    model.train()
    idx = torch.randint(0, config.vocab_size, (2, 9))
    active_blocks = compute_active_blocks(model, idx, block_size=32, active_fraction=0.5)
    x, y = idx[:, :-1].contiguous(), idx[:, 1:].contiguous()
    _logits, loss = bdh_blocksparse_forward(model, x, active_blocks, block_size=32, targets=y)
    loss.backward()

    assert model.encoder.grad is not None
    N = config.mlp_internal_dim_multiplier * config.n_embd // config.n_head
    column_indices = (active_blocks.view(-1, 1) * 32 + torch.arange(32)).reshape(-1)
    selected_grad_norm = float(model.encoder.grad[:, :, column_indices].norm())
    mask = torch.ones(N, dtype=torch.bool)
    mask[column_indices] = False
    unselected_grad_norm = float(model.encoder.grad[:, :, mask].norm())
    assert selected_grad_norm > 0, "selected columns should receive real gradient"
    assert unselected_grad_norm == 0.0, "unselected columns should receive exactly zero gradient this step"


def test_odd_block_size_is_rejected():
    config = _tiny_config()
    torch.manual_seed(3)
    model = BDH(config)
    idx = torch.randint(0, config.vocab_size, (1, 6))
    active_blocks = torch.tensor([0])
    try:
        bdh_blocksparse_forward(model, idx, active_blocks, block_size=31)
        assert False, "expected ValueError for odd block_size"
    except ValueError:
        pass


def test_loss_computed_when_targets_given():
    config = _tiny_config()
    torch.manual_seed(4)
    model = BDH(config)
    idx = torch.randint(0, config.vocab_size, (2, 9))
    active_blocks = compute_active_blocks(model, idx, block_size=32, active_fraction=0.5)
    x, y = idx[:, :-1].contiguous(), idx[:, 1:].contiguous()
    _logits, loss = bdh_blocksparse_forward(model, x, active_blocks, block_size=32, targets=y)
    assert loss is not None and torch.isfinite(loss)


def test_cheap_proxy_router_is_deterministic_and_returns_valid_blocks():
    config = _tiny_config()
    torch.manual_seed(6)
    model = BDH(config)
    idx = torch.randint(0, config.vocab_size, (2, 12))
    first = compute_active_blocks(model, idx, block_size=32, active_fraction=0.25, method="cheap_proxy")
    second = compute_active_blocks(model, idx, block_size=32, active_fraction=0.25, method="cheap_proxy")
    assert torch.equal(first, second)
    assert len(first) == 2
    assert int(first.min()) >= 0 and int(first.max()) < 8
    logits, _ = bdh_blocksparse_forward(model, idx, first, block_size=32)
    assert torch.isfinite(logits).all()


def test_router_rejects_unknown_method():
    model = BDH(_tiny_config())
    idx = torch.randint(0, 32, (1, 8))
    try:
        compute_active_blocks(model, idx, block_size=32, active_fraction=0.5, method="unknown")
        assert False, "expected method validation"
    except ValueError as exc:
        assert "unknown routing method" in str(exc)
