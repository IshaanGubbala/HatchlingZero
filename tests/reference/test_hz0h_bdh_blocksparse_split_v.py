from __future__ import annotations

import torch

from reference.hz0h_bdh_blocksparse_torch import bdh_blocksparse_split_v_forward, compute_active_blocks
from reference.hz0h_bdh_split_v_torch import BDHSplitV, BDHSplitVConfig


def _model():
    config = BDHSplitVConfig(n_layer=2, n_embd=32, n_head=4, mlp_internal_dim_multiplier=8, vocab_size=32, dropout=0.0)
    return BDHSplitV(config)


def test_blocksparse_split_v_all_blocks_matches_split_v_forward():
    torch.manual_seed(3)
    model = _model().eval()
    idx = torch.randint(0, 32, (2, 10))
    n = 32 * 8 // 4
    block_size = 4
    blocks = torch.arange(n // block_size)
    expected, _ = model(idx)
    actual, _ = bdh_blocksparse_split_v_forward(model, idx, blocks, block_size)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_blocksparse_split_v_routes_and_backpropagates_to_new_parameters():
    torch.manual_seed(4)
    model = _model()
    idx = torch.randint(0, 32, (2, 10))
    targets = torch.randint(0, 32, (2, 10))
    blocks = compute_active_blocks(model, idx, block_size=4, active_fraction=0.5)
    logits, loss = bdh_blocksparse_split_v_forward(model, idx, blocks, 4, targets=targets)
    assert logits.shape == (2, 10, 32)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.w_v.grad is not None and torch.isfinite(model.w_v.grad).all()
    assert model.w_o.grad is not None and torch.isfinite(model.w_o.grad).all()
    assert model.encoder.grad is not None and torch.isfinite(model.encoder.grad).all()


def test_blocksparse_split_v_requires_even_block_size():
    model = _model()
    idx = torch.randint(0, 32, (1, 4))
    try:
        bdh_blocksparse_split_v_forward(model, idx, torch.tensor([0]), 3)
        assert False, "expected block-size validation"
    except ValueError as exc:
        assert "even" in str(exc)
