from __future__ import annotations

import torch

from reference.hz0h_bdh_blocksparse_torch import bdh_blocksparse_direct_split_v_forward, compute_active_blocks
from reference.hz0h_bdh_torch import BDH, BDHConfig


def _model():
    return BDH(BDHConfig(n_layer=2, n_embd=32, n_head=4, mlp_internal_dim_multiplier=8, vocab_size=32, dropout=0.0))


def test_direct_split_v_sparse_forward_is_finite_and_preserves_parameter_set():
    torch.manual_seed(8)
    model = _model()
    before = {name for name, _ in model.named_parameters()}
    idx = torch.randint(0, 32, (2, 10))
    blocks = compute_active_blocks(model, idx, block_size=4, active_fraction=0.5)
    logits, loss = bdh_blocksparse_direct_split_v_forward(model, idx, blocks, 4, targets=idx)
    assert logits.shape == (2, 10, 32)
    assert torch.isfinite(loss)
    assert {name for name, _ in model.named_parameters()} == before
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name


def test_direct_split_v_requires_equal_head_value_slices():
    model = BDH(BDHConfig(n_layer=1, n_embd=33, n_head=4, mlp_internal_dim_multiplier=8, vocab_size=32, dropout=0.0))
    idx = torch.randint(0, 32, (1, 4))
    try:
        bdh_blocksparse_direct_split_v_forward(model, idx, torch.tensor([0]), 4)
        assert False, "expected divisibility error"
    except ValueError as exc:
        assert "divisible" in str(exc)
