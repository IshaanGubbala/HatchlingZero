from __future__ import annotations

import torch

from reference.hz0h_bdh_block_gated_torch import (
    BDHBlockGated, BDHBlockGatedConfig, bdh_block_gated_annealed_direct_split_v_forward,
    bdh_block_gated_forward,
)


def _model():
    return BDHBlockGated(BDHBlockGatedConfig(n_layer=2, n_embd=32, n_head=4,
        mlp_internal_dim_multiplier=8, vocab_size=32, dropout=0.0, block_size=4))


def test_dense_gate_phase_remains_exact_established_gate_forward():
    torch.manual_seed(9)
    model = _model().eval()
    idx = torch.randint(0, 32, (2, 9))
    expected, _ = bdh_block_gated_forward(model, idx)
    actual, _ = bdh_block_gated_annealed_direct_split_v_forward(model, idx, 1.0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_sparse_direct_value_gate_path_has_finite_gradients_including_gate():
    torch.manual_seed(10)
    model = _model()
    idx = torch.randint(0, 32, (2, 9))
    logits, loss = bdh_block_gated_annealed_direct_split_v_forward(model, idx, 0.5, targets=idx)
    assert logits.shape == (2, 9, 32)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.gate.grad is not None and torch.isfinite(model.gate.grad).all()
    assert float(model.gate.grad.norm()) > 0
    assert model.encoder.grad is not None and torch.isfinite(model.encoder.grad).all()
