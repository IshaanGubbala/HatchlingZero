"""Regression tests for reference/hz0h_bdh_vb_selective_variable_depth_torch.py,
mirroring test_hz0h_bdh_vb_variable_depth_torch.py's coverage for the
selective-write-gate arm.
"""
from __future__ import annotations

import torch

from reference.hz0h_bdh_vb_selective_torch import BDHVBSelective, BDHVBSelectiveConfig
from reference.hz0h_bdh_vb_selective_variable_depth_torch import bdh_vb_selective_variable_depth_forward


def _tiny_config() -> BDHVBSelectiveConfig:
    return BDHVBSelectiveConfig(n_layer=4, n_embd=32, n_head=4, mlp_internal_dim_multiplier=8, vocab_size=32, dropout=0.0, d_state=8)


def test_matches_dense_forward_at_matching_iteration_count():
    config = _tiny_config()
    torch.manual_seed(0)
    model = BDHVBSelective(config)
    model.eval()
    idx = torch.randint(0, config.vocab_size, (2, 10))

    with torch.no_grad():
        logits_dense, _ = model(idx)
        logits_var, _ = bdh_vb_selective_variable_depth_forward(model, idx, n_iterations=config.n_layer)

    assert torch.equal(logits_dense, logits_var)


def test_different_iteration_counts_produce_different_output():
    config = _tiny_config()
    torch.manual_seed(1)
    model = BDHVBSelective(config)
    model.eval()
    idx = torch.randint(0, config.vocab_size, (2, 8))

    with torch.no_grad():
        logits_2, _ = bdh_vb_selective_variable_depth_forward(model, idx, n_iterations=2)
        logits_8, _ = bdh_vb_selective_variable_depth_forward(model, idx, n_iterations=8)

    assert not torch.allclose(logits_2, logits_8, atol=1e-3)


def test_gradients_flow_for_arbitrary_iteration_count():
    config = _tiny_config()
    torch.manual_seed(2)
    model = BDHVBSelective(config)
    model.train()
    idx = torch.randint(0, config.vocab_size, (2, 9))
    x, y = idx[:, :-1].contiguous(), idx[:, 1:].contiguous()
    _logits, loss = bdh_vb_selective_variable_depth_forward(model, x, n_iterations=10, targets=y)
    loss.backward()
    assert model.write_gate.grad is not None and float(model.write_gate.grad.norm()) > 0
    assert model.encoder.grad is not None and float(model.encoder.grad.norm()) > 0
