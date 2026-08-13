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


def test_gate_reg_lambda_zero_matches_prior_behavior_exactly():
    config = _tiny_config()
    torch.manual_seed(3)
    model = BDHVBSelective(config)
    model.eval()
    idx = torch.randint(0, config.vocab_size, (2, 9))
    x, y = idx[:, :-1].contiguous(), idx[:, 1:].contiguous()
    with torch.no_grad():
        logits_default, loss_default = bdh_vb_selective_variable_depth_forward(model, x, n_iterations=6, targets=y)
        logits_explicit_zero, loss_explicit_zero = bdh_vb_selective_variable_depth_forward(model, x, n_iterations=6, targets=y, gate_reg_lambda=0.0)
    assert torch.equal(logits_default, logits_explicit_zero)
    assert torch.equal(loss_default, loss_explicit_zero)


def test_gate_reg_lambda_positive_lowers_loss_when_gate_is_decisive():
    config = _tiny_config()
    torch.manual_seed(4)
    model = BDHVBSelective(config)
    model.eval()
    idx = torch.randint(0, config.vocab_size, (2, 9))
    x, y = idx[:, :-1].contiguous(), idx[:, 1:].contiguous()
    with torch.no_grad():
        # push the gate weight to a large magnitude so gate activations polarize away from 0.5
        model.write_gate.mul_(50.0)
        _logits, loss_polarized = bdh_vb_selective_variable_depth_forward(model, x, n_iterations=6, targets=y, gate_reg_lambda=0.1)
        model.write_gate.zero_()
        _logits, loss_at_midpoint = bdh_vb_selective_variable_depth_forward(model, x, n_iterations=6, targets=y, gate_reg_lambda=0.1)
    # with write_gate == 0, every gate activation is exactly sigmoid(0) == 0.5, so the
    # regularizer contributes nothing there but subtracts a real positive amount when polarized
    assert float(loss_polarized) < float(loss_at_midpoint)


def test_gate_reg_lambda_requires_grad_flows_to_write_gate():
    config = _tiny_config()
    torch.manual_seed(5)
    model = BDHVBSelective(config)
    model.train()
    idx = torch.randint(0, config.vocab_size, (2, 9))
    x, y = idx[:, :-1].contiguous(), idx[:, 1:].contiguous()
    _logits, loss = bdh_vb_selective_variable_depth_forward(model, x, n_iterations=6, targets=y, gate_reg_lambda=0.1)
    loss.backward()
    assert model.write_gate.grad is not None and float(model.write_gate.grad.norm()) > 0
