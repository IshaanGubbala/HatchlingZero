"""Real correctness tests for reference/hz0h_bdh_vb_checkpointed_torch.py.

Mirrors tests/reference/test_hz0h_bdh_checkpointed_torch.py's structure
exactly, for the VB (value-bottleneck) variant: proves checkpointing
changes HOW gradients are computed (recompute vs store), not WHAT is
computed, by comparing against the uncheckpointed
bdh_vb_variable_depth_forward at the same seed/weights/input.
"""
from __future__ import annotations

import torch

from reference.hz0h_bdh_vb_torch import BDHVB, BDHVBConfig
from reference.hz0h_bdh_vb_variable_depth_torch import bdh_vb_variable_depth_forward
from reference.hz0h_bdh_vb_checkpointed_torch import bdh_vb_variable_depth_forward_checkpointed


def _tiny_config(n_layer: int = 3, n_embd: int = 32, n_head: int = 4) -> BDHVBConfig:
    return BDHVBConfig(n_layer=n_layer, n_embd=n_embd, n_head=n_head, mlp_internal_dim_multiplier=8, vocab_size=32, dropout=0.0, d_state=16)


def test_logits_match_uncheckpointed_forward():
    config = _tiny_config()
    torch.manual_seed(11)
    model = BDHVB(config)
    idx = torch.randint(0, config.vocab_size, (2, 16))

    with torch.no_grad():
        logits_plain, _ = bdh_vb_variable_depth_forward(model, idx, n_iterations=4)
        logits_ckpt, _ = bdh_vb_variable_depth_forward_checkpointed(model, idx, n_iterations=4)

    max_diff = (logits_plain - logits_ckpt).abs().max().item()
    assert torch.equal(logits_plain, logits_ckpt), f"checkpointed VB forward diverged from plain forward, max diff {max_diff}"


def test_gradients_match_uncheckpointed_forward():
    config = _tiny_config()
    torch.manual_seed(12)
    model_a = BDHVB(config)
    model_b = BDHVB(config)
    model_b.load_state_dict(model_a.state_dict())

    idx = torch.randint(0, config.vocab_size, (2, 12))
    x, y = idx[:, :-1].contiguous(), idx[:, 1:].contiguous()

    _logits_a, loss_a = bdh_vb_variable_depth_forward(model_a, x, n_iterations=4, targets=y)
    loss_a.backward()

    _logits_b, loss_b = bdh_vb_variable_depth_forward_checkpointed(model_b, x, n_iterations=4, targets=y)
    loss_b.backward()

    assert torch.isfinite(loss_b)
    assert model_b.encoder.grad is not None and torch.isfinite(model_b.encoder.grad).all()
    max_diff = (model_a.encoder.grad - model_b.encoder.grad).abs().max().item()
    assert torch.allclose(model_a.encoder.grad, model_b.encoder.grad, atol=1e-5), f"gradients diverge: max diff {max_diff}"


def test_different_iteration_counts_still_work():
    config = _tiny_config()
    idx = torch.randint(0, config.vocab_size, (2, 10))
    for n_iterations in (2, 6, 8):
        torch.manual_seed(13)
        model = BDHVB(config)
        with torch.no_grad():
            logits_plain, _ = bdh_vb_variable_depth_forward(model, idx, n_iterations=n_iterations)
            logits_ckpt, _ = bdh_vb_variable_depth_forward_checkpointed(model, idx, n_iterations=n_iterations)
        assert torch.equal(logits_plain, logits_ckpt), f"mismatch at n_iterations={n_iterations}"


def test_gradients_flow_to_all_shared_weights():
    config = _tiny_config()
    torch.manual_seed(14)
    model = BDHVB(config)
    idx = torch.randint(0, config.vocab_size, (2, 10))
    x, y = idx[:, :-1].contiguous(), idx[:, 1:].contiguous()

    _logits, loss = bdh_vb_variable_depth_forward_checkpointed(model, x, n_iterations=4, targets=y)
    loss.backward()

    for name in ("encoder", "encoder_v", "decoder", "P", "O"):
        param = getattr(model, name)
        assert param.grad is not None, f"{name} got no gradient"
        assert torch.isfinite(param.grad).all(), f"{name} has non-finite gradient"
        assert float(param.grad.norm()) > 0, f"{name} gradient is exactly zero"
