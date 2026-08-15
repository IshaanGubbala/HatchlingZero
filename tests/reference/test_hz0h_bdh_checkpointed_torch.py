"""Regression tests for reference/hz0h_bdh_checkpointed_torch.py
(Phase 6, `plans/HatchlingZero_Reality_Plan.md`). Pins down:
bdh_variable_depth_forward_checkpointed computes EXACTLY the same
math as bdh_variable_depth_forward (same logits, same gradients),
confirming that checkpointing changes HOW gradients are computed
(recompute vs store), not WHAT is computed. Also verifies that
activation checkpointing is transparent to existing training loops.
"""
from __future__ import annotations

import torch

from reference.hz0h_bdh_torch import BDH, BDHConfig
from reference.hz0h_bdh_variable_depth_torch import bdh_variable_depth_forward
from reference.hz0h_bdh_checkpointed_torch import bdh_variable_depth_forward_checkpointed


def _tiny_config() -> BDHConfig:
    return BDHConfig(
        n_layer=2,
        n_embd=32,
        n_head=4,
        mlp_internal_dim_multiplier=8,
        vocab_size=32,
        dropout=0.0,
    )


def test_logits_match_uncheckpointed_forward():
    """Core correctness: checkpointed and uncheckpointed forwards produce
    identical logits (same math, only gradient compute changes)."""
    config = _tiny_config()
    torch.manual_seed(0)
    model = BDH(config)
    model.eval()
    idx = torch.randint(0, config.vocab_size, (2, 10))

    with torch.no_grad():
        logits_uncheckpointed, _ = bdh_variable_depth_forward(model, idx, n_iterations=4)
        logits_checkpointed, _ = bdh_variable_depth_forward_checkpointed(
            model, idx, n_iterations=4
        )

    assert torch.allclose(logits_uncheckpointed, logits_checkpointed, atol=1e-5), (
        f"Logit mismatch: max diff = {(logits_uncheckpointed - logits_checkpointed).abs().max()}"
    )


def test_gradients_match_uncheckpointed_forward():
    """Critical correctness: gradients w.r.t. encoder (shared weight)
    match between checkpointed and uncheckpointed after backward pass.
    This proves checkpointing doesn't alter the mathematical gradients."""
    config = _tiny_config()
    torch.manual_seed(1)
    model = BDH(config)
    model.train()
    idx = torch.randint(0, config.vocab_size, (2, 8))
    targets = torch.randint(0, config.vocab_size, (2, 8))

    # Uncheckpointed backward
    torch.manual_seed(1)
    model_ref = BDH(config)
    model_ref.train()
    with torch.set_grad_enabled(True):
        logits_ref, loss_ref = bdh_variable_depth_forward(
            model_ref, idx, n_iterations=4, targets=targets
        )
        loss_ref.backward()
    grad_encoder_ref = model_ref.encoder.grad.clone()

    # Checkpointed backward
    torch.manual_seed(1)
    model_ckpt = BDH(config)
    model_ckpt.train()
    with torch.set_grad_enabled(True):
        logits_ckpt, loss_ckpt = bdh_variable_depth_forward_checkpointed(
            model_ckpt, idx, n_iterations=4, targets=targets
        )
        loss_ckpt.backward()
    grad_encoder_ckpt = model_ckpt.encoder.grad.clone()

    # Gradients should match (within numerical precision)
    assert torch.allclose(grad_encoder_ref, grad_encoder_ckpt, atol=1e-5), (
        f"Gradient mismatch: max diff = {(grad_encoder_ref - grad_encoder_ckpt).abs().max()}"
    )


def test_loss_matches_uncheckpointed():
    """Loss values should be identical between checkpointed and
    uncheckpointed (since they use the same math)."""
    config = _tiny_config()
    torch.manual_seed(2)
    model = BDH(config)
    model.eval()
    idx = torch.randint(0, config.vocab_size, (2, 9))
    targets = torch.randint(0, config.vocab_size, (2, 9))

    with torch.no_grad():
        _logits_ref, loss_ref = bdh_variable_depth_forward(
            model, idx, n_iterations=4, targets=targets
        )
        _logits_ckpt, loss_ckpt = bdh_variable_depth_forward_checkpointed(
            model, idx, n_iterations=4, targets=targets
        )

    assert torch.allclose(loss_ref, loss_ckpt, atol=1e-5)


def test_different_iteration_counts_still_work():
    """Checkpointing should work at arbitrary iteration counts, just like
    the uncheckpointed version."""
    config = _tiny_config()
    torch.manual_seed(3)
    model = BDH(config)
    model.eval()
    idx = torch.randint(0, config.vocab_size, (2, 8))

    with torch.no_grad():
        logits_2, _ = bdh_variable_depth_forward_checkpointed(
            model, idx, n_iterations=2
        )
        logits_6, _ = bdh_variable_depth_forward_checkpointed(
            model, idx, n_iterations=6
        )

    # Different iteration counts must produce different outputs
    assert not torch.allclose(logits_2, logits_6, atol=1e-3)


def test_gradients_flow_for_arbitrary_iteration_count():
    """Gradients should flow back through shared weights even with
    checkpointing at arbitrary iteration counts."""
    config = _tiny_config()
    torch.manual_seed(4)
    model = BDH(config)
    model.train()
    idx = torch.randint(0, config.vocab_size, (2, 9))
    x, y = idx[:, :-1].contiguous(), idx[:, 1:].contiguous()

    _logits, loss = bdh_variable_depth_forward_checkpointed(
        model, x, n_iterations=8, targets=y
    )
    loss.backward()

    assert model.encoder.grad is not None
    assert float(model.encoder.grad.norm()) > 0
    assert model.encoder_v.grad is not None
    assert float(model.encoder_v.grad.norm()) > 0
    assert model.decoder.grad is not None
    assert float(model.decoder.grad.norm()) > 0


def test_works_in_training_mode():
    """Checkpointing should work correctly in training mode with dropout
    and other training-only modules."""
    config = _tiny_config()
    torch.manual_seed(5)
    model = BDH(config)
    model.train()
    idx = torch.randint(0, config.vocab_size, (2, 10))

    logits, _loss = bdh_variable_depth_forward_checkpointed(
        model, idx, n_iterations=4
    )

    assert logits.shape == (2, 10, config.n_embd)
    assert torch.isfinite(logits).all()


def test_works_in_eval_mode():
    """Checkpointing should work in eval mode (no dropout effects)."""
    config = _tiny_config()
    torch.manual_seed(6)
    model = BDH(config)
    model.eval()
    idx = torch.randint(0, config.vocab_size, (2, 10))

    with torch.no_grad():
        logits, _loss = bdh_variable_depth_forward_checkpointed(
            model, idx, n_iterations=4
        )

    assert logits.shape == (2, 10, config.n_embd)
    assert torch.isfinite(logits).all()
